import os
import json
import random
import torch
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch import dtype as Dtype
from torch.nn import Module

from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from pandas import DataFrame

from torch.nn import CrossEntropyLoss

from datasets import Dataset as HfDataset

from torch.utils.data import Dataset

from template import TEMPLATE_MAPPING, get_template, Template

from transformers.modeling_utils import unwrap_model

from torch import Tensor, nn
from transformers.trainer_utils import EvalPrediction
from transformers import Seq2SeqTrainer as HfSeq2SeqTrainer

from typing import List, Optional, Tuple, Dict, Any, Union, Callable, Literal
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Seq2SeqTrainer(HfSeq2SeqTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc = torch.tensor(0.0).to(self.args.device)

    def train(self, *args, **kwargs) -> torch.Tensor:
        return super().train(*args, **kwargs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def compute_scaled_loss(
        self, labels: torch.Tensor, lm_logits: torch.Tensor, loss_scale: torch.Tensor
    ) -> torch.Tensor:
        device = lm_logits.device
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_scale = loss_scale[..., 1:]
        # Save memory
        masks = shift_labels != -100
        shift_logits = shift_logits[masks]
        shift_labels = shift_labels[masks].to(device)
        shift_scale = shift_scale[masks].to(device)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits, shift_labels)
        loss = shift_scale * loss
        return loss.mean()

    def compute_loss(self, model, inputs, return_outputs=None):
        if not hasattr(self, "_custom_metrics"):
            self._custom_metrics = {}

        labels = None
        loss_scale = None
        if "loss_scale" in inputs:
            labels = inputs.pop("labels")
            loss_scale = inputs.pop("loss_scale")

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")

        outputs = model(**inputs)
        if loss_scale is not None:
            outputs["loss"] = self.compute_scaled_loss(
                labels, outputs.logits, loss_scale
            )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None and loss_scale is None:
            unwrapped_model = unwrap_model(model)
            model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        preds = outputs.logits.argmax(dim=2)[..., :-1]
        if labels is None:
            labels = inputs["labels"]
        labels = labels[..., 1:]
        masks = labels != -100
        acc_strategy = getattr(self.args, "acc_strategy", "token")
        acc: Optional[Tensor] = None
        if preds.shape != labels.shape:
            pass
        elif acc_strategy == "sentence":
            acc_list = []
            for i, m in enumerate(masks):
                acc_list.append(
                    torch.all(preds[i, m] == labels[i, m]).to(torch.int64).item()
                )
            acc = torch.tensor(acc_list, device=preds.device).float().mean()
        else:
            acc = (
                (
                    torch.masked_select(preds, masks)
                    == torch.masked_select(labels, masks)
                )
                .float()
                .mean()
            )
        if model.training and acc is not None:
            if "acc" not in self._custom_metrics:
                self._custom_metrics["acc"] = self._acc
            self._custom_metrics["acc"] = (
                self._custom_metrics["acc"]
                + acc / self.args.gradient_accumulation_steps
            )
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset):
        return super().get_eval_dataloader(eval_dataset)

    def get_test_dataloader(self, test_dataset):
        return super().get_test_dataloader(test_dataset)


class LLMDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            data, _ = self.data[idx]
            return data
        elif isinstance(idx, str):
            return [d[0][idx] for d in self.data]
        else:
            raise ValueError(f"idx: {idx}")

    def select(self, idx_list: List[int]) -> "LLMDataset":
        data = [self.data[i] for i in idx_list]
        return self.__class__(data)

    def __len__(self) -> int:
        return len(self.data)


def get_dist_setting() -> Tuple[int, int, int, int]:
    rank = int(os.getenv("RANK", -1))
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))
    return rank, local_rank, world_size, local_world_size


def seed_everything(seed: int = None) -> int:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # torch.mlu.manual_seed_all(seed)
    # torch.npu.manual_seed_all(seed)
    # torch.xpu.manual_seed_all(seed)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def select_dtype(dtype: str) -> Tuple[Optional[Dtype], bool, bool]:
    dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_mapping[dtype]

    if torch_dtype == torch.float16:
        dtype = "fp32"
        torch_dtype = torch.float32
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            print("...")
            break
        print(
            f"[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}"
        )


def get_model_info(model: Module, name: Optional[str] = None) -> str:
    if name is None:
        name = model.__class__.__name__

    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())

    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = (
        f"{name}: "
        f"{n_params:.4f}M Params ({n_grads:.4f}M Trainable "
        f"[{100 * n_grads / n_params:.4f}%]), "
        f"{n_buffers:.4f}M Buffers."
    )
    return s


def get_model_tokenizer(
    torch_dtype: str,
    model_dir: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:

    
    print(f"Setting torch_dtype: {torch_dtype}")
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.torch_dtype = torch_dtype

    with nullcontext():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset_from_local(dataset_path: str) -> Optional[HfDataset]:
    dict_list: List[Any] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            dict_list.append(json.loads(line))

    data_dict: Dict[str, List[Any]] = {}
    for i, obj in enumerate(dict_list):
        for k, v in obj.items():
            if k not in data_dict:
                data_dict[k] = [None] * i
            data_dict[k].append(v)
        for k in set(data_dict.keys()) - set(obj.keys()):
            data_dict[k].append(None)
    dataset = HfDataset.from_dict(DataFrame.from_dict(data_dict).to_dict(orient="list"))
    return dataset


def dataset_map(
    dataset: HfDataset, map_func: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> Optional[LLMDataset]:
    data = []
    for d in tqdm(dataset):
        d = map_func(d)
        data.append(d)
    data = [d for d in data if d is not None]
    return LLMDataset(data)


def safe_tokenizer_decode(
    tokenizer: PreTrainedTokenizerBase, input_ids: List[int], **tokenizer_kwargs
) -> str:
    if len(input_ids) == 0:
        return ""
    result_str = ""
    for i in range(len(input_ids)):
        if i == 0:
            if input_ids[i] < 0:
                s = 0
            else:
                e = 0
            continue
        if input_ids[i] < 0 and input_ids[i - 1] >= 0:
            s = i
            result_str += tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)
        if input_ids[i] >= 0 and input_ids[i - 1] < 0:
            e = i
            result_str += f"[-100 * {e - s}]"
    if input_ids[-1] < 0:
        result_str += f"[-100 * {len(input_ids) - s}]"
    else:
        result_str += tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
    return result_str


def print_example(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    input_ids = example.get("input_ids")
    labels = example.get("labels")
    if input_ids is not None:
        print(f"[INPUT_IDS] {input_ids}")
        input_str = safe_tokenizer_decode(tokenizer, input_ids, **tokenizer_kwargs)
        print(f"[INPUT] {input_str}")
    if labels is not None:
        print(f"[LABLES_IDS] {labels}")
        labels_str = safe_tokenizer_decode(tokenizer, labels, **tokenizer_kwargs)
        print(f"[LABLES] {labels_str}")


def compute_acc_metrics(
    eval_prediction: EvalPrediction,
    acc_strategy: Literal["token", "sentence"] = "token",
) -> Dict[str, Tensor]:
    labels = eval_prediction.label_ids[..., 1:]
    predictions = eval_prediction.predictions[..., :-1]
    if predictions.shape != labels.shape:
        return {}
    masks = labels != -100
    if acc_strategy == "sentence":
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(predictions[i, m] == labels[i, m]))
        acc = np.mean(np.array(acc_list))
    else:
        acc = np.mean((predictions[masks] == labels[masks]).astype(np.float64))
    return {"acc": acc}


def preprocess_logits_for_metrics(logits: Tensor, labels: Tensor) -> Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM训练极简框架")

    parser.add_argument(
        "--dtype", type=str, default="bf16", help="dtype"
    )
    
    parser.add_argument("--epochs", type=int, default=1, help="Train Epochs")
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Eval Batch Size"
    )
    parser.add_argument("--eval_steps", type=int, default=20, help="Eval Steps")
    parser.add_argument(
        "--gpu_memory_fraction", type=float, default=1.0, help="GPU ratio"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning Rate"
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", help="lr_scheduler_type"
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max Length")
    parser.add_argument("--model_dir", type=str, required=True, help="Model dir")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dir")
    parser.add_argument("--template_type", type=str, required=True, help="Template")
    parser.add_argument("--train_data", type=str, required=True, help="Train Data dir")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Train Batch Size"
    )
    parser.add_argument("--save_steps", type=int, default=20, help="Save Steps")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--use_flash_attn", action="store_true", help="")
    parser.add_argument("--val_data", type=str, required=True, help="Val Data dir")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup Ratio")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight Decay")

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)

    rank, local_rank, world_size, local_world_size = get_dist_setting()
    print(
        f"rank: {rank}, local_rank: {local_rank}, "
        f"world_size: {world_size}, local_world_size: {local_world_size}"
    )

    for device_id in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device=device_id
        )

    # if is_deepspeed_zero3_enabled():
    #     model_kwargs = {'device_map': None}
    # else:
    model_kwargs = {"low_cpu_mem_usage": True}
    # if is_dist() and not is_ddp_plus_mp():
    model_kwargs["device_map"] = {"": local_rank}
    # else:
    #     model_kwargs['device_map'] = 'auto'

    kwargs = {"max_length": args.max_length}
    kwargs["use_flash_attn"] = args.use_flash_attn

    torch_dtype, fp16, bf16 = select_dtype(args.dtype)

    model, tokenizer = get_model_tokenizer(
        torch_dtype,
        args.model_dir,
        model_kwargs,
    )
    print(get_model_info(model))

    model.config.use_cache = False  # fix transformers==4.36
    model.enable_input_require_grads()

    train_dataset = load_dataset_from_local(args.train_data)
    val_dataset = load_dataset_from_local(args.val_data)

    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    template: Template = get_template(
        template_type=args.template_type,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_dataset = dataset_map(train_dataset, template.encode)
    val_dataset = dataset_map(val_dataset, template.encode)

    td0, tkwargs0 = train_dataset.data[0]
    print_example(td0, tokenizer, tkwargs0)

    data_collator = partial(template.data_collator, padding_to=None)

    training_args = Seq2SeqTrainingArguments(
        bf16=bf16,
        dataloader_num_workers=1,
        ddp_backend="nccl",
        ddp_broadcast_buffers=False,
        ddp_find_unused_parameters=False,
        do_eval=False,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        fp16=fp16,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        greater_is_better=False,
        learning_rate=args.learning_rate,
        logging_first_step=True,
        logging_steps=1,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=0.5,
        metric_for_best_model="loss",
        num_train_epochs=args.epochs,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        remove_unused_columns=False,
        save_on_each_node=True,
        save_only_model=True,
        save_steps=args.save_steps,
        sortish_sampler=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )
    print(training_args)

    trainer_kwargs = {}
    trainer_kwargs["compute_metrics"] = compute_acc_metrics
    trainer_kwargs["preprocess_logits_for_metrics"] = preprocess_logits_for_metrics

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[],
        **trainer_kwargs,
    )

    trainer.train()
