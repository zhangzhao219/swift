# Experimental environment: V100, A10, 3090
# 12GB GPU memory

NPROC_PER_NODE=2

CUDA_VISIBLE_DEVICES=2,3 \
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port 29502 \
    examples/pytorch/llm/llm_sft.py \
    --model_type gemma-2b-instruct \
    --model_id_or_path pretrained/google/gemma-2b-it \
    --sft_type full \
    --tuner_backend peft \
    --template_type gemma \
    --dtype bf16 \
    --output_dir output \
    --custom_train_dataset_path data/test.jsonl \
    --num_train_epochs 1 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.1 \
    --eval_steps 10 \
    --save_steps 20 \
    --save_total_limit 2 \
    --logging_steps 1 \
