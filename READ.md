# Swift 源码阅读

## 前言

个人认为很优秀的大模型训练框架，且不断进行迭代完善，准备好好阅读学习一下，以后直接跟踪最新进展

## 目录

```bash
.
├── asset # 群二维码
├── CODE_OF_CONDUCT.md # 代码道德准则
├── CONTRIBUTING_CN.md # 如何进行贡献提PR
├── CONTRIBUTING.md # 如何进行贡献提PR（英文版）
├── docs # 各种文档
├── examples # 示例运行脚本
├── LICENSE # Apache License 2.0
├── Makefile # 构建测试Python项目
├── MANIFEST.in # 指定在创建Python分发包时应包含哪些文件
├── README_CN.md # 介绍文档
├── README.md # 介绍文档（英文）
├── requirements # 依赖库的分文件详细说明
├── requirements.txt # 指向最主要的依赖库文件
├── resources # 仓库首图
├── scripts # 一些方便的测试脚本文件
├── setup.cfg # 配置Python项目的各种工具
├── setup.py # 安装环境的Python脚本
├── swift # 主代码
├── tests # 测试文件
└── tools # 一个merge LoRA权重的小工具
```

## 环境安装

安装为可编辑模式（相当于将这个文件夹作为库来安装，对文件夹的修改直接反映到包中）
```bash
pip install -e '.[llm]'
```

## LLM微调

### 入口

```bash
examples/pytorch/llm/llm_sft.py
```

```python
if __name__ == '__main__':
    output = sft_main()

sft_main = get_sft_main(SftArguments, llm_sft)

get_main(args, llm)

```

设置种子

```python
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
```

获取分布式的训练配置

```python
def get_dist_setting() -> Tuple[int, int, int, int]:
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 1))
    return rank, local_rank, world_size, local_world_size
```

设置GPU最大占用
```python
for device_id in range(torch.cuda.device_count()):
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=device_id)
```

设置device map max_length 和 flash attention
```python
# Loading Model and Tokenizer
if is_deepspeed_zero3_enabled():
    model_kwargs = {'device_map': None}
elif is_torch_npu_available():
    model_kwargs = {'device_map': local_rank if local_rank >= 0 else 0}
else:
    model_kwargs = {'low_cpu_mem_usage': True}
    if is_dist() and not is_ddp_plus_mp():
        model_kwargs['device_map'] = {'': local_rank}
    elif not use_torchacc():
        model_kwargs['device_map'] = 'auto'
```

## 其他文件

### swift

### docs

### requirements

### scripts

### tests