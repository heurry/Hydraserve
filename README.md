# HydraServe

HydraServe 是一个面向 Qwen3.5/Qwen3.6 混合注意力模型（GDN + GQA）的
Prefill–Decode 分离推理引擎原型。当前主线按 [`main.md`](main.md) 从零实现，
不复用仓库此前的引擎代码；旧仓库完整内容保存在
`archive/pre-implementation-2026-08-13` 分支。

## 当前进度

首个可验证纵切片已经完成：

- 从 Hugging Face `config.json` 动态解析混合层布局和所有状态维度；
- 4B/9B/27B 只是预置，不是模型规模白名单；
- Paged KV block allocator 与固定槽 FP32 recurrent-state pool；
- 真正的两值/byte grouped symmetric INT4 KV codec；
- 强校验的双状态传输描述符；
- `FULL`、`QUANTIZED`、`PARTIAL` 三种不同传输语义；
- 进程内测试后端与 POSIX shared-memory PARTIAL 后端；
- chunked prefill、first-token seeding、请求状态机、自适应路由；
- decode 端 PARTIAL KV 重算及状态安装的端到端 CPU 测试。

GPU 权重加载、GDN Triton kernel、Paged Attention Triton kernel 和真正的 CUDA
P2P/NVLink backend 尚未实现，接口已经为后续阶段留出。

## 模型兼容性

兼容性由架构字段决定，不由参数量或模型名字决定。一个模型需要提供：

- 每层的 `layer_types`，或可推导的 `full_attention_interval`；
- `num_hidden_layers`、`hidden_size`；
- GQA 的 attention/KV head 数和 `head_dim`；
- GDN 的 key/value head 数、head dim 和 convolution kernel；
- `mamba_ssm_dtype=float32`。

加载器会自动识别 Qwen 多模态配置中的嵌套 `text_config`。检查本机模型目录：

```bash
python -m hydraserve inspect-models /mnt/nvme-data/models/LLM_model
```

当前机器上的 4B、9B、27B BF16、27B FP8 和 27B AWQ/INT4 配置均已通过检查。
同架构的其他参数规模可以直接通过其 `config.json` 接入；若内部维度不同，缓存和
循环状态形状会自动跟随配置变化。

## 开发与测试

当前 CPU 协议层只依赖 NumPy：

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

本机环境包含一个依赖不完整的 ROS pytest 插件，因此示例显式关闭第三方插件自动加载。

## 代码结构

```text
hydraserve/
├── config.py                 # 动态模型配置与预置
├── model/                    # 模型运行时边界
├── cache/                    # KV block、FP32 state pool、INT4 codec
├── transfer/                 # 描述符、后端、双状态 pipeline
├── engine/                   # chunked prefill、decode 安装、请求状态机
└── router/                   # 自适应 PD 路由
```

设计目标、硬件数据和完整里程碑见 [`main.md`](main.md)。
