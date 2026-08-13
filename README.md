# HydraServe

HydraServe 是一个面向 Qwen3.5/Qwen3.6 混合注意力模型（GDN + GQA）的
Prefill–Decode 分离推理引擎原型。当前主线按 [`main.md`](main.md) 从零实现，
不复用仓库此前的引擎代码；旧仓库完整内容保存在
`archive/pre-implementation-2026-08-13` 分支。

## 当前进度

推理内核与首个真实模型纵切片已经完成：

- 从 Hugging Face `config.json` 动态解析混合层布局和所有状态维度；
- 4B/9B/27B 只是预置，不是模型规模白名单；
- Paged KV block allocator 与固定槽 FP32 recurrent-state pool；
- 真正的两值/byte grouped symmetric INT4 KV codec；
- 强校验的双状态传输描述符；
- `FULL`、`QUANTIZED`、`PARTIAL` 三种不同传输语义；
- 进程内测试后端与 POSIX shared-memory PARTIAL 后端；
- chunked prefill、first-token seeding、请求状态机、自适应路由；
- decode 端 PARTIAL KV 重算及状态安装的端到端 CPU 测试。
- 直接加载 sharded safetensors 的独立 Qwen text runtime；
- FlashAttention varlen GQA prefill；
- 自写 Triton RMSNorm、gated RMSNorm、causal conv、GDN recurrent rule；
- 自写 Triton Paged Attention 和 Paged KV scatter；
- 支持异构上下文长度的 Continuous Batching decode executor；
- Qwen3.5-4B BF16 真实 32 层 prefill/decode GPU smoke。
- 独立 prefill/decode worker、N-1 truncation 与首 token 一致性校验；
- 真实双进程、双 GPU 的 SHM PARTIAL_TRANSFER 端到端链路；
- FULL/INT4 QUANTIZED KV 安装路径与真实物理页读取；
- CUDA P2P 后端及硬件能力检测（本机 NODE 拓扑无 peer access，自动回退 SHM）；
- 完整块粒度的 full-attention prefix radix cache（不错误缓存 GDN 状态）；
- ShareGPT、HumanEval、LongBench、WikiText-103、GSM8K 低内存数据适配器。

OpenAI-compatible API、持续运行的多进程 serving loop 和 benchmark 指标采集仍待实现。
P2P 后端已实现，但当前两卡拓扑不支持 CUDA peer access，因此不能在本机伪装为
真实 P2P 实测；层级流水线也只完成协议和单测，不宣称已在 NVLink/P2P 上验证。

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

注意：真实 Qwen GDN recurrent state 按 value heads 保存，conv state 保存完整
Q/K/V depthwise-conv 通道。因此 FP32 双状态是 4B/9B 约 53.48 MB/请求，27B
约 158.86 MB/请求；早期设计文档中的 25/50 MB 估算偏低。

## 开发与测试

当前 CPU 协议层只依赖 NumPy：

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

本机环境包含一个依赖不完整的 ROS pytest 插件，因此示例显式关闭第三方插件自动加载。
GPU kernel 测试必须在 CUDA 可见环境运行；FlashAttention 测试使用安装了
`flash-attn` 的环境单独执行。

检查本机基准数据（大文件均流式读取，LongBench 不解压）：

```bash
python -m hydraserve inspect-datasets /mnt/nvme-data/datasets/benchmark --limit 1
```

当前目录中 `wikitext-103-raw.tar.gz` 和 `wikitext-103-test.csv` 是 0 字节，
加载器固定使用有效的 `wikitext-103-test.jsonl`。

## 代码结构

```text
hydraserve/
├── config.py                 # 动态模型配置与预置
├── model/                    # safetensors loader 与独立 Qwen runtime
├── kernels/                  # reference、Triton 与 FlashAttention 边界
├── cache/                    # Paged KV、FP32 state pool、INT4 codec
├── transfer/                 # 描述符、后端、双状态 pipeline
├── engine/                   # chunked prefill、continuous batching、状态机
├── router/                   # 自适应 PD 路由
└── benchmark/                # 流式数据集适配器
```

设计目标、硬件数据和完整里程碑见 [`main.md`](main.md)。
