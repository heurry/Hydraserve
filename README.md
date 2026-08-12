# HydraServe

> 面向混合注意力架构 LLM 的 Prefill-Decode 分离推理引擎

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 概述

HydraServe 从零实现面向混合注意力架构（Qwen3.5/3.6：Gated Delta Network + GQA）的 **Prefill-Decode 分离推理引擎**。处理 full attention KV Cache 与 linear attention 循环状态的**双状态传输**，支持 4B / 9B / 27B 多模型，在双卡 RTX 3090 上对比 Collocated / DP / TP / PD 分离四种策略。

### 核心亮点

- **双状态异构迁移**：full attention KV Cache（block 粒度，可 INT4 量化传输）+ linear attention 循环状态（FP32，不可量化，必须整体传输）
- **层级别异步流水线**：NVLink 112 GB/s 下 32K 上下文传输 9ms，100% 隐藏在 prefill 计算时间内
- **TransferBackend 抽象层**：支持 NVLink / PCIe P2P / SHM / Intra-GPU MPS / RDMA 多后端自适应
- **Triton 自定义 kernel**：GDN fused delta rule kernel（状态留在 SRAM 避免 32GB HBM IO）+ paged attention decode kernel
- **自适应路由**：硬件实测 cost model 按 prompt 长度和 decode 负载自动选择 Collocated 或 PD 分离路径

## 快速开始

### 环境要求

- 2x NVIDIA RTX 3090 (24GB VRAM)
- NVLink Bridge 可选（有则 112 GB/s 全量 BF16 传输，无则 PCIe P2P + INT4 KV 量化）
- Python 3.10+, CUDA 12.1+

### 安装

```bash
# 创建环境
conda create -n hydraserve python=3.10 -y
conda activate hydraserve

# 安装依赖
pip install -r requirements.txt

# 验证硬件
bash scripts/verify_nvlink.sh
```

### 下载模型

```bash
# Qwen3.5-4B (推荐用于快速测试)
bash scripts/download_model.sh 4B

# 或手动指定路径
export MODEL_DIR=/mnt/nvme-data/models/LLM_model
```

### 启动服务

```bash
# PD 分离模式 (默认)
python -m hydraserve.serve.serve \
    --model /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
    --model-name Qwen3.5-4B \
    --mode pd_disaggregated \
    --prefill-gpu 0 --decode-gpu 1

# Collocated 模式 (单卡)
python -m hydraserve.serve.serve \
    --model /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
    --mode collocated
```

### API 使用

```bash
# Chat Completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# 健康检查
curl http://localhost:8000/health
```

## 项目架构

```
HydraServe/
├── hydraserve/
│   ├── config.py              # 全局配置 + ModelSpec
│   ├── model/                  # 模型适配器
│   │   ├── adapter.py          # ModelAdapter 抽象接口
│   │   ├── qwen3_5.py          # Qwen3.5 (4B/9B) 实现
│   │   └── qwen3_6.py          # Qwen3.6 (27B) 实现
│   ├── kernels/                # Triton 自定义 kernel
│   │   ├── gdn_fused.py        # GDN delta rule fused kernel
│   │   ├── paged_attention.py  # PagedAttention decode kernel
│   │   └── rmsnorm.py          # RMSNorm kernel
│   ├── cache/                  # 双状态内存管理
│   │   ├── block_manager.py    # PagedAttention block 分配器
│   │   ├── state_pool.py       # Linear attention 状态池
│   │   ├── kv_quantizer.py     # KIVI INT4 KV 量化
│   │   └── prefix_cache.py     # Radix tree 前缀缓存
│   ├── transfer/               # 传输层
│   │   ├── backend.py          # TransferBackend 抽象 + NVLink/PCIeP2P/SHM/IntraGPU/RDMA
│   │   ├── descriptor.py       # StateTransferDescriptor
│   │   └── pipeline.py         # 层级别异步传输流水线
│   ├── engine/                 # 推理引擎
│   │   ├── prefill_engine.py   # Prefill 引擎 (GPU 0)
│   │   ├── decode_engine.py    # Decode 引擎 (GPU 1)
│   │   ├── scheduler.py        # CentralScheduler
│   │   └── chunked_prefill.py  # Chunked prefill 调度器
│   ├── router/                 # 自适应路由
│   │   ├── adaptive_router.py  # 路由决策器
│   │   ├── cost_model.py       # 延迟估算模型
│   │   └── profiler.py         # 微基准测试
│   ├── serve/                  # API 服务
│   │   ├── api_server.py       # FastAPI OpenAI 兼容
│   │   ├── protocol.py         # Pydantic 协议定义
│   │   └── serve.py            # 主入口
│   └── benchmark/              # 基准测试
│       ├── run_benchmark.py    # 测试编排
│       ├── datasets.py         # 数据集加载器
│       ├── metrics.py          # 指标收集
│       └── plot.py             # 可视化
├── tests/                      # 单元测试
│   ├── test_kernels.py         # Kernel 测试
│   ├── test_transfer.py        # 传输层测试
│   └── test_e2e.py             # 端到端测试
├── scripts/                    # 辅助脚本
│   ├── verify_nvlink.sh        # 硬件拓扑验证
│   ├── download_model.sh       # 模型下载
│   └── run_baseline.sh         # vLLM baseline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.md                     # 详细设计文档
```

## Benchmark 配置

| 配置 | 说明 | GPU 数 | 角色 |
|------|------|--------|------|
| A: 1-GPU Collocated | 单卡 prefill+decode | 1 | 参考 |
| B: DP | 两卡各跑独立实例 | 2 | 主对比组 |
| C: TP=2 | vLLM tensor parallelism | 2 | 主对比组 |
| D: PD 分离 | HydraServe GPU 0 prefill, GPU 1 decode | 2 | **主对比组** |
| E: vLLM unified | vLLM 单实例 | 1 | 参考 |
| F: intra-GPU PD (MPS) | GPU 0 MPS 共享 + GPU 1 独立 | 2 | 扩展对比 |

## 传输策略

| 后端 | 带宽 | TransferMode | 层级别流水线 | 32K 传输时间 |
|------|------|-------------|-------------|-------------|
| NVLink | 112 GB/s | FULL_TRANSFER | 是 | 9ms |
| PCIe P2P | ~12-16 GB/s | QUANTIZED_TRANSFER | 是 | 29ms |
| PCIe SHM | ~8-10 GB/s | QUANTIZED_TRANSFER | 否 | 43ms |
| RDMA | 25 GB/s | FULL_TRANSFER | 是 | 40ms |
| Intra-GPU MPS | 内存带宽 | INTRA_GPU | N/A | 0ms |

## 关键指标

| 指标 | 定义 | 预期 PD 分离改善 |
|------|------|-----------------|
| P50 TTFT | 首 token 延迟中位数 | 持平或略优 |
| P99 TTFT | 首 token 延迟 P99 | -20~40% |
| P50 TPOT | 每 token 延迟中位数 | 持平 |
| P99 TPOT | 每 token 延迟 P99 | -50~80% |
| 吞吐 (tok/s) | 总 token 吞吐 | 0~20% |
| GSM8K 精度 | exact_match | 无损 |

## 开发里程碑

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | 模型理解 + 环境搭建 | ✅ |
| 1 | 推理引擎 + Triton kernels | ✅ |
| 2 | 双状态内存管理 + ModelAdapter | ✅ |
| 3 | 传输层 + 双状态序列化 | ✅ |
| 4 | PD 分离核心 + N-1 truncation | ✅ |
| 5 | Continuous batching + chunked prefill | ✅ |
| 6 | 自适应路由 + 27B 适配 | ✅ |
| 7 | Benchmark + 对比实验 | 🚧 |
| 8 | intra-GPU MPS 模式 | 🚧 |

## License

MIT
