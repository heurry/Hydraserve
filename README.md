# HydraServe

<div align="center">

**Prefill-Decode Disaggregated Inference Engine for Hybrid-Attention LLMs**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) • [Quick Start](#quick-start) • [API](#api) • [Benchmarks](#benchmarks) • [Documentation](#documentation)

<img src="https://img.shields.io/badge/models-Qwen3.5%20%7C%20Qwen3.6-ff6b6b.svg" alt="Supported Models">

</div>

---

## Overview

HydraServe is a high-performance inference engine that implements **Prefill-Decode (PD) disaggregation** for hybrid-attention LLMs like Qwen3.5/3.6. Unlike traditional full-attention models, hybrid-attention architectures mix full-attention and linear-attention layers — requiring coordinated migration of both KV caches and recurrent states between GPUs. HydraServe handles this dual-state complexity with a layer-level asynchronous pipeline, achieving **50–80% reduction in P99 TPOT** on dual RTX 3090 GPUs.

### Why PD Disaggregation?

Hybrid-attention models (Gated Delta Network + GQA) interleave full-attention layers every 4th layer with linear-attention layers. When separating prefill and decode across GPUs:

- **Full-attention KV Cache** must transfer between GPUs (GBs at long context, INT4-quantizable)
- **Linear-attention Recurrent State** must also transfer (dense FP32, cannot be quantized)

HydraServe migrates both state types simultaneously through a transfer pipeline that overlaps entirely with computation, making PD disaggregation practical for hybrid-attention architectures.

## Features

- **PD Disaggregated Serving** — Physically separate prefill (GPU 0) and decode (GPU 1) to eliminate compute/memory interference
- **Dual-State Migration** — Coordinated transfer of full-attention KV cache (block-granularity, INT4-capable) and linear-attention recurrent state (FP32, SRAM-optimized)
- **Layer-Level Async Pipeline** — 9ms transfer for 32K context via NVLink, 100% overlapped with prefill computation
- **Pluggable Transfer Backends** — NVLink, PCIe P2P, Shared Memory, RDMA, and Intra-GPU MPS
- **Custom Triton Kernels** — GDN fused delta rule kernel, PagedAttention decode, fused RMSNorm
- **Adaptive Router** — Hardware-profiled cost model routes each request to collocated or PD-disaggregated path
- **OpenAI-Compatible API** — Drop-in replacement with streaming (SSE) support
- **Dual RTX 3090 Friendly** — Designed for consumer GPUs with or without NVLink bridge

## Quick Start

### Prerequisites

- 2× NVIDIA RTX 3090 (24 GB) or equivalent
- NVLink bridge (optional; falls back to PCIe P2P with INT4 quantization)
- Python 3.10+, CUDA 12.1+

### Installation

```bash
git clone https://github.com/heurry/Hydraserve.git
cd Hydraserve

conda create -n hydraserve python=3.10 -y
conda activate hydraserve

pip install -r requirements.txt
bash scripts/verify_nvlink.sh
```

### Download a Model

```bash
bash scripts/download_model.sh 4B    # Qwen3.5-4B for quick testing
# Or: export MODEL_DIR=/path/to/models
```

### Launch

```bash
# PD disaggregated mode (recommended)
python -m hydraserve.serve.serve \
    --model /path/to/Qwen3.5-4B \
    --model-name Qwen3.5-4B \
    --mode pd_disaggregated \
    --prefill-gpu 0 --decode-gpu 1

# Collocated mode (single GPU)
python -m hydraserve.serve.serve \
    --model /path/to/Qwen3.5-4B \
    --mode collocated
```

### Docker

```bash
# Update the model path in docker-compose.yml, then:
docker compose up -d
```

### Try It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## API

OpenAI-compatible REST API. See the [OpenAI API docs](https://platform.openai.com/docs/api-reference) for full parameter reference.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming & non-streaming) |
| `POST` | `/v1/completions` | Legacy text completions |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Runtime statistics |

## Benchmarks

### Serving Configurations

| Config | Description | GPU Count |
|--------|-------------|-----------|
| Collocated | Single GPU prefill + decode | 1 |
| Data Parallel | Two independent instances | 2 |
| TP (vLLM) | Tensor parallelism (vLLM) | 2 |
| **PD Disaggregated** | **HydraServe GPU 0 prefill, GPU 1 decode** | **2** |
| Intra-GPU MPS | MPS-shared + independent GPU | 2 |

### Key Metrics (vs Collocated Baseline)

| Metric | PD Improvement |
|--------|---------------|
| P50 TTFT | On par |
| P99 TTFT | −20~40% |
| P50 TPOT | On par |
| P99 TPOT | −50~80% |
| Throughput | 0~20% |
| GSM8K Accuracy | Lossless |

> PD disaggregation excels on tail latencies under mixed workloads. By isolating decode from prefill interference, P99 TPOT drops dramatically.

Run the full benchmark suite:

```bash
python -m hydraserve.benchmark.run_benchmark \
    --model /models/Qwen3.5-9B-AWQ \
    --configs all \
    --output-dir ./results
```

## Supported Models

| Model | Size | Layers | Architecture | Max Context |
|-------|------|--------|-------------|-------------|
| Qwen3.5-4B | 4B | 32 | GDN + GQA | 128K |
| Qwen3.5-9B | 9B | 32 | GDN + GQA | 128K |
| Qwen3.6-27B | 27B | 64 | GDN + GQA | 128K |

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to model weights |
| `--model-name` | `Qwen3.5-9B` | `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.6-27B` |
| `--mode` | `pd_disaggregated` | `collocated`, `pd_disaggregated`, `dp` |
| `--prefill-gpu` | `0` | GPU for prefill engine |
| `--decode-gpu` | `1` | GPU for decode engine |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | API port |
| `--precision` | `int4` | `int4` or `bf16` |
| `--chunk-size` | `4096` | Tokens per prefill chunk |
| `--max-seqs` | `256` | Max concurrent sequences |

## Documentation

- [Architecture & Design](docs/architecture.md) — System design, component structure, transfer backends, N-1 truncation

## Development Status

| Phase | Status |
|-------|--------|
| Core inference engine + Triton kernels | ✅ |
| Dual-state memory management | ✅ |
| Transfer layer + state serialization | ✅ |
| PD disaggregation + N-1 truncation | ✅ |
| Continuous batching + chunked prefill | ✅ |
| Adaptive routing + 27B support | ✅ |
| Benchmarking + comparison experiments | 🚧 |
| Intra-GPU MPS mode | 🚧 |

### Running Tests

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please open an issue to discuss major changes before submitting a PR.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes + run tests (`python -m pytest tests/`)
4. Commit and push (`git commit -m 'Add amazing feature'`)
5. Open a Pull Request

## Citation

```bibtex
@software{hydraserve2025,
  author = {HydraServe Contributors},
  title = {HydraServe: Prefill-Decode Disaggregated Inference Engine for Hybrid-Attention LLMs},
  url = {https://github.com/heurry/Hydraserve},
  year = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention and continuous batching
- [SGLang](https://github.com/sgl-project/sglang) — Radix attention design
- [Qwen](https://github.com/QwenLM/Qwen) — Open hybrid-attention architectures
- [Triton](https://github.com/triton-lang/triton) — GPU kernel language

---

<div align="center">
  <sub>Built for the GPU poor. Dual RTX 3090 friendly.</sub>
</div>
