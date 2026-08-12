# Architecture

## System Overview

```
                            ┌─────────────────────┐
                            │   API Server :8000   │
                            │  (FastAPI + uvicorn) │
                            └──────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │  Central Scheduler   │
                            │    + Adaptive Router │
                            └───┬──────────────┬───┘
                                │              │
                    ┌───────────▼──┐      ┌───▼───────────┐
                    │ Prefill GPU 0│      │  Decode GPU 1  │
                    │              │ NV   │                │
                    │ ┌──────────┐ │ Link │ ┌────────────┐ │
                    │ │ Chunked  │ │ ────►│ │ CB Decode  │ │
                    │ │ Prefill  │ │      │ │   Loop     │ │
                    │ └────┬─────┘ │      │ └─────┬──────┘ │
                    │      │       │      │       │        │
                    │ ┌────▼─────┐ │      │ ┌─────▼──────┐ │
                    │ │  State   │ │      │ │  State     │ │
                    │ │ Extract  │ │      │ │  Receive   │ │
                    │ └────┬─────┘ │      │ └─────┬──────┘ │
                    │      │       │      │       │        │
                    │ ┌────▼─────┐ │      │ ┌─────▼──────┐ │
                    │ │  Async   │ │      │ │ Dual-State │ │
                    │ │ Transfer │─┼──────┼►│  Memory    │ │
                    │ └──────────┘ │      │ └────────────┘ │
                    └──────────────┘      └────────────────┘
```

## Request Lifecycle

1. **API receives request** → Central Scheduler assigns a request ID
2. **Adaptive Router decides** → Collocated or PD-disaggregated based on prompt length and decode load
3. **Prefill Engine (GPU 0)** runs chunked prefill, extracts dual-state (KV cache + recurrent states)
4. **Transfer Pipeline** asynchronously migrates state to GPU 1 via NVLink/PCIe P2P
5. **Decode Engine (GPU 1)** runs continuous batching decode loop, serves tokens back via streaming or batched response

## Project Structure

```
HydraServe/
├── hydraserve/
│   ├── config.py                  # Central config: ModelSpec, TransferConfig, CacheConfig
│   ├── model/                     # Model adapters
│   │   ├── adapter.py             # Abstract ModelAdapter interface
│   │   ├── qwen3_5.py             # Qwen3.5 (4B/9B) with GDN + GQA
│   │   └── qwen3_6.py             # Qwen3.6 (27B) support
│   ├── kernels/                   # Custom Triton kernels
│   │   ├── gdn_fused.py           # GDN delta-rule fused kernel
│   │   ├── paged_attention.py     # PagedAttention decode
│   │   └── rmsnorm.py             # Fused RMSNorm
│   ├── cache/                     # Dual-state memory management
│   │   ├── block_manager.py       # PagedAttention block allocator
│   │   ├── state_pool.py          # Linear-attention state pool
│   │   ├── kv_quantizer.py        # KIVI INT4 KV quantization
│   │   └── prefix_cache.py        # Radix-tree prefix caching
│   ├── transfer/                  # GPU-to-GPU state migration
│   │   ├── backend.py             # TransferBackend abstraction
│   │   ├── nvlink_transfer.py     # NVLink backend
│   │   ├── pcie_p2p_transfer.py   # PCIe P2P backend
│   │   ├── intra_gpu_transfer.py  # MPS zero-copy backend
│   │   ├── descriptor.py          # StateTransferDescriptor
│   │   └── pipeline.py            # Layer-level async pipeline
│   ├── engine/                    # Inference engines
│   │   ├── prefill_engine.py      # GPU 0: chunked prefill + state extraction
│   │   ├── decode_engine.py       # GPU 1: continuous batching decode
│   │   ├── scheduler.py           # Central request scheduler
│   │   └── chunked_prefill.py     # Chunked prefill scheduling
│   ├── router/                    # Adaptive request routing
│   │   ├── adaptive_router.py     # Route decision per request
│   │   ├── cost_model.py          # Latency estimation model
│   │   └── profiler.py            # Micro-benchmark calibration
│   ├── serve/                     # API server layer
│   │   ├── api_server.py          # FastAPI OpenAI-compatible server
│   │   ├── protocol.py            # Pydantic request/response models
│   │   └── serve.py               # Entry point + server factory
│   └── benchmark/                 # Benchmarking suite
│       ├── run_benchmark.py       # Test orchestration
│       ├── datasets.py            # Dataset loaders
│       ├── metrics.py             # Metric collection
│       └── plot.py                # Visualization
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Technical Details

### N-1 Truncation

Linear attention maintains a recurrent state that encodes the entire prefix. After prefill processes tokens `[0, N]`, the recurrent state represents the prefix up to token `N-1`. To advance the state boundary to token `N` before regular decode begins, HydraServe replays token `N` in a single-step forward pass (< 5 ms). This is transparent to the user and guarantees mathematically identical outputs to collocated serving.

### First-Token Seeding

When enabled, the prefill engine samples the first output token immediately after prefill completes. This sampled token is packed into the transfer descriptor, eliminating the need for N-1 replay on the decode side and reducing end-to-end latency by one decode step.

### Layer-Level Transfer Pipeline

The transfer pipeline overlaps KV cache and recurrent state migration with prefill computation at the layer level. As soon as layer *i*'s states are extracted, they begin transferring while layer *i+1*'s prefill runs. With NVLink's 112 GB/s bandwidth and depth-2 CUDA stream pipelining, 32K context transfers complete in 9 ms — entirely hidden behind the remaining prefill work.

### Transfer Backends

| Backend | Bandwidth | Transfer Mode | Pipeline | 32K Latency |
|---------|-----------|---------------|----------|-------------|
| NVLink | 112 GB/s | Full BF16 | Yes | 9 ms |
| PCIe P2P | ~12–16 GB/s | INT4 Quantized | Yes | 29 ms |
| PCIe SHM | ~8–10 GB/s | INT4 Quantized | No | 43 ms |
| RDMA | 25 GB/s | Full BF16 | Yes | 40 ms |
| Intra-GPU MPS | HBM bandwidth | Zero-copy | N/A | 0 ms |

### Prefix Cache

A radix-tree-based prefix cache avoids redundant KV computation for shared prompt prefixes. When multiple requests share a system prompt or few-shot examples, the prefix is computed once and shared across requests, reducing prefill latency and memory overhead.

### Model Specifications

| Model | Hidden Size | Layers | Full-Attn Interval | Max Context |
|-------|------------|--------|-------------------|-------------|
| Qwen3.5-4B | 2560 | 32 | 4 | 128K |
| Qwen3.5-9B | 4096 | 32 | 4 | 128K |
| Qwen3.6-27B | 5120 | 64 | 4 | 128K |
