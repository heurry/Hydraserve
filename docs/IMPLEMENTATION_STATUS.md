# Implementation status

## 2026-08-13 — clean-room foundation

This mainline was restarted from the design document. The previous repository
tip (`97fe3db`) is retained on `archive/pre-implementation-2026-08-13` and no
previous engine source was reused.

Implemented:

- architecture-driven Qwen hybrid config parsing, including nested `text_config`;
- dual-state descriptors with invariant checks;
- fixed FP32 recurrent-state slots and paged KV block identities;
- packed grouped symmetric INT4 KV representation;
- in-memory and POSIX shared-memory transfer backends;
- PARTIAL transfer vertical slice with decode-side KV recomputation;
- chunked prefill boundaries, first-token seeding, routing, and lifecycle states.

Next implementation slice:

1. safetensors weight index and layer-wise Qwen checkpoint loader;
2. reference PyTorch GDN and attention forward passes;
3. Triton GDN and paged-attention kernels with reference comparisons;
4. CUDA P2P/NVLink backend and asynchronous per-layer pipeline;
5. OpenAI-compatible serving loop and continuous decode batching.
