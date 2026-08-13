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

## 2026-08-13 — GPU inference slice

Implemented and tested on 2x RTX 3090:

- direct sharded-safetensors Qwen text weight loader;
- complete 32-layer Qwen3.5-4B text forward without Transformers/vLLM model execution;
- FlashAttention varlen prefill (the explicitly permitted external kernel);
- HydraServe Triton RMSNorm, gated RMSNorm, causal convolution, recurrent GDN,
  paged KV scatter, and decode Paged Attention;
- CPU mathematical oracles and CPU/GPU numerical comparison tests;
- physical multi-layer Paged KV storage and per-request block tables;
- heterogeneous-position batched decode and in-flight failure cleanup;
- real Qwen3.5-4B BF16 single-token and multi-token GPU smoke tests.

Corrected architecture fact: GDN recurrent state uses value heads and convolution
state uses all projected Q/K/V channels. The resulting FP32 state is 53.48 MB for
4B/9B and 158.86 MB for 27B, not the earlier 25/50 MB estimate.

Next implementation slice:

1. multi-process prefill/decode workers using the tested runtime;
2. PARTIAL_TRANSFER state installation using real GPU state tensors;
3. asynchronous per-layer transfer pipeline and CUDA P2P/NVLink backend;
4. prefix cache and chunked-prefill history handling;
5. OpenAI-compatible streaming API and benchmark harness.
