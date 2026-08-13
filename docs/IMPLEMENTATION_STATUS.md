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

## 2026-08-13 — PD worker and benchmark-data slice

Implemented:

- runtime-state codec between per-layer GPU tensors and contiguous FP32 transfer regions;
- N-1 GDN-state boundary, first-token seeding, and decode-side replay verification;
- real two-process/two-GPU Qwen3.5-4B SHM PARTIAL_TRANSFER test;
- full-attention KV gather/install for FULL and grouped-INT4 QUANTIZED modes;
- capability-checked CUDA P2P backend with an explicit failure when peer access is absent;
- manifest-first per-layer pipeline protocol (unit tested; not hardware validated here);
- block-aligned full-attention radix prefix cache with reference-safe LRU eviction;
- streaming adapters for GSM8K, gzip HumanEval, 673 MB ShareGPT JSON arrays,
  WikiText-103 JSONL, and all LongBench members directly inside its ZIP.

Validation on this host: 32 CPU tests pass; 62 CUDA/Triton tests pass with three
intentional skips; the opt-in real two-GPU PD test passes. The two RTX 3090s are
on a NODE topology and report no CUDA peer access, so the P2P implementation is
not claimed as a real hardware result.

Next implementation slice:

## 2026-08-13 — serving and benchmark slice

Implemented:

- persistent admission/decode loop with dynamic Continuous Batching, cancellation,
  EOS/length termination, and failure cleanup;
- direct `tokenizer.json` loading and incremental byte-safe decoding;
- `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, and SSE;
- tokenizer-aware concurrent benchmark runner with TTFT, TPOT, latency,
  request throughput, output-token throughput, and P50/P95/P99;
- command-line `serve`, `benchmark`, and `inspect-datasets` entry points.

Real validation: Qwen3.5-4B returned ` Paris.` for a two-token completion through
the HTTP API. A two-request GSM8K CLI smoke completed successfully. These tiny,
cold-start samples validate integration and are not performance claims.

Next implementation slice:

## 2026-08-13 — persistent PD orchestration slice

Implemented and real-GPU validated:

- long-lived GPU0 prefill and GPU1 decode processes;
- request RPC, startup/error handling, shutdown, and decode-side resource cleanup;
- SHM PARTIAL state handoff followed by decode-side Paged KV recomputation;
- multiple admitted requests combined into GPU1 batched decode;
- `serve --pd` and `benchmark --pd` using the same API and metric definitions as
  collocated mode.

The persistent two-request Qwen3.5-4B PD test passed. A two-request GSM8K PD CLI
smoke also completed 2/2 requests. Its cold, short-prompt numbers are integration
evidence only and correctly show no PD benefit at this scale.

Next implementation slice:

## 2026-08-13 — chunked Paged prefill history

Implemented and numerically validated:

- separate reserved KV capacity from the currently readable logical prefix;
- carry GDN convolution/recurrent state across chunks;
- attend continuation queries to all preceding physical KV pages with the correct
  per-query causal length;
- reuse HydraServe's Triton online-softmax Paged Attention across flattened
  `[request, query-position]` programs, avoiding dense score matrices;
- retain the permitted FlashAttention fast path for the first chunk when installed;
- use HydraServe Paged prefill for all chunks when FlashAttention is disabled.

Whole-prompt and chunked tiny-model results match numerically on CPU and CUDA.
Qwen3.5-4B completed a real 16-token-chunk GSM8K smoke (2/2 requests), and the
real persistent two-GPU PD regression still passes.

Next implementation slice:

## 2026-08-13 — reproducible arrivals and P/D overlap

Implemented:

- excluded warmup requests;
- burst, fixed-rate, and reproducibly seeded Poisson arrivals;
- asynchronous PD admission so waiting for GPU0 prefill does not pause active
  GPU1 decode;
- serialized GPU1 prepare/decode/release RPCs to keep one response stream safe.

Small controlled result (Qwen3.5-4B, GSM8K, 2 warmup + 8 measured, burst C=4,
8 output tokens): collocated output throughput 58.58 tok/s, P50 TTFT 171.17 ms,
P50 TPOT 49.45 ms; PARTIAL PD 22.65 tok/s, 679.81 ms, and 67.85 ms. The short
prompt workload is below crossover. Async overlap improved PD from 21.09 to
22.65 tok/s and P50 TPOT from 108.94 to 67.85 ms versus the earlier serialized
coordinator, but does not offset state transfer plus KV recomputation.

Next implementation slice:

## 2026-08-13 — 9B BF16 and 27B AWQ execution

Implemented and real-GPU validated:

- independent top-level `lm_head.weight` loading (9B/27B are not tied like 4B);
- Qwen3.5-9B BF16 32-layer, 2-token-chunk Paged prefill;
- compressed-tensors packed asymmetric INT4 loading;
- HydraServe Triton group-128 INT4 GEMM with packed weights and zero-points;
- CPU-resident embedding lookup for the memory-bound 27B AWQ layout;
- Qwen3.6-27B AWQ complete 64-layer prefill followed by one decode token.

The first 27B attempt reached the end of all layers but exposed a 4.74 GiB
temporary caused by converting the entire BF16 lm_head to FP32. The corrected
path performs BF16 GEMM and converts only logits to FP32. The full prefill+decode
test then passed. Model weights used about 22.02 GiB of PyTorch allocation.

Block-scaled 27B FP8 is explicitly rejected until a HydraServe FP8 GEMM exists;
it is not silently expanded to BF16. The local FP8 language tensors total about
25.08 GiB and do not fit one 24 GB card regardless.

Next implementation slice:

1. broaden B-vs-D across prompt lengths, concurrency, and arrival rates;
2. optimize and benchmark the new INT4 kernel;
3. P2P/NVLink validation on capable hardware.

## 2026-08-14 — production scheduling, recovery, sampling, and decode memory

Implemented and regression tested:

- joint KV/recurrent-state admission, bounded queues, and HTTP 429 backpressure;
- cost-aware physical-page prefix cache, per-request adaptive routing, and 1P+ND
  worker binding with capacity/cache/topology scoring;
- exact preemption replay, transactional state/KV rollback, batch/worker failure
  isolation, decode-worker restart, and health/restart metrics;
- aging weighted-fair decode selection and admission without head-of-line blocking;
- per-request temperature/top-p/top-k/min-p, penalties, seed, stop sequences,
  completion/chat logprobs, and streaming usage semantics;
- typed atomic single-envelope SHM with pinned staging, a layer-major contiguous
  GPU GDN-state pool sized from actual free memory, and in-place transactional
  recurrent-state commits;
- one metadata build per decode iteration, one batched Triton KV scatter per
  full-attention layer, and 16-token tiled Triton Paged Attention.

The transferred prefill token is authoritative for N-1 execution. A replay
argmax difference caused by cross-GPU floating-point ties is counted in
`hydraserve_pd_replay_mismatches_total` instead of incorrectly failing the
request; replay still advances and installs the decode-side state.

Real two-GPU Qwen3.5-4B tests cover physical prefix reuse, worker crash/restart,
seeded PD sampling, typed SHM, and pooled-state decode. On one RTX 3090, batched
KV scatter measured 1.15x/10.25x/42.79x over per-request launches at batch
1/8/32. Tiled Paged Attention measured 0.0241/0.1098/0.3479 ms for context
128/512/2048 at B=4, QH=16, D=128. These are kernel microbenchmarks, not
end-to-end service claims. N>1 decode workers and CUDA P2P remain unvalidated on
this two-GPU, no-peer-access host.
