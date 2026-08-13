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

At this milestone block-scaled 27B FP8 was still rejected instead of being
silently expanded to BF16. Native execution is covered by the later FP8
milestone below.

## 2026-08-14 — CPU-resident embedding input placement

The memory-bound 27B AWQ path previously constructed token IDs on CUDA in every
backend and immediately copied them back with `input_ids.cpu()` for the
CPU-resident embedding. Runtime now exposes the embedding's `input_device`, and
collocated serving, batch recovery, prefill/decode workers, and PD replay create
IDs there directly. Only selected embedding rows cross to the execution GPU.
GPU-resident BF16 models retain their original all-CUDA path.

A mixed-device tiny-model test matches an all-GPU embedding baseline, while all
serving adapters retain compatibility with generic runtimes that only expose
`device`. The real Qwen3.6-27B AWQ 64-layer prefill plus decode smoke passed with
CPU token tensors and the existing single-3090 memory bound.

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
- Paged KV block tables and logical lengths packed on the host and uploaded as
  two contiguous tensors, eliminating O(batch) row/scalar CUDA updates.

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

On the same RTX 3090, contiguous Paged KV metadata construction measured
0.0168/0.0199/0.0381/0.0439 ms at batch 1/8/32/64, versus
0.0255/0.2101/0.7250/1.5161 ms for the previous per-row update path
(1.52x/10.55x/19.00x/34.50x). The regression test also asserts that metadata
construction performs exactly two contiguous device tensor builds independent
of batch size.

## 2026-08-14 — long-context and serving stress validation

Implemented and validated:

- decode-side `PARTIAL_TRANSFER` KV recomputation now uses bounded chunked
  prefill instead of materializing a complete long-prompt forward;
- benchmark records the actual route, route reason and decode worker binding;
- concurrent submit/cancel stress and repeated service lifecycle tests;
- a real two-GPU 9,000-token LongBench regression, with HydraServe Paged
  Attention enabled for every chunk, completed 2/2 after the memory fix.

The same 9K workload measured 28.65 s collocated TTFT P50 and 42.81 s partial-PD
TTFT P50. This is an important negative result: the current static 8K routing
threshold is not suitable for SHM partial transfer because decode-side KV
recomputation duplicates substantial work. Detailed commands, short-prompt
throughput, and percentile tables are in
[`BENCHMARK_2026-08-14.md`](BENCHMARK_2026-08-14.md).

Next implementation slice:

1. replace static route thresholds with calibrated transfer/recompute/load cost;
2. add online latency observations and conservative fallback when calibration is
   unavailable;
3. validate route decisions under mixed prompt lengths and decode pressure.

## 2026-08-14 — calibrated cost-aware routing

Implemented and tested:

- risk-adjusted quadratic latency curves for collocated and PD prefill;
- minimum absolute and relative savings gates, plus a conservative minimum
  prompt gate;
- prompt-length-bucketed EWMA corrections from real route latency observations;
- JSON route profiles for model/transport/hardware-specific calibration;
- immutable request decisions carrying both predicted costs, estimated savings,
  confidence, route reason, and decode worker binding;
- route-calibration health output and Prometheus observation/correction metrics.

The threshold router remains available as an explicit policy object for tests
and controlled experiments. Production `--adaptive` defaults to the calibrated
SHM/PARTIAL cost profile. A real cold 9K adaptive gate selected collocated with
predicted costs 28.65 s vs risk-adjusted 47.10 s PD and completed successfully;
this is the intended correction to the earlier static-threshold decision.

Next implementation slice:

1. automate multi-length/multi-concurrency calibration traces and curve fitting;
2. separate mean service cost from SLO-tail externality under active decode load;
3. add route-decision hysteresis and profile drift alarms.

## 2026-08-14 — reproducible route-profile fitting

Implemented:

- a `fit-router-profile` CLI consuming native benchmark JSON outputs;
- strict exclusion of failed requests and a three-distinct-length coverage gate;
- numerically scaled, nonnegative quadratic least-squares fitting without an
  external optimization backend;
- fit metadata containing sample count, prompt range and RMSE;
- direct loading of fitted profiles, including their audit metadata.

The initial 4B/RTX-3090/SHM-PARTIAL base curves were regenerated from 10
collocated and 10 partial-PD concurrency-1 observations spanning 9 distinct
lengths from 26 to 9,000 tokens. Base-fit RMSE was 68.02 ms for collocated and
25.67 ms for partial PD. Loaded traces are handled as a separate multiplicative
externality rather than being folded into the base length coefficients.

## 2026-08-14 — route stability and first-order load externality

Implemented:

- admission-time decode load persisted in every benchmark request record,
  including fixed collocated and fixed PD calibration runs;
- separate nonnegative `decode_load_scale` fitting on loaded traces while the
  base length curve is fitted only from low-load samples;
- per-prompt-bucket Schmitt-trigger hysteresis around the PD savings boundary;
- online profile-drift detection after a configurable minimum observation count;
- fail-closed collocated routing on drift, degraded health, and Prometheus drift
  gauges.

Adding 8 C4 observations per route produced load scales of 1.08 collocated and
1.83 partial PD. The combined collocated RMSE improved to 58.22 ms, while
partial-PD RMSE increased to 246.30 ms. The latter is retained as evidence that
decode load alone is insufficient: serial prefill queue position and in-flight
prompt work must be explicit features in the next routing model.

## 2026-08-14 — queue/service latency decomposition

The first loaded profile incorrectly treated end-to-end TTFT as model service
time. Production instrumentation now records, per request:

- submission-to-admission wait;
- predicted work ahead at route binding;
- directly timed executor queue wait;
- directly timed backend prefill service;
- end-to-end TTFT and internal request/admission order.

The router adds common queue work to both TTFT predictions but compares only
route-specific service cost. Online EWMA consumes direct backend service time,
not TTFT-minus-an-estimate. Running futures contribute only predicted remaining
work. Benchmark warmup now resets online correction and hysteresis state after
kernel compilation while keeping workers and kernels resident; this prevents a
cold compile from applying the 4x correction cap to measured requests.

On clean C4 traces, predicted executor-queue MAE fell from 107.8 to 56.7 ms for
adaptive collocated and from 681.1 to 69.9 ms for adaptive partial PD. The final
direct-service profile uses 17 collocated and 18 PD samples over 26–9,000
tokens; RMSE is 22.37 ms collocated and 64.15 ms PD. Admission wait and event-
loop/decode interference remain separately visible instead of contaminating
the service curve.

## 2026-08-14 — active-set and deadline scheduling

Implemented and tested:

- independent `max_active_requests` and decode `max_batch_size` configuration,
  with state-pool capacity sized for the active limit and validation that the
  active limit cannot be smaller than one batch;
- deadline-aware weighted-fair decode selection while retaining priority and
  aging-based starvation prevention;
- deadline expiration before admission, after prefill, and at decode result
  boundaries, with transactional backend resource release;
- no token emission when a decode call returns after the request deadline;
- HydraServe `timeout_ms`, HTTP 408 for non-streaming requests, and structured
  timeout events after SSE headers have been committed;
- scheduler gauges for admission pending, prefill pending, and active requests.

Deadlines are cooperative at GPU kernel boundaries because an in-flight CUDA
kernel is not safely cancellable. Unit tests cover waiting-admission expiry,
decode-result expiry, urgency ordering, and four active requests scheduled with
a decode batch size of two.

## 2026-08-14 — online collocated preemption and exact recovery

Implemented and validated in the production `ContinuousGenerationLoop` and
`RuntimeGenerationBackend` path:

- a strictly more urgent arrival (higher priority, or an earlier deadline at
  equal priority) can preempt a lower-urgency active request at a decode
  iteration boundary;
- preemption atomically releases both physical KV ownership and recurrent-state
  slots, and a configurable per-request cap prevents unbounded thrashing;
- recovery recomputes `prompt + generated[:-1]`, because the most recently
  emitted token has not yet been consumed by decode, and resumes without
  sampling or emitting a duplicate token;
- cancellation and deadline expiry are processed while requests are suspended;
  recovery failures are request-scoped and release partial allocations;
- health and Prometheus surfaces report suspended depth and preemption/recovery
  success and failure counters.

Unit tests compose preemption with streaming, priority, deadlines, exact replay,
failure cleanup, and allocator reservation lengths. An opt-in real Qwen3.5-4B
  GPU test forced a preemption after the first decode boundary, matched the full
  recovered token stream against uninterrupted greedy generation, and audited all
  KV pages and recurrent-state slots as free afterward.

## 2026-08-14 — async PD and multi-worker preemption recovery

Implemented and validated after the collocated slice:

- async admission can preempt at the same decode iteration boundary without
  treating an in-flight prefill as an active victim;
- recovery is submitted through the persistent prefill executor and identified
  separately from ordinary prefill completion, so it installs state without
  emitting or sampling a second first token;
- the decode worker validates `replay == prompt + generated[:-1]`, replaces the
  old reservation transactionally, performs chunked local recompute, restores
  the original sampling history, and returns updated capacity/cache telemetry;
- fixed 1P+1D and adaptive 1P+ND coordinators both expose preempt/recover, with
  the latter allowed to rebind a suspended request to a currently healthy worker;
- transient recovery admission failure remains suspended and retries; ambiguous
  preemption failure conservatively terminates only the victim and attempts
  idempotent cleanup rather than decoding against missing state.

Two opt-in real Qwen3.5-4B tests passed on the local RTX 3090 pair. The fixed PD
test used seeded temperature/top-k sampling and matched all recovered tokens to
an uninterrupted run, proving sampling-step continuity. The multi-worker test
covered coordinator rebinding and local recovery RPC. Both ended with zero live
allocations and all physical KV blocks free.

## 2026-08-14 — active-request survival across worker loss

Implemented and validated:

- a decode-worker timeout or exit atomically marks the worker unhealthy and
  invalidates every request binding owned by that worker, including requests
  outside the failing decode batch;
- host-side reservation and route ownership are cleared together, while request
  token history remains in the serving coordinator;
- partial decode results from healthy workers are committed normally; only
  requests whose device-local state was lost enter fault suspension;
- suspended requests retry admission while the replacement process loads, then
  may rebind to any healthy capacity and use the same exact replay recovery path;
- an explicit fault-suspension counter distinguishes hardware/process recovery
  from policy-driven priority preemption.

An opt-in real test emitted the first token from Qwen3.5-4B, terminated the bound
decode subprocess at the next decode boundary, waited for automatic model reload
and capacity handshake, then recovered the active request. Its seeded sampled
token stream exactly matched an uninterrupted baseline, with one worker restart,
one fault suspension, and one successful request recovery.

## 2026-08-14 — prefill-worker liveness and route restoration

Implemented for the adaptive 1P+ND coordinator:

- admission probes prefill-process liveness before choosing a route, so a worker
  that died while idle cannot consume one sacrificial PD request or wait for the
  full operation timeout;
- in-flight RPC polling checks process liveness every 100 ms and quarantines the
  PD route on exit or timeout;
- while unhealthy or reloading, new requests are explicitly routed collocated
  with `prefill_unavailable`, preserving service availability;
- recovery replaces both IPC queues, starts a new prefill process, validates the
  model-name handshake, uses bounded exponential-backoff retries, and only then
  re-enables PD decisions;
- health and Prometheus expose prefill health, recovering state, and restart
  attempt/success/failure counts separately from decode-worker recovery.

The real two-GPU test terminated an idle Qwen3.5-4B prefill process. The very
first subsequent long request completed on the collocated route, and after one
successful background restart the same prompt returned to PD and completed.

## 2026-08-14 — supervised legacy adaptive and fixed-PD workers

The original one-prefill/one-decode backends now use the same production
failure semantics instead of blocking on raw multiprocessing queue reads:

- prefill and decode RPCs poll child liveness every 100 ms and have independent
  bounded restart state machines, fresh queues, model-name handshakes, health,
  and restart counters;
- single-decode adaptive mode fails closed to collocated during prefill reload
  and restores PD only after a successful handshake;
- fixed `--pd` cannot change route, so admission remains deferred while prefill
  reloads and proceeds without exposing a transient worker failure to clients;
- decode loss invalidates all local reservations, suspends affected active
  requests, and exact-replays them after the replacement decode process is ready;
- API health/metrics consume the same decode and prefill recovery interfaces for
  legacy and 1P+ND backends.

Real Qwen3.5-4B validation killed both legacy adaptive subprocesses in one
service lifetime: prefill failover/restoration passed, then an active seeded
sampling request survived decode loss and matched an uninterrupted baseline.
A separate fixed-PD test killed prefill before submission; the request aged in
admission during model reload and then completed normally on PD.

## 2026-08-14 — KV safety headroom, audit, and observability

Implemented and validated:

- configurable KV block headroom excluded from admission while remaining
  visible as physical free capacity;
- prefix eviction under active-request pressure that reclaims only the shortage
  and never consumes headroom;
- allocator high-watermark and failure counters, request/shared/reference
  ownership, logical/reserved tokens, and internal-fragmentation statistics;
- Prefix Cache referenced/evictable/byte/hit-token counters plus bounded reason
  labels for admission rejection and eviction;
- Paged KV audits reconciling free list, refcounts, request block references,
  prefix owners and per-request prefix metadata;
- health and Prometheus export from collocated, single-decode PD, and aggregated
  multi-worker backends.

A deterministic 2,000-operation allocator fault/transaction soak audits every
step and ends with all 64 physical blocks and zero references. A real 4B
collocated run with four active requests ended with 16/16 pages free, zero state
slots and references, 14 allocatable pages plus 2 headroom, and a four-page high
watermark. A real two-GPU PD test ended with 8/8 remote decode pages free, 6
allocatable plus 2 headroom, and no active allocation/reference leak.

## 2026-08-14 — allocation-free transactional GDN-state batching

Implemented and validated:

- a memory-budgeted, fixed-size layer-major decode workspace for pooled FP32
  recurrent and convolution states;
- two batched gathers and two batched commits per decode step instead of
  per-layer `cat` allocations and per-layer/per-request copies;
- caller-provided Triton convolution next-state storage, so the kernel writes
  directly into the transactional workspace;
- state publication only after the final logits projection succeeds; exceptions
  leave pooled state and sequence lengths unchanged for exact retry;
- workspace size derived from the configured decode batch rather than the
  potentially larger active-request limit, with actual capacity included in the
  CUDA memory budget and exposed through health/Prometheus metrics.

Using Qwen3.5-4B's actual 53.48 MB/request state shape on one RTX 3090, isolated
state movement at batch 1/4/8/16 measured 0.2898/1.2135/2.2507/4.6958 ms versus
0.5310/1.6869/3.0533/5.7370 ms previously (1.83x/1.39x/1.36x/1.22x). The old
path created 52/204/408/816 MiB of peak transient allocations per step; the
preallocated hot path created none. A real two-request Qwen3.5-4B batched decode
matched sequential BF16 logits within 0.12 absolute tolerance and produced
identical argmax tokens. Real preemption/recovery also passed unchanged.

## 2026-08-14 — native block-scaled FP8 execution on Ampere

Implemented and validated:

- a `BlockScaledFP8Weight` loader that pairs each E4M3 tensor with its 128x128
  `weight_scale_inv` grid without materializing BF16 weights;
- a HydraServe Triton GEMM that manually decodes E4M3FN bit patterns on SM86,
  applies two-dimensional inverse scales, and accumulates BF16 dot products;
- exact tests over every finite positive and negative E4M3FN encoding, partial
  scale blocks, CPU oracle execution, host-resident streaming, and runtime
  loader dispatch;
- memory-aware placement that keeps embedding/lm_head on CPU and selects the
  smallest set of largest FP8 projections for host streaming until a 1 GiB CUDA
  execution reserve is preserved;
- corrected recurrent-state budgeting that independently enforces both the
  configured free-memory fraction and hard reserve.

The real 10240x5120 projection from the local Qwen3.6-27B-FP8 checkpoint matches
a materialized BF16 oracle. On one RTX 3090, the final tiled kernel measured
0.2645/0.2444/0.2461/0.4009/1.5158 ms at 1/8/32/128/512 rows. The complete
64-layer checkpoint loaded with 21.636 GiB of PyTorch allocation and completed
one-token prefill plus pooled decode with finite 248,320-way logits; peak
allocation was 22.099 GiB. This is an Ampere compatibility
path, not a claim of native Hopper FP8 Tensor Core throughput.

## 2026-08-14 — memory-planned KV capacity and Paged PD prefill

Implemented and validated:

- `cache_tokens` is a requested upper bound converted into physical pages only
  after model weights have been placed and actual free CUDA memory is known;
- the planner reserves a complete recurrent-state transaction, the state-pool
  fraction constraint, 512 MiB hard CUDA headroom, and a 64 MiB allocator guard;
- requested/planned blocks, allocated bytes, reserved bytes, and clamp state are
  exported through health and Prometheus;
- the FP8 placement planner receives the requested cache size and offloads the
  minimum additional large projections needed to preserve it when possible;
- persistent prefill workers now use self-managed Paged KV for both the first
  and continuation chunks and free all pages after transfer, replacing dense KV
  concatenation in the PD prefill process.

On the 24 GB RTX 3090, Qwen3.6-27B-FP8 now honors the default 65,536-token target
with all 4,096 physical 1 MiB pages (4 GiB), then allocates the default pooled
transactional state and completes prefill plus decode. A real two-GPU
Qwen3.5-4B persistent PD generation also passed after switching the prefill
worker to Paged KV.
