# HydraServe validation report — 2026-08-14

This report records integration and performance evidence collected on the local
two-GPU development host. It is deliberately not presented as a general PD
crossover claim.

## Environment

- model: `/mnt/nvme-data/models/LLM_model/Qwen3.5-4B` (BF16);
- GPUs: 2x RTX 3090 24 GiB, no CUDA peer access;
- transport: typed POSIX shared memory with pinned host staging;
- PD transfer mode: `PARTIAL_TRANSFER` (FP32 GDN state transfer plus decode-side
  full-attention KV recomputation);
- FlashAttention disabled, so both first and continuation chunks exercise
  HydraServe's Paged Attention implementation;
- dataset root: `/mnt/nvme-data/datasets/benchmark`.

## Short-prompt throughput matrix

GSM8K, 2 warmups + 8 measured requests, 16 output tokens, 512-token prompt cap,
8,192 cache tokens. Prompt lengths in the measured sample were 26–110 tokens.

| Mode | Concurrency | Success | Request/s | Output tok/s | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | Latency P50/P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| collocated | 1 | 8/8 | 2.958 | 47.330 | 48.87 / 136.83 | 17.46 / 18.06 | 310.93 / 400.11 |
| collocated | 4 | 8/8 | 7.805 | 124.878 | 108.13 / 193.72 | 25.33 / 30.61 | 501.79 / 540.06 |
| partial PD | 1 | 8/8 | 1.507 | 24.107 | 302.03 / 355.24 | 24.12 / 24.45 | 662.15 / 714.49 |
| partial PD | 4 | 8/8 | 2.701 | 43.212 | 719.07 / 1003.45 | 44.61 / 56.84 | 1393.15 / 1542.85 |

The short workload is below crossover. With no P2P and no concurrent long
decode workload to isolate from prefill, duplicate KV computation and IPC make
partial PD slower.

## 9K-token long-context regression

LongBench `gov_report`, 1 warmup + 2 measured requests, concurrency 1, 8 output
tokens, prompts truncated to exactly 9,000 tokens, 12,288 cache tokens and
512-token prefill chunks.

| Mode | Success | Request/s | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | Latency P50/P95 (ms) |
|---|---:|---:|---:|---:|---:|
| collocated | 2/2 | 0.0346 | 28647.15 / 28761.66 | 35.69 / 35.76 | 28897.00 / 29011.99 |
| adaptive → partial PD | 2/2 | 0.0232 | 42807.32 / 42809.21 | 47.76 / 47.98 | 43141.63 / 43141.94 |

Both adaptive requests were recorded as `pd_disaggregated` with
`route_reason=long_prompt_pd`. This proves that the real route and long-context
execution path work, but it also falsifies the static 8K route threshold for
this transport mode: partial PD TTFT is about 49% higher than collocated TTFT.
The next routing milestone must therefore use measured transfer/recompute cost,
not prompt length alone.

After the cost router was installed, a cold one-request 9K gate selected
`collocated` with `route_reason=cost_model_collocated`. The profile predicted
28.65 s collocated versus 47.10 s risk-adjusted partial PD (−18.45 s estimated
savings); the request completed successfully and its cold TTFT was 38.51 s.
The observation is retained by the online length-bucket calibration while the
service remains resident.

The checked-in route profile was then regenerated with
`python -m hydraserve fit-router-profile` from the concurrency-1 short and long
outputs in this report. Each route supplied 10 successful samples across 9
distinct lengths (26–9,000 tokens). The nonnegative quadratic fits achieved
68.02 ms collocated RMSE and 25.67 ms partial-PD RMSE. These figures measure fit
error on the small calibration set, not prediction error on an independent
holdout.

A second C4 run persisted admission decode load for all requests. Fitting those
loaded samples separately from the C1 base curves yielded
`decode_load_scale=1.08` collocated and `1.83` partial PD. With 18 observations
per route, collocated RMSE was 58.22 ms and partial-PD RMSE was 246.30 ms. The
large PD residual is a negative diagnostic result: requests at the same 0.75
decode load still had materially different TTFT because the current feature set
does not encode serial prefill queue position. The route model therefore uses a
risk margin, hysteresis and drift fail-closed behavior rather than treating this
one-dimensional load fit as exact.

A final real adaptive gate used the checked-in profile on 8 GSM8K requests at
C4 with 8 output tokens. It completed 8/8 at 7.79 request/s and 62.31 output
token/s. All short prompts were conservatively collocated; each result recorded
worker 0 and its admission load (0.0, 0.25, 0.5, or 0.75). This run validates
route metadata and load capture, not a comparison with the earlier 16-output-
token throughput matrix.

### Corrected adaptive-path calibration

The initial load fit above used standalone collocated TTFT as a proxy for the
adaptive worker path and therefore mixed admission queue, executor queue,
decode/event-loop interference, and model service. That diagnostic is retained
above because it motivated the correction, but it is not the final profile.

The corrected runner records internal request order, submission-to-admission
wait, predicted and directly measured executor queue wait, and directly timed
backend prefill service. Warmup compiles kernels, then resets only online EWMA
and hysteresis so cold compilation cannot contaminate measured route costs.
Both collocated and PD calibration traces now run through the same adaptive
coordinator; a temporary force-PD profile was used only to collect the PD arm.

After the warmup reset, C4 predicted-queue MAE was 56.7 ms collocated and
69.9 ms PD, down from 107.8 and 681.1 ms respectively. The final checked-in
direct-service curves use 17 collocated and 18 PD observations spanning
26–9,000 tokens. Their RMSE is 22.37 ms and 64.15 ms. These values supersede the
earlier TTFT/load fit for routing; admission and event-loop waits remain
separate SLO diagnostics.

## OOM found and fixed

The first 9K adaptive run failed 0/2 on the decode worker with an attempted
8.33-GiB allocation. The prefill worker was chunked, but the decode-side
`PARTIAL_TRANSFER` KV recomputation called a single full-prompt `forward`.

The decode worker now calls chunked `runtime.prefill`, using the same configured
`prefill_chunk_size` as the prefill worker. A unit test forces two-token chunks,
and the real 9K run above completed 2/2 without OOM. During execution each GPU
used about 12 GiB; after coordinator shutdown both returned to their idle
footprint.

## Test coverage added in this slice

- route, route reason and worker id are persisted per measured request;
- benchmark summaries aggregate successful requests by actual route;
- 128 concurrent submissions with cancellation, bounded active set and decode
  batch size are exercised in a synthetic soak;
- 20 repeated start/close cycles assert that every request and backend resource
  is released;
- the complete default CPU/CUDA/Triton suite passes. Hardware-intensive real
  model tests remain explicit opt-in tests and the 9K commands above provide the
  real-model regression for this change.

## Reproduction

```bash
/home/xdu/anaconda3/envs/deepseek/bin/python -m hydraserve benchmark \
  /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  /mnt/nvme-data/datasets/benchmark \
  --dataset longbench --subset gov_report --limit 2 --warmup 1 \
  --max-new-tokens 8 --max-prompt-tokens 9000 --concurrency 1 \
  --adaptive --device cuda:0 --decode-device cuda:1 \
  --cache-tokens 12288 --prefill-chunk-size 512 \
  --no-flash-attention --output benchmark_output/longbench-adaptive.json
```
