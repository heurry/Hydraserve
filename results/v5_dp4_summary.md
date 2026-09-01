# V5 DP4 结果(2026-09-01)

负载:真实内容 M1(48 short + 16 long)/ B1(8 short + 8 long),greedy,`ignore_eos=false`,
`kv-quant int8`,`cache-tokens 65536`,`block-size 256`,`concurrency 16`,`warmup 8`,
`prefix-cache off`。V5 SLO:Short e2e TTFT ≤ 1s / TPOT ≤ 100ms;Long e2e TTFT ≤ 10s / TPOT ≤ 150ms / admission ≤ 30s。

> 注:m1_s42 首跑 1 个 long 在 GPU2 OOM,重跑加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
> 后 64/64 通过。建议正式矩阵所有 run 统一使用该 alloc 配置(仅分配器稳定性,不影响延迟)。

## M1(三 seed 中位数 [min-max])

| 类别 | 指标 | p50 | p95 | p99 |
|---|---|---|---|---|
| Short | e2e TTFT ms | 639 [543-650] | 9591 [5468-15507] | 10128 [6809-16641] |
| Short | TPOT ms | 81 [73-89] | 114 [103-147] | 126 [108-155] |
| Long | e2e TTFT ms | 2638 [2586-2762] | 9240 [5498-12987] | 9659 [6189-17733] |
| Long | TPOT ms | 83 [81-86] | 114 [96-120] | 121 [99-130] |

## SLO goodput(逐 seed)

| run | Short met | Short goodput tok/s | Long met | Long goodput tok/s | 总 tok/s |
|---|---|---|---|---|---|
| m1_s42(重跑) | 27/48 | 37.5 | 14/16 | 41.8 | 113.9 |
| m1_s43 | 32/48 | 49.0 | 16/16 | 45.4 | 117.3 |
| m1_s44 | 32/48 | 48.0 | 16/16 | 45.1 | 117.1 |
| **中位数** | - | **48.0** | - | **45.1** | **117.1** |

正确性:三个 seed 全部 64/64 成功,0 失败,无 starvation(admission max < 130ms)。

## B1 seed42

16/16 成功;Short SLO 8/8、Long SLO 8/8。
Short e2e TTFT p50/p95/p99 = 573/697/711ms;Long = 2167/2878/3019ms。

## 观察(仅 DP,trend)

- Short e2e TTFT p95 达 5.5-15.5s,主要来自 long prefill(8-16K)在 collocated worker 上与
  short 排队竞争 → 是 H1 应该用 prefill 隔离改善的核心负载。
- Long 在 10s SLO 下基本达标;short 约 56-67% 达标。
- **H1 未纳入**:H1-2GPU 在 PD shm-ring 分块传输上死锁(prefill 发 KV→ring 满→decode 不消费→
  循环等待),见 `BENCHMARK_PLAN_V5.md` 讨论与 `scripts/gen_v5_trace.py` 最小复现
  `traces/v5/tiny_seed42.jsonl`。非分块路径 ring 槽 64MB 容不下 8K+ KV(322MB),亦不可行。
