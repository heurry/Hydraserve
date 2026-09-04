# V5 DP4 vs H1-4GPU 结果(2026-09-04,更新计划复跑)

> 本轮按作者更新后的 BENCHMARK_PLAN_V5(commit `8949816`)复跑:DP4 与 H1-4GPU 都跑在
> 当前代码上。8949816 给两侧都加了"更公平"的机制 —— benchmark 给 `klass=short` 请求更高
> admission priority;DP worker 选择按 outstanding prefill tokens / token-weighted work /
> pending / cache-state load 负载均衡;H1 新增 token-aware admission、Hybrid Long-pressure
> 动态门控与 INT8 wire(`pd_prefill_token_budget=32768`、`hybrid_short_max_assigned_work=8192`、
> `hybrid_long_pressure_hold_ms=1000`、`pd_transfer_quant=int8`)。
> 另含本机修复(commit `202bd12`):prefill worker 释放 KV 块前 device sync,消除
> `--pd-transfer-quant int8` 高并发下的间歇 segfault。

负载:真实 LongBench 内容 M1(48 short 1-4K + 16 long 8-16K)/ B1(8+8),3 seed,greedy,
`ignore_eos=false`。统一:`kv-quant int8`,`cache-tokens 65536`,`block-size 256`,
`concurrency 16`,`warmup 8`,`prefix-cache off`,`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
DP4=`--dp-devices 0 1 2 3`;H1-4GPU=`--prefill-devices 0 --decode-devices 1 2 3`。
V5 SLO:Short e2e TTFT≤1s/TPOT≤100ms;Long e2e TTFT≤10s/TPOT≤150ms/admission≤30s。

- DP4 结果:`results/v5_dp4/`(git_commit 202bd12)
- H1-4GPU 结果:`results/v5_h1_4g/`(git_commit 202bd12/89498166;m1_s44 首跑 GPU2 OOM 已重跑通过)

## M1 三 seed 中位数(ms)

| 指标 | DP4 | H1-4GPU | 备注 |
|---|---|---|---|
| Short e2e TTFT p50 | 656 | **634** | H1 略优 |
| Short e2e TTFT p95 | 10306 | 13758 | H1 尾部更差 |
| Short e2e TTFT p99 | 11905 | 17177 | |
| Short TPOT p50 | **87** | 91 | |
| Long e2e TTFT p50 | **2779** | 6934 | H1 2.5x 差 |
| Long e2e TTFT p99 | 12614 | 21680 | |

## M1 SLO goodput(逐 seed)

| seed | DP short met | DP short tok/s | DP long met | DP total | H1 short met | H1 short tok/s | H1 long met | H1 total |
|---|---|---|---|---|---|---|---|---|
| 42 | 26/48 | 39.5 | 14/16 | 122.0 | 20/48 | 28.9 | 14/16 | 116.5 |
| 43 | 29/48 | 44.1 | 12/16 | 117.0 | 29/48 | 42.8 | 10/16 | 110.4 |
| 44 | 34/48 | 51.7 | 16/16 | 120.1 | 26/48 | 37.6 | 12/16 | 113.0 |
| 中位数 | - | **44.1** | - | **120.1** | - | **37.6** | - | **113.0** |

## B1 seed42

| | DP4 | H1-4GPU |
|---|---|---|
| 成功 | 16/16 | 16/16 |
| Short SLO | 8/8 | 8/8 |
| Long SLO | 8/8 | 8/8 |
| 总 tok/s | 66.7 | 59.4 |

## V5 §8.1 判定(新方法学)

M1 下 H1 **仍不满足 §8.1**:
1. short SLO goodput:H1 仅 s43 打平(29/48),s42/s44 均低于 DP;中位数 **37.6 vs 44.1**(-15%);
2. short e2e TTFT p50 中位 H1 634 vs DP 656(H1 略优),但 TPOT p50 H1 91 vs DP 87 略差;不满足"≥2 seed 均改善";
3. long e2e TTFT p50 **6934 vs 2779**(2.5x 差),单 Hybrid prefill 瓶颈仍在(计划原文已承认);
4. 总吞吐 113.0 vs 120.1(H1 为 94%,过 90% 线);
5. 无 starvation、无 >30s admission(H1 功能正常)。

**结论(新方法学下)**:8949816 的 token-aware admission + Long-pressure 门控 + INT8 wire 把 H1
从旧配置的全面落后(-63% short goodput)拉到了 short SLO 只差 15%、short TTFT p50 反超、总吞吐
94% 的水平,s43 打平。但 **M1 真实混合 RAG 下 H1 仍不能称"short SLO goodput 高于 DP"**。
short 的改善集中在中位数,尾部(排队 long 的 short)仍差于 DP,隔离不彻底。
**B1 边界验收通过**(H1 全 SLO 达标、无 starvation)。

注:8949816 计划原文自述"不改变单 Hybrid 在 Long-heavy 负载下 prefill bottleneck 的事实,
不保证 H1 超过强 DP,最终结论以正式复跑为准" —— 本轮正式复跑与此一致。

## 变更记录(相对上一版结论)

- 上版(`21fc97b`/`e19648b`)结论基于 **8949816 之前的代码**:DP 无 short-priority admission、
  H1 无 token-aware 调度与 INT8 wire,当时 H1 short goodput 只有 DP 的 30-45%。**已过期**。
- 本轮 DP4 与 H1 均跑在 8949816 + 本机 sync 修复(202bd12)上,是干净的对照。
- 间歇 segfault:`--pd-transfer-quant int8` 高并发下 prefill worker 偶发 CUDA IMA/segfault
  (约 1/4 次),`CUDA_LAUNCH_BLOCKING` 可复现规避;commit `202bd12` 加 device sync 后单轮未复现,
  仍需更多轮确认。见 `results/v5_h1_2g/` 与 `scripts/run_h1_v2_remaining.sh`。
