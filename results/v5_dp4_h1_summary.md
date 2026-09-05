# V5 DP4 vs H1-4GPU vs H2-4GPU 结果(2026-09-05,三方对比)

> 三方 arm:DP4 / H1-4GPU(1P+3D)/ H2-4GPU(2P+2D)。DP4 与 H1-4GPU 跑在 `8949816`+本机
> `202bd12`(short admission priority + token-weighted 负载均衡已含);H2-4GPU 跑在 `5569386`
> (作者把 202bd12 的 device-sync 升级为更优的非阻塞 `_DeferredCudaCacheFree`)。
> 8949816/5569386 的公平机制:benchmark 给 `klass=short` 请求更高 admission priority;
> DP worker 按 outstanding prefill tokens / token-weighted work / pending / cache-state load
> 负载均衡;H 系列用 token-aware admission、Hybrid Long-pressure 动态门控与 INT8 wire。
> 注意:H2 与 DP4/H1 差一个 commit(runner 默认路径仅在新增 opt-in `--closed-loop-clients`
> 处分支,默认行为一致),严格三方同 commit 对比需在 5569386 上重跑 DP4/H1。

负载:真实 LongBench 内容 M1(48 short 1-4K + 16 long 8-16K)/ B1(8+8),3 seed,greedy,
`ignore_eos=false`。统一:`kv-quant int8`,`cache-tokens 65536`,`block-size 256`,
`concurrency 16`,`warmup 8`,`prefix-cache off`,`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
DP4=`--dp-devices 0 1 2 3`;H1-4GPU=`--prefill-devices 0 --decode-devices 1 2 3`(P budget 32768);
H2-4GPU=`--prefill-devices 0 1 --decode-devices 2 3`(P budget 65536)。
V5 SLO:Short e2e TTFT≤1s/TPOT≤100ms;Long e2e TTFT≤10s/TPOT≤150ms/admission≤30s。

- DP4 结果:`results/v5_dp4/`(git_commit 202bd12)
- H1-4GPU 结果:`results/v5_h1_4g/`(git_commit 202bd12/89498166;m1_s44 首跑 GPU2 OOM 已重跑通过)
- H2-4GPU 结果:`results/v5_h2_4g/`(git_commit 5569386)

## M1 三 seed 中位数(ms)

| 指标 | DP4 | H1-4GPU | H2-4GPU |
|---|---|---|---|
| Short e2e TTFT p50 | 656 | 634 | **631** |
| Short e2e TTFT p95 | 10306 | 13758 | 14522 |
| Short TPOT p50 | 86.6 | 91.0 | **78.6** |
| Long e2e TTFT p50 | **2779** | 6934 | 5793 |
| Long TPOT p50 | **82.8** | 94.6 | 134.0 |
| Short SLO goodput tok/s | **44.1** | 37.6 | 30.3 |
| Long SLO goodput tok/s | **43.7** | 31.7 | 22.2 |
| 总吞吐 tok/s | **120.1** | 113.0 | 92.7 |

## M1 SLO goodput(逐 seed)

| seed | DP short | DP short tok/s | DP total | H1 short | H1 short tok/s | H1 total | H2 short | H2 short tok/s | H2 total |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 26/48 | 39.5 | 122.0 | 20/48 | 28.9 | 116.5 | 27/48 | 30.1 | 90.0 |
| 43 | 29/48 | 44.1 | 117.0 | 29/48 | 42.8 | 110.4 | 27/48 | 38.7 | 111.0 |
| 44 | 34/48 | 51.7 | 120.1 | 26/48 | 37.6 | 113.0 | 11/48 | 13.2 | 92.7 |
| 中位数 | - | **44.1** | **120.1** | - | 37.6 | 113.0 | - | 30.3 | 92.7 |

## B1 seed42

| | DP4 | H1-4GPU | H2-4GPU |
|---|---|---|---|
| 成功 | 16/16 | 16/16 | 16/16 |
| Short SLO | 8/8 | 8/8 | 8/8 |
| Long SLO | 8/8 | 8/8 | 8/8 |
| 总 tok/s | **66.7** | 59.4 | 60.2 |

## V5 §8.1 判定(三方)

**M1 下 H1 与 H2 都不满足 §8.1(short SLO goodput 高于 DP):**
- H1:short goodput 37.6 vs DP 44.1(-15%);仅 s43 打平;short TTFT p50 634 vs 656 略优、
  TPOT p50 91 vs 87 略差;long TTFT p50 6934 vs 2779(2.5x);总吞吐 113(-6%)。
- H2(2P+2D):**总吞吐反而最低(92.7,-23%)**;short goodput 30.3;s42 的 short SLO(27)略超 DP(26),
  但 **s44 崩塌到 11/48**(DP 34),2 张 decode 卡拥塞、不稳定;short TPOT p50(78.6)与
  short TTFT p50(631)最好,long TPOT 恶化到 134ms。
- 无 starvation、无 >30s admission(三个 arm 功能都正常,均为 64/64)。

**结论**:
1. **DP4 仍是 4 卡 short-heavy RAG 负载下最好最稳的 arm**。吞吐跟着 decode 引擎数走
   (DP4 4 > H1 ~3.5 > H2 2):H1 用 1 张专 prefill 卡换 short 首 token 略快,但 goodput 与
   尾部略亏;H2 用 2 张 prefill 卡处理本就不多的 16 个 long(容量过剩),decode 饿着 → 总吞吐
   -23% 且 s44 不稳定。**加 prefill 并行没有缓解瓶颈,反而把瓶颈换到了 decode 侧**。
2. 计划原文对 H2 的预期("缓解单 Hybrid prefill bottleneck 但减少 D-bound decode capacity")
   被数据验证,且**减少 decode 的代价更大**。
3. B1 边界验收三个 arm 全过。

注:计划原文自述 H 系列"不改变单 Hybrid 在 Long-heavy 负载下 prefill bottleneck 的事实,
不保证超过强 DP,最终结论以正式复跑为准" —— 本轮三方复跑与此一致。

## 变更记录

- `e19648b` 结论基于 8949816 之前代码(H1 无 token-aware 调度),已过期。
- `52ad9ea` DP4/H1 跑在 8949816+202bd12(本机 sync 修复):H1 short goodput -15%,short TTFT
  p50 反超,总吞吐 94%。
- 本轮新增 H2-4GPU(5569386):2P+2D 总吞吐 -23%、s44 崩塌 → 2P 方案不成立。
- 间歇 segfault:`--pd-transfer-quant int8` 高并发下 prefill worker 偶发 CUDA IMA/segfault;
  作者 5569386 已用非阻塞 `_DeferredCudaCacheFree` 取代本机 202bd12 的阻塞 device-sync。
