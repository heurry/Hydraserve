# V5 DP4 vs H1-4GPU 结果汇总(2026-09-03)

负载:真实 LongBench 内容 M1(48 short 1-4K + 16 long 8-16K)/ B1(8+8),greedy,
`ignore_eos=false`,3 seed(42/43/44)。统一配置:`kv-quant int8`,`cache-tokens 65536`,
`block-size 256`,`concurrency 16`,`warmup 8`,`prefix-cache off`,`conditional-pd-tokens 6144`(仅 H1),
`hybrid-long-overflow-ms 5000`(仅 H1)。V5 SLO:Short e2e TTFT≤1s/TPOT≤100ms;
Long e2e TTFT≤10s/TPOT≤150ms/admission≤30s。
跑时统一 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(DP m1_s42 首次 GPU2 碎片化 OOM,加后通过)。

- DP4 = `--dp-devices 0 1 2 3`;H1-4GPU = `--prefill-devices 0 --decode-devices 1 2 3`(1P+3D)。
- 结果 JSON:`results/v5_dp4/`、`results/v5_h1_4g/`、`results/v5_h1_2g/`(2 卡门禁)。

## M1 三 seed 中位数(ms)

| 指标 | DP4 | H1-4GPU |
|---|---|---|
| Short e2e TTFT p50 | 639 | 772 |
| Short e2e TTFT p95 | 9591 | 17156 |
| Short e2e TTFT p99 | 10128 | 17374 |
| Short TPOT p50 | 81 | 97 |
| Long e2e TTFT p50 | 2638 | 10126 |
| Long e2e TTFT p95 | 9240 | 22019 |
| Long e2e TTFT p99 | 9659 | 28287 |

## M1 SLO goodput(逐 seed)

| seed | DP short met | DP short tok/s | DP long met | DP long tok/s | H1 short met | H1 short tok/s | H1 long met | H1 long tok/s |
|---|---|---|---|---|---|---|---|---|
| 42 | 27/48 | 42.3 | 14/16 | 40.9 | 13/48 | 17.6 | 8/16 | 22.9 |
| 43 | 32/48 | 49.0 | 16/16 | 45.4 | 16/48 | 22.4 | 8/16 | 22.4 |
| 44 | 32/48 | 48.0 | 16/16 | 45.1 | 12/48 | 15.8 | 8/16 | 22.8 |
| 中位数 | - | **48.0** | - | **45.1** | - | **17.6** | - | **22.8** |

总吞吐中位数:DP4 **117 tok/s** vs H1-4GPU **108 tok/s**(H1 为 DP 的 92%,勉强过 90% 线)。

## B1 seed42

| | DP4 | H1-4GPU |
|---|---|---|
| 成功 | 16/16 | 16/16 |
| Short SLO | 8/8 | 7/8 |
| Long SLO | 8/8 | 7/8 |
| 总 tok/s | 67.7 | 58.4 |

## V5 §8.1 判定(H1 是否比 DP 有收益)

三 seed **全部不满足**:
1. short SLO goodput:H1 全部低于 DP(H1 为 DP 的 30-45%),中位数 -63%;
2. short e2e TTFT p50 / TPOT p50:三 seed 全部恶化;
3. long e2e TTFT p50:H1 ~10.1s vs DP ~2.6s(约 4 倍);
4. 总吞吐 108 vs 117(-8%);
5. H1 无 starvation(max admission <7s),功能正常但性能全面落后。

**结论:M1 真实混合 RAG 负载下,H1-4GPU(1P+3D)不能称有收益,反而明显劣于 DP4。**
这是三 seed 一致的结构性结果,不是 seed 噪声。功能面(H1 死锁修复 `aee5213`)已验证:
最小复现 4/4、H1-2GPU 64/64、H1-4GPU 64/64,无死锁。

## 问题定位(H1 落后的调度根因)

H1 落后主因是**准入/路由的负载模型缺陷 + 单 prefill 引擎结构瓶颈**,详见
`results/v5_h1_scheduling_issue.md`。要点:

1. H1 路由 short→hybrid 只用 `decode_load`(KV/state 占用率)对比,**无 token 加权工作量账本**
   (对比 DP 的 `_prefill_tokens`/`_assigned_work`),short 可抢占唯一 prefill 槽;
2. 1P+3D 下 16 个 8-16K long 的 prefill 全部串行在 GPU0(`_hybrid_prefill_slot_available` 一次只能接 1 个 long);
3. decode worker 上 short 与 long decode 混排,部分 short 排到 13s。
