# benchmark_output

HydraServe 压测数据归档(2026-08-15,4×RTX 3090,无 P2P/CNS)。完整结论见 `../log.md`。

## context_sweep/ — 上下文扫描(concurrency=1,单 chunk 走 FA)

DP(collocated) vs PD(1P+1D,FULL 传输)。文件名 `{mode}_ctx{上下文}_c1.json`。

| 上下文 | DP TTFT | PD TTFT | PD/DP |
|---|---|---|---|
| 32768 | 12.3s | 17.0s | 1.38× |
| 65536 | 27.8s | 36.2s | 1.30× |
| 131072 | 65.4s | 74.8s | 1.14× |

注:`pd_ctx131072_c1.json` 是 OOM 记录(空 ttft),有效数据见 `../diagnostics/pd_128k_retry.json`(74.8s)。

## concurrency_sweep/ — 并发扫描(8K 上下文,max_new=128)

DP(collocated) vs PD(1P+1D)。文件名 `{mode}_ctx8192_c{并发}.json`。

| 并发 | DP TPOT P99 | PD TPOT P99 | DP TPOT P50 | PD TPOT P50 | DP tok/s | PD tok/s |
|---|---|---|---|---|---|---|
| 1 | 33.0ms | 32.5ms | 32.2 | 32.3 | 18.0 | 16.8 |
| 4 | 109.3ms | 37.3ms | 86.4 | 36.3 | 30.4 | 34.7 |
| 32 | 825.9ms | 40.2ms | 498.4 | 35.8 | 34.5 | 34.8 |

注:`dp_ctx8192_c16.json` 是中断的错误文件(336B,无数据)。

## diagnostics/ — 各诊断数据点

| 文件 | 含义 |
|---|---|
| `bench_sharegpt_c1.json` | 首个 4K sharegpt 冒烟(8/8 成功) |
| `pd_longbench_gov_c4.json` | 30K LongBench 1P+3D adaptive(无 FA,TTFT 124s) |
| `diag_32k_clean.json` | 32K B1 修复前(chunk 4096,TTFT 268s) |
| `diag_32k_b1.json` | 32K B1 修复后(chunk 32768,TTFT 11.25s) |
| `pd_32k_b1.json` | PD 32K FA 开(22.7s) |
| `pd_32k_fa_fix.json` | PD 32K FA 修复验证 |
| `pd_32k_full.json` | PD 32K FULL 传输(15.9s) |
| `diag_64k_chunk32768.json` | 64K chunk 32768(TTFT 607s,B2 慢) |
| `diag_64k_chunk65536_fix.json` | 64K 单 chunk(grid 修复后,27.4s) |
| `diag_128k_chunk131072_v2.json` | 128K 单 chunk(65.6s) |
| `pd_128k_retry.json` | PD 128K(74.8s,expandable_segments 修复碎片化) |
| `diag_8k_int8.json` | 8K INT8 KV 正确性验证(2.8s) |
| `fp8_8k.json` | FP8 测试(非本会话生成) |
