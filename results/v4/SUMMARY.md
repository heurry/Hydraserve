# HydraServe V4 压测结果：动态 Hybrid vs 4×DP

> 4×RTX 3090 24GB，Qwen3.5-4B BF16 + INT8 KV，SHM 4.58 GB/s。
> 基线 commit：`a090ec9`（动态 Hybrid 角色调度）。
> 负载窗口 120s，Poisson 到达（long 偏移预计算），0.8× offered load（λ_max=2.5，SHORT_RATE=2.0 req/s）。
> 统一：C16、block 256、chunk 16384、max_step 8192、kv int8、prefix off、shm-ring、4096 PD 阈值。
> 拓扑：D0（4×DP，cache 131072）、H1（1 Hybrid+3D 动态，cache 65536）、H2（2 Hybrid+2D）、P0（静态 2P+2D）。

## 一、主结论

**1P+3D 动态 Hybrid（H1）在三个 PD 有利负载下全部占优 4×DP**：

| 负载 | D0 SLO | H1 SLO | D0 吞吐 | H1 吞吐 | D0 TPOTp50 | H1 TPOTp50 |
|---|---|---|---|---|---|---|
| R1 (RAG-QA) s42 | 16/48 | **28/48** | 101 | **108** | 133 | **75** |
| R1 s43 | 16/48 | **31/48** | 98 | **109** | 156 | **74** |
| R1 s44 | 16/48 | **29/48** | 96 | **115** | 153 | **70** |
| R2 (Doc-Sum) s42 | 16/48 | **33/48** | 79 | 73 | 155 | **97** |
| R2 s43 | 16/48 | **32/48** | 71 | 66 | 161 | **76** |
| R2 s44 | 16/48 | **32/48** | 78 | 75 | 116 | **78** |
| R3 (Code-Anal) s42 | 16/44 | **30/44** | 94 | **112** | 153 | **73** |

- **SLO goodput**：H1 28-33/48 vs D0 16/48（**+75%~+106%**）
- **TPOT p50**：H1 70-97 vs D0 116-161（**-38%~-52%**）
- **吞吐**：R1 三 seed + R3 **超 D0（+7%~+19%）**；R2 为 D0 的 92%（≥60% 判据）

满足 V4 主验收（SLO 占优、TPOT 大幅改善、吞吐保持/反超）。

## 二、单点完整对比（R1 0.8×，seed42，四拓扑）

| 拓扑 | 成功 | short SLO | TPOT p50/p99 | TTFT p99 | 吞吐 |
|---|---|---|---|---|---|
| D0 (4×DP) | 64/64 | 16/48 (33%) | 133/227 | 2.4s | 100.5 |
| **H1 (1H+3D)** | 64/64 | **28/48 (58%)** | **75/216** | 2.8s | **108.0** |
| H2 (2H+2D) | 64/64 | 23/48 (48%) | 96/271 | 2.9s | 94.8 |
| P0 (静态 PD) | 64/64 | 11/48 (23%) | 153/273 | 6.1s | 81.1 |

趋势：**H1 > H2 > D0 > P0**（SLO）。静态 PD 最差（23%），动态 Hybrid 最佳（58%）。

## 三、关键机制

1. **H1 是 R1/R2/R3 的最优配置**：P 利用率 ≤50%（R2 最高 50%），1 张 Hybrid 够用。H2 少 1 张常驻 D 卡（baseline decode 容量 -25%），TPOT 反而差。
2. **prefill 隔离价值随干扰增大**：R2（16K，5s prefill）SLO +106%；R1/R3（8K，2.5s）+75%。
3. **吞吐反超**：H1 R1/R3 吞吐超 D0——D 卡 decode 不被干扰、每 token 更快；传输开销（66ms/long）远小于干扰节省。
4. **work-conserving**：Hybrid 卡空闲时服务 short（collocated），long 到达切回 prefill——保留接近 4×DP 的 decode 容量 + PD 隔离。

## 四、已知限制 / 口径修正

1. **H2 的 R2（16K）死锁**：2 个 P 进程同时向同一 D 的 SHM ring 发 512MB KV，3×64MB ring 不够。工程问题，记为已知限制，不影响主结论。
2. **`--conditional-pd-tokens` 是 `>=` 比较**：2048 会把恰好 2K 的 short 判为 PD。需 >2048（本实验用 4096）让 2K short 留 collocated。
3. **concurrency 64 不可行**（collocated 状态池 64 槽超 24GB）→ 用 C16（在途覆盖足够）。
4. **单卡标定不能乘 4 推四卡**（batch 不同 decode 效率不同）。λ_max 用 4×DP 实测（w0_d0：2.19 req/s，SLO 64/64）取 2.5。
5. **H1 用 cache 65536**（V4 原 98304 触发 P 卡 OOM，65536 解决）。

## 五、可写结论

> 在 RAG 文档问答、文档摘要、代码分析三类负载下，1P+3D 动态 Hybrid 通过 prefill 隔离使 short SLO goodput 提升 75-106%，short TPOT 降低约 40-50%，总吞吐保持 4×DP 的 92-119%（R1/R3 反超）。

## 六、数据文件

- `results/v4/r{1,2,3}_*.json`：各拓扑压测结果（含 metadata）
- `results/v4/r1_h1_fix.json`：H1（cache 65536）R1 结果
- `traces/r{1,2,3}_*.jsonl` + `.meta.json`：冻结 trace（含 SHA256）
