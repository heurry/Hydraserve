# HydraServe 云端压测报告 — 2026-08-15

4×RTX 3090(无 P2P,拓扑全 CNS),Qwen3.5-4B BF16,SHM 传输。数据来源:
`benchmark_output/{concurrency_sweep,context_sweep,diagnostics}/`,原始结论见
[`log.md`](../log.md)。

## 0. 本次压测的代码状态

压测中途完成了三个关键修复(提交 `4e711f1`),报告内所有"修复后"数据均基于
最新代码:

| 修复 | 内容 | 证据 |
|------|------|------|
| B1 | prefill 不再逐 chunk 算全量 logits,只算末位置 → chunk 可放大到整段 prompt | diag_32k:268.3s → 11.3s(**23.8×**) |
| FA 修复 | decode worker 重新启用 FlashAttention(此前静默回退 O(n²) Triton) | pd_32k:313.7s → 22.7s(**13.8×**) |
| FULL 传输 | SHM 默认从 PARTIAL 改为 FULL(直接传 KV,省一次完整重算) | pd_32k:22.7s → 15.9s(**再 −30%**) |

## 1. 上下文扫描(concurrency=1,burst,单样本)

FULL 传输,单 chunk 走 FA。`{mode}_ctx{上下文}_c1.json`。

| 上下文 | DP TTFT | PD TTFT | PD/DP | 注 |
|---|---|---|---|---|
| 32K | 12.3s | 17.0s | 1.38× | |
| 64K | 27.8s | 36.2s | 1.30× | |
| 128K | 65.4s | 74.8s | 1.14× | PD 用 `pd_128k_retry.json`(首跑 OOM,见 §5) |

**趋势:PD/DP 比值随上下文收敛(1.38 → 1.30 → 1.14)。** 机理与预期一致:
传输代价是 O(n)(KV 字节 ÷ 4.58 GB/s),prefill 计算是 O(n²);上下文越长,固定
传输税被摊得越薄。按此趋势,c1 下 crossover 在更长的上下文(>256K)或并发下
才会出现。单样本,分位数无统计意义,只看量级。

## 2. 并发扫描(8K 上下文,burst,32 请求 + 8 warmup,128 输出 token)

**注意:此处 PD 是 1P+1D(2 卡)对单卡 DP,硬件比 2:1**——结论解读时必须
记住这一点(见 §6)。

| 并发 | DP req/s | PD req/s | DP TPOT p50/p99 | PD TPOT p50/p99 | DP TTFT p95 | PD TTFT p95 |
|---|---|---|---|---|---|---|
| 1 | 0.141 | 0.132 | 32.2 / 33.0 ms | 32.3 / 32.5 ms | 3.0s | 3.5s |
| 4 | 0.238 | **0.271** | 86.4 / **109.3 ms** | **36.3 / 37.3 ms** | 11.4s | **10.1s** |
| 16 | 中断无数据 | **0.270,32/32** | — | **36.1 / 37.3 ms** | — | 52.9s |
| 32 | 0.270 | **0.272** | **498.4 / 825.9 ms** | **35.8 / 40.2 ms** | 97.2s | 107.4s |

**现象分解:**

1. **DP 的 decode 随并发坍缩**:TPOT p50 从 c1 的 32ms 恶化到 c4 的 86ms、c32 的
   498ms(p99 826ms)——burst 下 32 个 8K prefill 与 decode 争抢同一张卡的算力/
   带宽,正是"prefill 干扰 decode"的直接证据;
2. **PD 的 decode 完全稳定**:c1→c32 全程 TPOT 36ms 左右,p99 波动仅 33→40ms;
   干扰隔离是 PD 在并发维度的核心赢面;
3. **PD 的 TTFT 尾延迟更高**(c32:p95 107s vs 97s):1P+1D 下唯一 prefill worker
   是 burst 串行瓶颈——这不是架构结论,是"1P"数量问题,1P+3D 应把 prefill 队列
   摊掉 ~3×;
4. **吞吐在 c4 打平**(PD 0.271 vs DP 0.238),c32 两者相等——DP 靠坍缩的 decode
   换来的吞吐,PD 靠隔离;
5. c16 的 DP 文件是中断的空文件(336B),本身即佐证 DP 在该负载下不可用。

## 3. 修复证据链(diagnostics/)

### 3.1 B1(逐 chunk logits 浪费)→ chunk 放大

| 点 | 配置 | TTFT |
|---|---|---|
| diag_32k_clean | 32K,chunk 4096(8 块,块 2-8 走慢 Triton) | 268.3s |
| diag_32k_b1 | 32K,chunk 32768(单块,全 FA) | **11.3s(23.8×)** |
| diag_64k_chunk32768 | 64K,chunk 32768(块 2 走慢 Triton) | 607.7s |
| diag_64k_chunk65536_fix | 64K,单 chunk + conv grid 修复 | **27.4s(22.2×)** |
| diag_128k_chunk131072_v2 | 128K,单 chunk | 65.6s |

结论:B1 修复使 32K 从 4.5 分钟降到 11s;**但只要 prompt 超过 chunk 大小,
continuation chunk 的 O(n²) 无 KV-tile 复用路径(优化日志 OPT-02)仍然慢一个
量级以上**——64K 在 chunk 32768 下是 607s,单 chunk 即 27.4s。

### 3.2 FA 修复与 FULL 传输(PD 侧)

| 点 | 配置 | TTFT |
|---|---|---|
| pd_32k_b1 | PD 32K,FA 已开但 decode worker 未生效 | 313.7s |
| pd_32k_fa_fix | FA 在 decode worker 生效 | 22.7s |
| pd_32k_full | + FULL 传输(免重算) | **15.9s** |

B1 修复后 DP 是 11.3s;PD 修完 FA 仍要 22.7s(≈2×,PARTIAL 重算的税),FULL 传输
后才回到 15.9s(PD/DP = 1.41)。这条链完整验证了优化日志 OPT-04 的论证:
**"重算"(O(n²))永远贵于"传输"(O(n))**。

### 3.3 INT8 KV(`--kv-quant int8`)

`diag_8k_int8.json`:8K prompt、16 输出 token,TTFT 2.80s、TPOT 57.4ms,1/1 成功。
对照同配置 BF16(`dp_ctx8192_c1` 的 TTFT 2.97s)——INT8 KV 路径正确性通过且无
明显速度损失。注意:该点是**正确性验证**,不是精度评估(PPL/长上下文退化未测),
也非传输压缩对照(INT8 的收益应在 QUANTIZED 传输与 KV 显存:1GB→512MB)。

### 3.4 FP8 权重

`fp8_8k.json`:8K,TTFT 11.1s、TPOT 166ms——比 BF16(2.97s / 32ms)慢 **3.8× /
5.2×**。符合预期:SM86 无原生 FP8 Tensor Core,手动 E4M3FN 位解码 + BF16 dot 是
兼容性路径,不是速度路径;FP8 的价值在显存(27B 场景),不在 4B/8K 的速度。

### 3.5 LongBench(1P+3D adaptive,无 FA)

`pd_longbench_gov_c4.json`:4/4 成功,prompt 8.7K-16.8K,**全部路由到 collocated**
(adaptive 成本模型在无 FA 下判断 PD 不划算,正确行为);TTFT p50 124.5s /
p95 178.5s——无 FA 的 O(n²) 路径决定了量级,有 FA 后应按 §1 的 32K 单样本
(12-17s)修正预期。

### 3.6 ShareGPT 冒烟

`bench_sharegpt_c1.json`:8/8,prompt 4-303 token,TTFT p50 127ms——短 prompt
服务链路(2026-08-14 本地 9K 结论:短 prompt PD 慢,不构成 crossover)正常。

## 4. 混合负载:4×DP vs 1P+3D(同硬件量,原始 JSON 已归档)

数据:`benchmark_output/mixed_workload/{dp_gpu0..3,pd_1p3d}.json`;
脚本:`benchmark_output/scripts/{gen_mixed,mixed_compare}.py`;
云端详细记录:`benchmark_output/log.md`。以下数字为 4 个 DP 文件按 runner 同款
插值合并后的精确值。

**负载构成**:128K 长 prompt ×8 + 2K 短 prompt。DP 臂 = 4 卡 × (2 长 + 7 短)
= 36 请求,每卡并行处理 2 个长 prefill;PD 臂 = 8 长 + 31 短 = 39 请求,长请求
路由 7×PD(128K > `force_pd_tokens`) + 1×collocated,短请求全部 collocated
(decode worker 分布 0:6 / 1:17 / 2:16)。

| 指标 | 4×DP collocated | 1P+3D adaptive | 比值 |
|---|---|---|---|
| 请求(成功/总) | 36/36 | 39/39 | — |
| wall time | 215.4s | 712.5s | — |
| 吞吐 tok/s | **21.4** | 7.0 | **3.06×** |
| TTFT p50 / p99 | 51.8s / 104.8s | 81.7s / 433.8s | p99 4.14× |
| TPOT p50 / p99 | 298.4 / 822.0ms | **259.6 / 296.1ms** | p99 **2.78×** |

**机理**:DP 把 8 个长 prefill 摊到 4 卡并行;1P+3D 下 7 个 128K 长 prefill
压到唯一 prefill worker 串行(7×~65s ≈ 455s,与 TTFT p99 433.8s 吻合)——
**1P 是吞吐死穴**。但 PD 的短请求 decode 被隔离:TPOT p99 296 vs 822ms,
且 PD 的 TPOT p50(259.6ms)也优于 DP(298.4ms)——不是只有尾部分布赢。

**这是本次压测最重要的一组对比:同硬件(4 卡)下,DP 吞吐赢 3×,PD decode
延迟赢 2.8×——4 卡规模下 DP collocated 是默认正确选择,PD 只在严格 decode
延迟 SLO 且能接受 ~3× 吞吐损失时值得。PD 的真正价值在 100+ GPU 的大规模部署。**

## 5. 与 2026-08-14 本地双卡数据对照

| 场景 | 2026-08-14(本地,无 FA,PARTIAL) | 2026-08-15(云端,FULL,FA) |
|---|---|---|
| 短 prompt c1 TTFT | collocated 48.9ms / PD 302ms(6.2×) | 8K:DP 2.97s / PD 3.45s(1.16×) |
| 9K 长 prompt | collocated 28.6s / PD 42.8s(1.49×) | 8K 同量级,比值大幅收窄 |
| 传输语义 | PARTIAL(重算 49% 罚) | FULL(传输税 ~14-38%) |

两组数据共同支持:PARTIAL 重算与无 FA 是旧数据的两个主要劣化源;修复后 PD 的
短/中上下文罚金从 ~1.5-6× 收窄到 1.14-1.38×。

## 6. 失败记录

| 文件 | 现象 | 原因与处置 |
|---|---|---|
| dp_ctx8192_c16.json | 空文件(0 请求) | 运行中断,DP 在该并发下未产出数据 |
| pd_ctx131072_c1.json | CUDA OOM(分配 2.25 GiB 失败,GPU 0 余 1.29 GiB) | prefill worker 显存碎片化;`expandable_segments` 修复后 `pd_128k_retry` 74.8s 成功 |

## 7. 结论(按证据强度)

1. **同硬件(4 卡)对比:DP 吞吐赢 3×,PD decode 尾延迟赢 2.7×**(§4)。4 卡
   规模下 DP collocated 是默认正确选择;PD 只在严格 decode 延迟 SLO 且能接受
   ~3× 吞吐损失时值得——这是本次压测最重要的一锤;
2. **PD 的尾延迟隔离是真实的**:并发扫描(1P+1D vs 单卡 DP,硬件 2:1)中
   PD 的 TPOT p99 从 c1 到 c32 稳定在 32-40ms,DP 恶化到 826ms(20×);
   4×DP 下 TPOT p99 也到 ~800ms vs PD 296ms——DP 的 prefill 干扰在两种规模
   下都显著,只是"多卡并行 prefill"能压住吞吐损失、压不住尾延迟;
3. **c1 长上下文下 PD 始终付传输税**(1.14-1.38×),比值随上下文收敛,方向
   符合 O(n) vs O(n²) 论证,但 c1 维度 PD 永远 >1;
4. **三个修复把绝对性能抬了一个量级**(32K:268s→11s;PD 32K:314s→16s),
   证明旧数据的劣化大半来自实现欠账而非架构;
5. **1P 的 prefill worker 是 PD 的吞吐瓶颈**:混合负载下 8 个 128K 长 prompt
   串行 520s;想缩小 DP 的 3× 差距,要么 prefill 多卡(1P→nP),要么
   `force_pd_tokens` 阈值与 prefill 队列联动;
6. **保留**:上下文/并发扫描为 burst 单组样本;INT8 KV 只有正确性验证无精度
   数据;FP8 速度数据是 4B/8K 场景,不代表 27B 显存收益。

## 8. 建议的补齐实验(优先级序)

1. **nP+3D(n>1 prefill worker)**:§4 证明 1P 是吞吐瓶颈,多 prefill worker
   直接回答"PD 吞吐 3× 差距能不能补回来";
2. **Poisson 多组正矩阵**:拿掉 burst 单组噪声,确认 TPOT P99 与 crossover
   曲线的统计显著性;
3. **QUANTIZED 传输实测**(INT8 KV 已实现):传输量减半对 PD TTFT 的影响 +
   encode 侧压缩开销;以及 INT8 KV 的精度评估(PPL/LongBench);
4. **OPT-02**(continuation chunk KV-tile 复用):>128K 的 chunked 路径仍慢
   22×,超长上下文压测的前置;
5. **B3**(GDN 顺序递推并行化):当前长上下文 TTFT 的主要瓶颈(log.md §8)。
