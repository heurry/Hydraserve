# HydraServe 单卡优化前后基准（2026-08-22）

## 环境与口径

- GPU：RTX 3090 24 GiB，GPU 0
- 模型：`/mnt/nvme-data/models/LLM_model/Qwen3.5-4B`，BF16
- 数据：GSM8K，2 warmup + 8 measured，16 output tokens，prompt cap 512
- cache tokens：8192；FlashAttention 关闭
- 历史基线：2026-08-14 本机同配置结果
- 当前值：首次运行用于生成/缓存新执行路径，表格取随后两次 kernel cache 已热的均值；
  CUDA Graph 缓存在进程内，新的 benchmark 进程仍会重新捕获未覆盖的图形状

## 历史基线 vs 当前优化后稳态

| 单卡并发 | 版本 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | 2026-08-14 基线 | 47.330 | 48.87 ms | 17.46 ms | 310.93 ms |
| C1 | 2026-08-22 optimized | 60.621 | 36.27 ms | 13.79 ms | 247.64 ms |
| C1 | 变化 | **+28.1%** | **-25.8%** | **-21.0%** | **-20.4%** |
| C4 | 2026-08-14 基线 | 124.878 | 108.13 ms | 25.33 ms | 501.79 ms |
| C4 | 2026-08-22 optimized | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms |
| C4 | 变化 | **-14.7%** | **+20.2%** | **+13.4%** | **+14.9%** |

初步结论：当前版本显著改善单请求 C1，但默认 Graph 路径下 C4 相比历史基线存在回退。
后续 Graph on/off 隔离已证明回退来自计时区间内的首次图捕获，见下文“C4 回退根因”。

## 当前代码 fused/unfused 直接 A/B

该 A/B 在同一代码、同一机器、已完成编译缓存后运行，只切换 `fuse_projections`。

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | unfused | 59.191 | 35.04 ms | 14.35 ms | 255.50 ms |
| C1 | fused（两次均值） | 60.621 | 36.27 ms | 13.79 ms | 247.64 ms |
| C1 | fused 变化 | **+2.4%** | +3.5% | **-3.9%** | **-3.1%** |
| C4 | unfused（两次均值） | 109.115 | 142.44 ms | 29.90 ms | 580.09 ms |
| C4 | fused（两次均值） | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms |
| C4 | fused 变化 | **-2.3%** | **-8.7%** | **-3.9%** | **-0.6%** |

融合投影降低了 TPOT/中位延迟，但默认 Graph 路径下 C4 的总 wall time/吞吐略差。
后续真实权重 GEMM 微基准显示 batch 2–4 的主要合并投影均更快；端到端异常主要由
CUDA Graph 首次捕获和调度尾部造成，暂时没有证据支持在 C4 关闭投影融合。

## C4 回退根因：CUDA Graph 冷捕获与形状碎片化

同一当前 fused 代码，仅关闭 CUDA Graph 后的结果：

| C4 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---:|---:|---:|---:|---:|
| fused + Graph on（两次均值） | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms | 约 1.20 s |
| fused + Graph off | **148.225** | **92.45 ms** | **21.42 ms** | **423.69 ms** | **0.864 s** |
| 2026-08-14 历史基线 | 124.878 | 108.13 ms | 25.33 ms | 501.79 ms | 1.025 s |

当前 Graph on 相对 Graph off 吞吐低约 28.1%，而 Graph off 比历史基线高约 18.7%。
本次 C4 共出现 8 个 `(batch_size, block_table_width)` Graph key；`--warmup 2` 串行
执行，只覆盖 batch 1，正式计时阶段仍首次捕获 6 种形状。每种形状会执行 3 次完整
decode warmup 和 1 次完整 graph capture，并快照/恢复 recurrent、convolution 和 KV
页。对于 8 请求、16 output tokens 的短测试，捕获成本无法摊薄。

因此短突发 C4 在完善 Graph 预热前可设置 `HYDRASERVE_CUDA_GRAPH=0`。生产长稳态负载
应改为并发感知预热、常用 shape bucket 预捕获，或新 shape 多次出现后再捕获，而不是
永久关闭 Graph。完整排障过程见
[优化与排障日志](../docs/OPTIMIZATION_LOG_2026-08-22.md)。

## 第二轮优化：Graph 策略与融合 activation（最终当前结果）

根据上述根因继续实施：

- block-table width 按 2 的幂 bucket，减少 Graph key 数量；
- 新 shape 默认观察 16 次后才捕获，避免一次性 shape 在请求关键路径支付捕获成本；
- capture 前完整 warmup 从 3 次降为 1 次；
- MLP `SiLU(gate) * up` 合并为一个 Triton kernel；
- GDN `sigmoid(beta)` 与 decay 参数化合并为一个 Triton kernel。

默认配置复测（两次均值）：

| 并发 | 第一轮 fused + 立即 Graph | 第二轮最终当前 | 变化 |
|---|---:|---:|---:|
| C1 output tok/s | 60.621 | **65.940** | **+8.8%** |
| C1 TPOT P50 | 13.79 ms | **13.43 ms** | **-2.5%** |
| C4 output tok/s | 106.578 | **146.272** | **+37.2%** |
| C4 TPOT P50 | 28.72 ms | **21.73 ms** | **-24.3%** |
| C4 Latency P50 | 576.35 ms | **428.88 ms** | **-25.6%** |

第二轮默认 C4 已接近显式 Graph-off 的 150.226 tok/s，同时不牺牲 C1。相对
2026-08-14 历史基线，最终当前 C1/C4 吞吐分别提升约 **39.3%/17.1%**。

固定 128-token INT8 C1 复测：

| 模式 | output tok/s | TTFT P50 | TTFT P95 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|---:|
| 最终当前 Graph off | 54.651 | 56.51 ms | 62.27 ms | 17.98 ms | 2339.84 ms |
| 最终当前 Graph on | **58.262** | 56.54 ms | 62.07 ms | **16.84 ms** | **2195.14 ms** |

Graph 在长稳态负载中的净吞吐收益为 **6.6%**；旧策略 Graph-on 的 TTFT P95
244.62 ms 已降至 62.07 ms，说明新策略同时改善冷形状尾延迟并保留 replay 收益。

## 第三轮优化：批量贪心采样

默认贪心、无惩罚、无需 logprobs 的请求现在对整个 batch 执行一次 `argmax` 和一次
host 回传，不再逐行 clone logits、计算无用 `log_softmax`、准备默认惩罚 tensor 和同步。
同代码 A/B 使用 `HYDRASERVE_BATCHED_GREEDY=0` 恢复旧路径，每种配置运行两次：

| 并发 | 模式 | output tok/s | TPOT P50 | Latency P50 | wall time |
|---|---|---:|---:|---:|---:|
| C1 | 旧逐行 | 66.720 | 13.24 ms | 240.52 ms | 1.9185 s |
| C1 | 批量 | 66.742 | 13.24 ms | 240.53 ms | 1.9178 s |
| C4 | 旧逐行 | 149.798 | 20.99 ms | 418.64 ms | 0.8545 s |
| C4 | 批量 | **151.039** | **20.86 ms** | **418.41 ms** | **0.8475 s** |

C1 持平；C4 吞吐提升 **0.83%**，TPOT P50 降低 **0.58%**。采样函数微基准中 batch=4
由 0.1177 ms 降到 0.0238 ms（-79.8%），端到端收益较小是因为模型 forward 仍占主要成本。

本轮也评估了 fused residual-add + RMSNorm。C4 两次均值吞吐 147.577→147.526 tok/s，
TPOT 21.49→21.64 ms，未获得净收益，故源码候选已回退。详情和原始文件见
[优化与排障日志](../docs/OPTIMIZATION_LOG_2026-08-22.md)。

## 第四轮优化：P1 GDN kernel 收敛

对实际剩余的逐层 GDN 热点一次性实施：

- causal-conv 从 one-program-per-channel 改为 256-channel blocked grid；
- conv 直接写连续 Q/K/V，消除 mixed 后的布局复制；
- recurrent 直接把 16 compact Q/K heads 映射到 32 value heads，去掉两次
  `repeat_interleave`；
- sequence=1 decode 使用无动态 token loop 的专用 recurrent kernel；
- `HYDRASERVE_GDN_KERNEL=legacy` 恢复整组旧路径用于严格 A/B。

每种模式两次反序均值：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | legacy | 66.690 | 34.81 ms | 13.28 ms | 240.92 ms |
| C1 | P1 final | **71.654** | **24.55 ms** | **12.89 ms** | **220.85 ms** |
| C1 | 变化 | **+7.44%** | **-29.47%** | **-2.92%** | **-8.33%** |
| C4 | legacy | 150.284 | 93.60 ms | 20.96 ms | 417.04 ms |
| C4 | P1 final | **176.147** | **68.43 ms** | **18.93 ms** | **353.59 ms** |
| C4 | 变化 | **+17.21%** | **-26.90%** | **-9.70%** | **-15.21%** |

相对 2026-08-14 历史基线，当前 C1/C4 吞吐累计提高约 **51.4%/41.1%**。长 prefill 的
多 token recurrence 仍保持精确串行，尚未实现 FlashInfer 风格 chunk scan；这里的
sequence=1 specialization 专门针对每个 decode step。

## 第五轮优化：P2 runtime 固定开销

第一批完成 native logits、per-layer weight slots、热路径 import/contiguous 清理、positions 与
state slot_ids buffer 复用；CUDA P2P receive 从 CPU event synchronize 改为 GPU stream wait。

最终 native logits 两次均值：

| 并发 | P1 final | P2 第一批 | 变化 |
|---|---:|---:|---:|
| C1 output tok/s | 71.654 | **71.788** | **+0.19%** |
| C1 TTFT P50 | 24.55 ms | 24.62 ms | +0.28% |
| C1 TPOT P50 | 12.89 ms | **12.88 ms** | -0.11% |
| C4 output tok/s | 176.147 | **177.836** | **+0.96%** |
| C4 TTFT P50 | 68.43 ms | 68.84 ms | +0.60% |
| C4 TPOT P50 | 18.93 ms | **18.81 ms** | **-0.64%** |

在最终相同代码上只切换 `HYDRASERVE_FP32_LOGITS=1`，native BF16 logits 的 C1 吞吐
提高 0.30%；C4 吞吐持平但 TTFT P50 降低 1.04%，并把 Graph static logits buffer 减半。
相对历史基线，当前 C1/C4 吞吐累计提高约 **51.7%/42.4%**。

SHM codec/poll 和 runtime codec host staging（原清单 #17～#19）属于每请求 PD 传输协议，
将在独立阶段修改和验证，不能通过简单删除同步来伪异步化。完整逐项状态见
[优化与排障日志](../docs/OPTIMIZATION_LOG_2026-08-22.md)。

## 第六轮优化：P3 边际清理

本轮合入两项不会改变模型计算图的低风险修改：删除 stdlib HTTP 无缓冲 `_SocketWriter` 上
每个 SSE event 的空 `flush()` 调用，以及在三处内存预算代码中直接使用 `dtype.itemsize`。

GDN output 预分配完成两轮反序 A/B，但 C1/C4 吞吐分别下降 0.46%/1.36%，已撤回。当前
RTX 3090（SM 8.6）不支持目标 Hopper FP8 Tensor Core 路径；Marlin INT4 则需要真实 AWQ
模型和独立架构专项，未把未覆盖代码算作优化完成。

最终代码两次均值：

| 并发 | P2 第一批 | P3 最终 | 变化 | TTFT P50 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 71.788 tok/s | 71.475 tok/s | -0.44% | 25.72 ms | 12.95 ms | 222.20 ms |
| C4 | 177.836 tok/s | 177.895 tok/s | +0.03% | 68.26 ms | 18.82 ms | 350.97 ms |

两项保留修改只影响 SSE Python 固定调用和初始化期 dtype 查询，因此端到端推理吞吐应持平；
该组结果不声明新增吞吐收益。详细 A/B、否决原因和原始文件见
[优化与排障日志](../docs/OPTIMIZATION_LOG_2026-08-22.md)。

## INT8 KV CUDA Graph 净收益

历史 C1 文件没有开启 INT8 KV。为隔离本次 P0 修复，使用固定 128-token synthetic prompt、
128 output tokens、C1，对比当前代码 graph on/off；graph off 模拟旧版 INT8 eager 路径。

| 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|
| INT8 eager（graph off） | 53.001 | 57.95 ms | 18.55 ms | 2414.03 ms |
| INT8 CUDA Graph on | 56.047 | 57.60 ms | 17.24 ms | 2249.83 ms |
| 变化 | **+5.7%** | -0.6% | **-7.1%** | **-6.8%** |

CUDA Graph 对该单卡短上下文 C1 的收益是 5–7%，不是分析报告预估的 2–3 倍；更大的收益
需要在高并发、长上下文和 4 卡正式负载上重新测量。不同 block-table 宽度的首次图捕获仍会
抬高 TTFT 尾部，服务预热策略需要覆盖常用宽度。

## FlashAttention 长 prompt A/B

安装 `flash-attn 2.8.3.post1` 后，使用 Graph off、C1、4 条 1024-token synthetic prompt、
chunk/page=256、2 warmup、16 output tokens 做热态 A/B。paged KV、varlen GQA 和 reference
数值对照测试均通过。

| 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|
| FlashAttention off | 37.533 | 212.03 ms | 14.065 ms | 422.73 ms |
| FlashAttention on | **38.416** | **201.52 ms** | 14.080 ms | **413.21 ms** |
| 变化 | **+2.35%** | **-4.96%** | +0.11% | **-2.25%** |

收益集中在 prefill/TTFT，TPOT 基本持平。原始结果：
`2026-08-22_synthetic1024_4b_flash_ab_off_final.json`、
`2026-08-22_synthetic1024_4b_flash_ab_on_final.json`。

## 原始结果

- `2026-08-14_gsm8k_4b_collocated_c1.json`
- `2026-08-14_gsm8k_4b_collocated_c4.json`
- `2026-08-22_gsm8k_4b_collocated_c1_optimized_run2.json`
- `2026-08-22_gsm8k_4b_collocated_c1_optimized_run3.json`
- `2026-08-22_gsm8k_4b_collocated_c1_current_unfused.json`
- `2026-08-22_gsm8k_4b_collocated_c4_optimized_run2.json`
- `2026-08-22_gsm8k_4b_collocated_c4_optimized_run3.json`
- `2026-08-22_gsm8k_4b_collocated_c4_fused_graph_off.json`
- `2026-08-22_gsm8k_4b_collocated_c4_current_unfused.json`
- `2026-08-22_gsm8k_4b_collocated_c4_current_unfused_run2.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_on.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_off.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_run1.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_run2.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_run1.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_run2.json`
- `2026-08-22_gsm8k_4b_c4_fused_kernels_graph_off.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_policy_fused_kernels_on.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_policy_fused_kernels_off.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_final.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_final.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_final_run2.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_off.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_off_run2.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_on.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_on_run2.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_off.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_off_run2.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_on.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_on_run2.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_legacy.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_legacy_run2.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_final.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_legacy.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_legacy_run2.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_final.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_final_run2.json`
- `2026-08-22_gsm8k_4b_c1_p2_final_fp32.json`
- `2026-08-22_gsm8k_4b_c1_p2_final_fp32_run2.json`
- `2026-08-22_gsm8k_4b_c1_p2_buffers.json`
- `2026-08-22_gsm8k_4b_c1_p2_final_native_run2.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_fp32.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_fp32_run2.json`
- `2026-08-22_gsm8k_4b_c4_p2_buffers.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_native_run2.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_alloc.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_alloc_run2.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_prealloc.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_prealloc_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_alloc.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_alloc_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_prealloc.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_prealloc_run2.json`
- `2026-08-22_gsm8k_4b_c1_p3_final.json`
- `2026-08-22_gsm8k_4b_c1_p3_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_final.json`
- `2026-08-22_gsm8k_4b_c4_p3_final_run2.json`
