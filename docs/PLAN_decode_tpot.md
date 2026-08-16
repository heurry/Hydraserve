# 帕累托点攻关计划 — decode 侧 TPOT

> 目标:64K 混合负载(8 长 64K + 64 短 2K,128 输出)达到 **吞吐 ≥18 tok/s 且 TPOT p99 <400ms**。
> 状态:吞吐已达标,TPOT 差 ~11%,瓶颈在 decode 侧。本文档记录根因与攻关计划。

## 1. 现状(2026-08-16 实测)

| 拓扑 | 吞吐 | TPOT p99 | 结论 |
|---|---|---|---|
| 4×DP | 50.1 tok/s | ~457ms | 吞吐达标,TPOT 差 57ms |
| 2P+2D | 31.1 tok/s | 447ms | 吞吐达标,TPOT 差 47ms |
| 3P+1D | OOM | — | 撞内存墙 |

配套报告:[BENCHMARK_2026-08-16_PARETO.md](BENCHMARK_2026-08-16_PARETO.md)。

## 2. 根因:64K+128 是 decode 受限,不是 prefill 受限

长/短 TPOT 分解证明瓶颈在 decode:

| 场景 | 长请求 TPOT p99 | 短请求 TPOT p99 |
|---|---|---|
| 2P+2D(每 D 卡 ~4 长) | 436ms | 445ms |
| DP gpu0(2 长) | 223ms | 298ms |
| DP gpu2(3 长) | 454ms | 454ms |

**TPOT 由 decode batch 里的长请求数量主导**:每 D 卡 decode batch 挤 N 个 64K 长请求时,
每步 attention 扫 N×64K token,batch 内每个 token(不分长短)都被拖到 ~450ms。这与
prefill 无关——所以 plan 原定的 W4/W5/W6(prefill 侧杠杆,为 128K 校准)对这个负载失效。

## 3. 已尝试的杠杆(结论)

| 项 | 结果 | 根因 |
|---|---|---|
| W4(P 卡服务短请求) | **退化** 23 tok/s / TPOT 1047ms | prefill worker 串行命令循环,短请求堵在长 prefill 后;decode 拆 4 卡后每卡 batch 变小 |
| W6(continuation chunk KV-tile 复用) | 阻塞 | 方向错(W6 优化 prefill TTFT,非 decode TPOT);且 `flash_attn_with_kvcache` 的 `page_block_size` 必须是 256 的倍数,与 block_size=16 不匹配 |

## 4. 三条 decode 侧杠杆(按优先级)

| 杠杆 | 预期 TPOT | 复杂度 | 说明 |
|---|---|---|---|
| **1. decode attention kernel 优化** | 1.5–2×(~450→~250ms) | 中 | 首选,见 §5 |
| **2. INT4 KV** | ~1.5×(带宽减半) | 中 | 现有 INT8 之上再压一半,代价是精度;需先做精度评估 |
| **3. 投机解码** | ~2×(一步出 2 token) | 高 | 需 draft 模型 + 验证,工程量最大 |

三者可叠加;第 1 项是投入产出比最高的首攻点。

## 5. 首选方案:decode attention kernel 优化

### 5.1 现状

`hydraserve/kernels/paged_attention.py::_paged_attention_kernel`:
- grid = `(batch, query_heads)`,每个 (request, head) 一个 program;
- 每个 program **串行**扫完整 context,`BLOCK_T=16` 逐 tile 载入 KV,online softmax;
- 64K context = 4096 次 tile 迭代,memory-bound(每步读 N×64K×head_dim×2)。

### 5.2 优化方向(FlashDecoding 式)

1. **split-K**:把 KV 按 K 段切开,K 个 program 各算一段的 (max, sum, acc),再归约——把
   64K 串行扫描并行到 K 个 SM;
2. **加大 BLOCK_T**(16 → 64/128):减少循环迭代、提升访存合并;
3. **向量化 + eviction hint**:`tl.load` 用 `evict_first`(KV 只读一遍)、`evict_last`(query/acc 复用);
4. 按 head_dim=256 / num_kv_heads=4 / block_size=16 调 `num_warps` 与 tile 形状。

### 5.3 验证(单卡即可)

- **correctness**:新 kernel vs 现有 `paged_attention`(正确但慢)+ `reference.py` CPU 参考,BF16 对齐;
- **微基准**:64K context 的 decode batch,实测每步 attention 耗时,扫 BLOCK_T / num_warps / split-K;
- **端到端**:达标后重跑 64K 混合负载,确认 TPOT p99 <400ms(这一步才需要整机)。

### 5.4 里程碑

- M1:kernel correctness 对齐 + 单卡微基准测出 ≥1.3× 加速;
- M2:端到端 TPOT p99 压到 <400ms,画出 4×DP / 2P+2D 的新帕累托点;
- M3(可选):叠加 INT4 KV 或投机解码,继续逼近。
