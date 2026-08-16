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

- ✅ **M1 已完成(2026-08-16,单卡)**:
  - correctness:12 个配置(block_t 64/128 × num_splits 2/4/8 × num_warps 4/8)
    全部对齐旧 kernel 与 CPU reference(max diff ≤5e-4);新增 6 个 pytest 用例;
  - 微基准(64K context):batch 1/4/8 = **15.3× / 4.83× / 4.43×**;
  - 端到端 A/B(单卡 64K 混合负载,c4、固定交错序、prefix cache 开):
    **TPOT p99 448.4 → 346.4ms(−23%,<400ms)**,TPOT p50 162.5 → 66.5ms(−59%),
    长请求 TPOT p50 178 → 71.7ms(−60%),吞吐 17.0 → 21.4 tok/s(+26%)。
    注意:本地 A/B 是 c4 + 固定顺序,云端标准是 **c16 + seed 42 随机打乱**,
    方向可迁移,收官数据必须按云端标准配置重跑。
- **M2(待云端)**:按 [BENCHMARK_2026-08-16_PARETO.md](BENCHMARK_2026-08-16_PARETO.md)
  的标准配置(72 条 seed 42 随机打乱、c16、int8 KV、prefix 4000、chunk 65536、
  burst、128 输出)重跑 **2P+2D 与 4×DP**,确认两者 TPOT p99 <400ms 且吞吐
  ≥18 tok/s,更新帕累托图。A/B 开关:`HYDRASERVE_PAGED_ATTENTION=reference`
  回退旧 kernel。
- M3(可选):叠加 INT4 KV 或投机解码,继续逼近。

## 6. 杠杆 A 结果与 B3-lite 结论(2026-08-16,单卡)

**A(CUDA Graph decode)已实现并位级验证**:

- [runtime.py](../hydraserve/model/runtime.py) 按 (batch, 表宽) 惰性捕获 decode 步;
  静态输入缓冲(token/位置/页表/长度)+ replay;捕获副作用(池槽位、KV 页、
  batch 工作区)快照/恢复——捕获期 warmup 会真实执行事务 4 次,不恢复即污染状态;
- 真实 4B 上 8 步 graph vs eager:logits/recurrent **位级一致**(diff 0.0);
- 收益 7.5%(2K/4K 上下文 19.3→17.9 ms/步),低于预期:profile 中的 host 开销
  大头在 decode_batch 之外(serving loop 调度/采样/tokenizer)。若继续收割,
  下一项是采样路径优化或把调度纳入图;默认开,`HYDRASERVE_CUDA_GRAPH=0` 回退。

**B3-lite 已穷尽,无收益**:

- `loop_unroll_factor` 在 triton 3.0 不受支持;`num_stages=2` 仅 15.52→15.41s
  (0.7%,噪声)——逐 token 串行依赖使流水化无效;
- 结论:B3 的真实解是 **chunkwise 仿射扫描**(S' = (aI − βkkᵀ)S + βkvᵀ 的关联
  扫描,chunk 内并行,参考 fla 的 chunked delta rule),研究级 kernel,建议独立
  立项,不在本轮继续。

## 7. 上线后待办:长上下文 router profile 拟合(#3)

**现象**:默认 profile(partial_transfer_default)的二次曲线由短 prompt(≤9K)数据
拟合,外推到 64K/128K 严重失准;云端混合负载实测 17 个请求触发 drift 保护
fail-closed 到 collocated。

**任务**:用已有 32K/64K/128K 三个长度实测点(context_sweep + 单请求诊断数据)
跑 `python -m hydraserve fit-router-profile`,产出的 profile 作为 `--router-profile`
上线;prefill-load 感知路由上线后,拟合样本需同时覆盖低/高 prefill-load 工况。
验收:64K 混合负载下 route_reason 不再出现 cost_model_drift,且长请求路由与
成本模型预测一致。
