# HydraServe 优化日志

> 记录每一项优化的完整过程:现象 → 定位 → 根因 → 方案 → 验证数据 → 结论。
> 状态标记:`待办` / `进行中` / `已完成` / `放弃(附原因)`。

## 记录格式(新条目模板)

```markdown
## OPT-NN — 标题

- **日期**:YYYY-MM-DD
- **现象**:观测到的数据(命令、耗时、显存等)
- **定位**:文件:行号
- **根因**:
- **方案**:
- **验证**:对照实验与数值
- **结论**:
- **状态**:
```

---

## 已登记条目

### OPT-01 — prefill 全 chunk logits 浪费(高优先)

- **日期**:2026-08-15
- **现象**:云端 4×3090 压测,32K prompt TTFT ~160s;`--prefill-chunk-size 32768` 直接 OOM(logits 32768×248320 张量)
- **定位**:[hydraserve/model/runtime.py:265](../hydraserve/model/runtime.py#L265) `forward` 对每个 chunk 计算 `[chunk, vocab]` logits 并 `.float()`(FP32,chunk=4096 时约 4 GiB/张);[runtime.py:284](../hydraserve/model/runtime.py#L284) prefill 循环只保留最后一份,中间全部浪费
- **根因**:只有最后 chunk 的末位置 logits 被消费(首 token 生成),其余位置/其余 chunk 的 logits 是纯浪费;chunk 大小被这个张量卡死,导致 32K 只能切 8 块、只有第 1 块走 FlashAttention
- **方案**:`forward` 增加"跳过 lm_head / 只算末位置"选项;prefill 中间 chunk 完全跳过 lm_head,最后 chunk 只算末位置 logits
- **预期收益**:单 chunk 显存峰值 ~4 GiB → 忽略不计;`--prefill-chunk-size` 可提至 32768 → 32K prefill 全程 FlashAttention
- **验证**:本地 4B 与旧路径逐位对齐 + 云端 32K c1 TTFT 对照
- **状态**:待办(暂缓,用户要求先不动代码)

### OPT-04 — SHM 后端 transfer_mode 硬编码 PARTIAL(高优先,云端实测引出)

- **日期**:2026-08-15
- **现象**:云端压测 PD ≈ 2× DP;32K 下 PARTIAL 传输省 206ms 带宽,却换来 decode 端完整 prefill 重算(~11s FA 开 / 313s FA 关)
- **定位**:[hydraserve/transfer/backend.py:126-127](../hydraserve/transfer/backend.py#L126-L127) SHM 后端 `transfer_mode` 无条件返回 PARTIAL;[hydraserve/engine/pd_worker.py:151-157](../hydraserve/engine/pd_worker.py#L151-L157) PARTIAL 时 decode worker 重跑完整 `runtime.prefill`(GDN 在内)重建 KV
- **根因**:PARTIAL 的 tradeoff 前提是"重算便宜";但混合注意力模型里 full-attn KV 的输入依赖所有前序 GDN 层的中间激活(递推状态无法回放时间轴),重算 = 完整 O(n²) prefill,远贵于 O(n) 传输(SHM 1GB ≈ 218ms << 11s 重算)
- **方案**:让 SHM 后端 transfer_mode 可配置(FULL / QUANTIZED / PARTIAL),配置项穿过 PDWorkerConfig → backend;两端链路已就绪(prefill 端 extract_kv、decode 端 install_kv、INT4 codec 已单测)
- **预期收益**:PD 侧省掉一次完整 prefill;32K FULL ≈ +218ms 传输 vs -11s 重算
- **验证**:云端 32K DP vs PD 对照(FULL 与 QUANTIZED 各一组);QUANTIZED 需计入 encode 侧 INT4 压缩开销
- **状态**:待办

### OPT-05 — 统一池:nP 多 prefill worker + P 卡服务短请求(高优先,压测结论引出)

- **日期**:2026-08-15
- **现象**:混合负载(8×128K 长 + 31×2K 短)实测:4×DP 21.4 tok/s 但 TPOT p99 822ms;1P+3D adaptive 7.0 tok/s 但 TPOT p99 296ms。吞吐差根源 = 8×65s=520s 长 prefill 只有 1 张卡承担(串行 ~455s);D 卡在短请求上兼作 DP(route_counts collocated 32 / pd 7)已自动生效,非路由问题
- **根因**:prefill 并行度不足 + 角色静态绑定(prefill worker 只做 PD prefill,不接 collocated 短请求)
- **方案(统一池)**:每卡同时具备 P/D/DP 能力,逐请求分配——长请求走 PD(prefill 摊到 nP),短请求落"动态空闲"卡(含 P 卡);路由的 decode_load 输入扩展为 prefill+decode 双向负载
- **第一约束(KV 内存墙,在时间墙之前)**:4B KV = 32KB/token(8 层×4 KV heads×256 head_dim×4B,config.py:121);128K 请求 = 4GB BF16 / 2GB INT8。3P+1D 单 D 卡 8 长 = 32GB ❌(INT8 24.5GB 也超);**2P+2D+INT8 每 D 卡 4×2GB+8.5GB 权重 ≈ 16.5GB ✅ 是最内存可行组合**。注意:已归档混合负载的 prompt 为重复文本拼接(gen_mixed.py),prefix cache 去重掩盖了 KV 压力;真实 distinct 128K 请求会立即撞墙
- **预期收益**:2P+2D 关键路径 ~260s → 吞吐 ~19 tok/s + 短请求 TPOT 干净;后续 3P+1D 需 eviction 前置
- **验证**:同混合负载(distinct prompt 版)跑 2P+2D+INT8,与 4×DP(21.4/822ms)、1P+3D(7.0/296ms)画帕累托前沿
- **状态**:阶段 1 已实现(上游 de0f0ee:prefill_devices + n 进程 + round-robin + 恢复);遗留 Bug A(第 2 个 P worker 启动崩溃)、Bug B(prefix cache hit_rate=0);阶段 2(P 卡服务短请求 + prefill-load 路由)未做。详细计划见 [[docs/IMPROVEMENT_PLAN.md]]

### OPT-02 — continuation chunk paged attention 无 KV-tile 复用(中优先)

- **日期**:2026-08-15
- **现象**:32K/4096 分块时 chunk 2-8 走 Triton paged attention,TTFT ~160s
- **定位**:[hydraserve/kernels/paged_attention.py](../hydraserve/kernels/paged_attention.py) `paged_prefill_attention` 复用 decode 式逐查询 program
- **根因**:每个查询位置独立扫描历史块,没有 FlashAttention 式的跨查询 tile KV SRAM 复用
- **方案**:把 KV-tile 复用写进带历史的 paged kernel(或 block-causal 结构)
- **预期收益**:continuation chunk 加速一个量级;OPT-01 完成后仅超长 prompt 走此路径
- **状态**:待办

### OPT-03 — GDN 顺序递推串行瓶颈(研究性)

- **日期**:2026-08-15
- **现象**:长上下文 prefill 中 24 个 GDN 层 × 32K 步顺序递推占比高
- **定位**:[hydraserve/kernels/gdn.py](../hydraserve/kernels/gdn.py) `for token in tl.range(0, sequence, 1)`
- **根因**:delta rule 的串行数据依赖;每 (batch, head, value_block) program 顺序跑完整序列
- **方案**:chunkwise 递推(状态转移矩阵化并行)/ tl.range 软流水与 unroll
- **状态**:待办

---

## 历史记录

(从 OPT-01 开始,按模板逐条追加)
