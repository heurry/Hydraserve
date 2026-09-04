# H1 调度问题:诊断 → 修复(8949816)→ 复跑状态

> 记录 H1-4GPU(1P+3D)在 V5 M1 上落后 DP4 的调度根因、作者修复与本机复跑结果。

## 1. 原始诊断(2026-09-03,旧代码,commit e19648b 前后)

当时 H1 落后 DP4(short goodput 只有 DP 的 30-45%)。核对代码后确认:

- **H1(`MultiWorkerGenerationBackend`)无 token 加权负载账本**。路由只用:
  - `_prefill_available()`(二元健康)、`_prefill_long_inflight[i]`(0/1)、
    `_prefill_pending[i]`(计数)、`decode_load`(KV/state 占用率快照)、round-robin。
  - `len(request.token_ids)` 仅用于阈值/元数据,不参与负载比较。
- **DP(`MultiGPUCollocatedBackend`)有 token 加权账本**:`_select_worker` 按
  `_prefill_tokens[i]`(outstanding prefill token,prefill 完成时精确增减)+ `_assigned_work[i]`
  (token 加权 prompt+output)多级比较 → 16 个 long 的 prefill 摊到 4 卡。
- 后果:`work-conserving` 的 short 只看 decode_load 就会抢占唯一 hybrid prefill 槽;
  1P 下 16 个 8-16K long 的 prefill 全串行在 GPU0。

## 2. 作者修复(commit 8949816,"Optimize hybrid PD scheduling")

8949816 实现了缺失的机制(与原始诊断一致):

- **token 加权账本**:`_decode_prefill_tokens[i]` / `_prefill_prefill_tokens[i]`(per-worker
  outstanding prefill token)、`_request_loads`(work=max(1,prompt)+output 同款 DP 模型)。
- **Long-pressure 动态门控**:`_should_defer_long_for_prefill()` + `_hybrid_long_pressure_until`
  + `hybrid_long_pressure_hold_ms`(Long 在等时短暂关闭 Hybrid 的 short 入口)。
- **预算旋钮**:`pd_prefill_token_budget`、`hybrid_short_max_assigned_work`、
  `hybrid_short_max_prefill_backlog_tokens`。
- **INT8 wire**:`--pd-transfer-quant int8`(kv_quant=int8 时 KV 缓存直传,免 dequant→requant)。
- 计划 §5 同步更新,明确"不保证 H1 超过强 DP,以正式复跑为准"。

## 3. 本机补充修复(commit 202bd12)

`--pd-transfer-quant int8` 高并发下 prefill worker 间歇 CUDA IMA/segfault(约 1/4 次,在
`runtime_codec._extract_raw_int8_kv` 的 `stage` sync 处爆出;`CUDA_LAUNCH_BLOCKING` 可规避 →
时序竞态)。修复:prefill worker 释放请求 KV 块前 `torch.cuda.synchronize(cache.device)`,
使块复用严格晚于所有异步访问。M1 单轮未复现,仍需更多轮确认。

## 4. 修复后复跑(新方法学,2026-09-04)

DP4 与 H1-4GPU 均跑在 8949816 + 202bd12,结果见 `results/v5_dp4_h1_summary.md`:

- H1 short SLO goodput 中位从旧配置的 ~15-17 tok/s 提升到 **37.6 tok/s**(DP 44.1,-15%);
- H1 short e2e TTFT **p50 反超 DP**(634 vs 656ms),但 p95/p99 仍差于 DP → 隔离改善中位数、
  尾部仍受 long 排队影响;
- H1 long e2e TTFT p50 6934 vs DP 2779(2.5x),单 Hybrid prefill 瓶颈未变;
- B1 边界验收通过(H1 全 SLO 达标、无 starvation)。

**结论**:调度账本补上后 H1 明显更接近 DP,但 M1 上仍不满足 §8.1(short goodput 未超 DP)。
剩余差距主要是结构性的单 Hybrid prefill bottleneck + short 尾部隔离。计划原文对此已有预期。

## 复现/复跑材料

- 跑法:作者 `scripts/run_dp4_v5.sh`、`scripts/run_h1_4g_v5.sh`;本机剩余 seed 脚本
  `scripts/run_h1_v2_remaining.sh`、`scripts/run_dp4_v2.sh`。
- 对比:`scripts/compare_dp_h1.py`。
- 结果:DP4 `results/v5_dp4/`,H1 `results/v5_h1_4g/`。
