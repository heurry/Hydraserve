# HydraServe V5 压测方案：真实混合 RAG 与 Long-heavy 边界

## 1. 目标

V5 不再用随机 Token 的 R1/R2/R3 直接支撑“真实客服 RAG、文档摘要、代码分析”结论。旧 trace
保留为机制压力测试，V5 的正式结论只回答两个问题：

1. 在真实内容、持续混合到达、Short SLO 优先的客服 RAG 负载下，修复后的 DP 与 H1 谁更好；
2. 当 Long 密集或成对到达时，单 Hybrid 的排队、回退和适用边界是什么。

正式最小矩阵只比较 `DP` 与 `H1`，共 8 次运行。H2、静态 PD、vLLM/SGLang、Prefix Cache、
多档 chunk 和完整负载扫描均不进入 V5 首轮。

## 2. 测试分层

### 2.1 本机双卡门禁，不产生简历数字

本机 2×RTX 3090 使用：

- `DP2`：GPU 0、1 都是修复后的 collocated DP worker；
- `H1-2GPU`：GPU 0 为 Hybrid，GPU 1 为 D-bound worker。

只运行两个 trace 的 seed42，每个拓扑一次，共 4 次。目标是验证正确性、路由、状态迁移、
Long 回退和统计口径。双卡结果只写“趋势”，不能替代四卡正式数字。

### 2.2 四卡正式矩阵

目标机器为 4×RTX 3090：

- `DP4`：`--dp-devices 0 1 2 3`；
- `H1-4GPU`：`--prefill-devices 0 --decode-devices 1 2 3`。

| 负载 | Seed | DP4 | H1-4GPU | 运行数 |
|---|---|---:|---:|---:|
| M1 真实混合 RAG | 42、43、44 | 3 | 3 | 6 |
| B1 Long-heavy 边界 | 42 | 1 | 1 | 2 |
| 合计 | - | 4 | 4 | 8 |

## 3. Trace 定义

所有拓扑必须重放同一份 JSONL；Prompt、Token 数、到达偏移、采样参数和 trace SHA256 必须一致。
使用 greedy sampling，正式真实模式设置 `ignore_eos=false`。

### 3.1 M1：真实混合 RAG

目标是模拟同一在线服务中，大量普通 RAG 请求与少量长上下文 RAG 请求持续混部，而不是先集中
注入 Short、再刻意注入 Long。

| 属性 | Short RAG | Long RAG |
|---|---:|---:|
| 请求数 | 48 | 16 |
| 比例 | 75% | 25% |
| Prompt | 真实问题 + 真实检索文档 | 真实问题 + 更多真实检索文档 |
| 输入长度目标 | 1K～4K Token | 8K～16K Token |
| `max_new_tokens` | 256 | 512 |
| EOS | 启用 | 启用 |

到达规则：

- 测试窗口 120 秒，结束后允许 drain；
- Short 与 Long 分别由独立、带 seed 的 Poisson 过程生成；
- 两类请求都覆盖完整 0～120 秒窗口；
- 不允许所有 Short 在前 20 秒结束，也不允许 Long 只在预选干扰时刻出现；
- 三个 seed 使用相同分布，但使用不同内容采样与到达偏移。

真实内容优先来自脱敏客服/知识库记录；没有业务数据时，使用真实多文档问答数据构造
`system + question + retrieved documents + answer instruction`，不能使用随机词表 Token。

### 3.2 B1：Long-heavy 边界

目标不是让 H1 获胜，而是验证单 Hybrid 饱和后的排队、5 秒回退和无饥饿性。

| 属性 | Short RAG | Long RAG |
|---|---:|---:|
| 请求数 | 8 | 8 |
| 输入/输出分布 | 与 M1 相同 | 与 M1 相同 |
| 测试窗口 | 60 秒 | 60 秒 |

Long 以四个二请求 burst 到达，例如 `10s、25s、40s、55s` 各同时到达两条；Short 在完整
60 秒窗口内 Poisson 到达。H1 使用当前 `--hybrid-long-overflow-ms 5000`，必须记录每条 Long
是等待 Hybrid、走 PD，还是超时回退到 D-bound collocated 路径。

## 4. 正式开跑前的代码门禁

现有 benchmark 尚不能直接产出可信的 V5 真实业务结果，必须先完成：

1. 增加从真实数据记录冻结 JSONL trace 的生成器，而不是只支持 synthetic `--trace-out`；
2. 校验 trace 的实际重编码 Token 数、到达窗口、类别计数和 SHA256；
3. 将 Short SLO 从硬编码 `TTFT=5s、TPOT=200ms` 改为可配置；
4. SLO 使用端到端 TTFT，即包含 client executor queue；
5. `ignore_eos=false` 时，正常 EOS 必须算完整成功，不能因为未生成满 `max_new_tokens` 被判失败；
6. 增加 Long SLO、最大 admission wait、Hybrid overflow/fallback 和 starvation 统计；
7. 保证 route reason、worker owner、实际输出 Token 数及失败原因进入结果 JSON。

建议 V5 默认 SLO：

| 类别 | TTFT | TPOT | 额外约束 |
|---|---:|---:|---|
| Short | e2e TTFT ≤ 1000ms | ≤ 100ms | 正常 EOS 或达到输出上限，且无错误 |
| Long | e2e TTFT ≤ 10s | ≤ 150ms | admission wait ≤ 30s，且无错误 |

这组阈值是 V5 的服务目标，不应根据跑出的结果临时调整。原始分位数仍需完整保留，避免单阈值
掩盖延迟分布。

## 5. 公平对照配置

正式 DP4 与 H1-4GPU 除拓扑和 H1 路由参数外，统一：

```text
model                   = Qwen3.5-4B
weights                 = BF16
kv_quant                = int8
block_size              = 256
cache_tokens            = 65536 per worker
prefill_chunk_size      = 16384
max_step_tokens         = 8192
concurrency             = 16
prefix_cache            = off
arrival                 = frozen trace replay
sampling                = greedy
warmup                   = 8 synthetic requests（不消费正式 trace）
conditional_pd_tokens   = 6144（仅 H1）
hybrid_long_overflow_ms = 5000（仅 H1）
pd_schedule              = load-aware（仅 H1）
```

要求：

- 不得为 DP 和 H1 分别选择有利的 chunk、cache、并发度或采样参数；
- Prefix Cache 首轮关闭，避免当前混合注意力 Prefix Cache 不完整污染主结论；
- 同一 seed 的 DP/H1 必须使用相同 trace hash；
- 正式运行必须基于 clean commit，并保存模型 manifest、CLI、CUDA/Triton、GPU 和拓扑；
- 任一请求 OOM、超时或输出不完整，该次运行不能进入性能汇总。

## 6. 命令模板

以下模板假设 V5 trace 已生成。`MODEL`、`DATASETS`、`TRACE`、`SEED`、`OUT`、`WORKERS`
必须显式替换。

### 6.1 双卡 DP2 门禁

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic \
  --trace TRACE --dp-devices 0 1 --concurrency 16 --warmup 8 \
  --kv-quant int8 --cache-tokens 65536 --block-size 256 \
  --prefix-cache-blocks 0 --prefill-chunk-size 16384 --max-step-tokens 8192 \
  --worker-log-dir WORKERS --output OUT --seed SEED
```

### 6.2 双卡 H1 门禁

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic --adaptive \
  --trace TRACE --prefill-devices 0 --decode-devices 1 \
  --conditional-pd-tokens 6144 --hybrid-long-overflow-ms 5000 \
  --prefill-short-policy work-conserving --pd-schedule load-aware \
  --concurrency 16 --warmup 8 --kv-quant int8 --cache-tokens 65536 \
  --block-size 256 --prefix-cache-blocks 0 --prefill-chunk-size 16384 \
  --max-step-tokens 8192 --pd-transfer-backend shm-ring \
  --worker-log-dir WORKERS --output OUT --seed SEED
```

四卡正式命令只把 DP 改为 `--dp-devices 0 1 2 3`，H1 改为
`--prefill-devices 0 --decode-devices 1 2 3`，其余参数不得变化。

## 7. 必报指标

每个 seed 分别报告，三 seed 再给中位数与范围：

| 指标 | 必报口径 |
|---|---|
| Short TTFT | e2e p50/p95/p99 |
| Short TPOT | p50/p95/p99 |
| Long TTFT | e2e p50/p95/p99 |
| Long TPOT | p50/p95/p99 |
| SLO goodput | Short 与 Long 分开，req/s 和 tok/s |
| 排队 | admission wait p50/p99/max |
| 吞吐 | 完成输出 Token/s，另报请求/s |
| 正确性 | 成功/失败/EOS/达到上限数量 |
| 调度 | route count、fallback count、每 worker 请求和 Token 分布 |

总体 TTFT 可以附录报告，但不能作为 headline；64个请求中Short占75%，总体中位数会天然偏向Short。

## 8. 判定规则

### 8.1 M1 中可以称 H1 有收益

必须同时满足：

1. 三个 seed 全部请求成功；
2. 至少两个 seed 的 Short SLO goodput 高于 DP，三 seed 中位数相对提升至少 10%；
3. Short e2e TTFT p50 和 TPOT p50 至少两个 seed 改善；
4. 总 Token 吞吐不低于 DP 的 90%；
5. Long admission wait 不超过 30 秒，且不存在 starvation；
6. 必须同时公开 Long TTFT 和 p99，即使其恶化。

若只改善 Short TTFT，但 Long 大幅排队或吞吐低于 90%，结论只能写成“明确的 Short 优先权衡”，
不能写成“H1 全面超过 DP”。

### 8.2 B1 边界验收

B1 不要求 H1 胜过 DP，只要求：

- 所有请求成功；
- 两条 Long 同时到达时无死锁、状态错装或首 Token 不一致；
- overflow/fallback 原因可观测；
- 无 Long 超过 30 秒 admission wait；
- 结果明确显示高 Long 压力下谁更优，而不是隐藏负结果。

## 9. V5 可支持的最终表述

若 M1 通过，只能写：

> 在真实内容、持续混合到达的客服 RAG 负载下，动态 H1 通过隔离 Long Prefill 改善 Short
> SLO goodput；Long-heavy 对照表明该收益以 Long 排队和吞吐边界为条件。

在没有重新运行真实文档摘要和真实代码分析之前，不再写“三类真实场景均提升75%～106%”。旧
V4 数字保留并明确标注为“随机 Token、固定输出的三类合成业务画像结果”。
