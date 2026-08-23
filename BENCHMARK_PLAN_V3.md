# HydraServe 四卡压测方案 V3：混合注意力 PD 优势与适用边界

> 目标：在同一套 4×RTX 3090、同一模型、同一输入 trace 和同一精度口径下，证明
> HydraServe 的优势来自 **混合状态 PD、prefill/decode 干扰隔离、低开销 SHM 数据面、
> 长上下文 decode kernel 和分层前缀缓存**，同时公开 PD 在纯 decode/balanced 负载下的边界。
>
> 历史基线：`BENCHMARK_PLAN_V2.txt`、`BENCHMARK_RESULTS_2026-08-17.md`，基线 commit
> `a8f747a`。V3 首先复跑 V2 的核心点，再测试当前代码新增能力。

---

## 一、为什么需要 V3

V2 已证明，在 8×32K 长请求 + 32×2K 短请求的 burst 负载下：

| 拓扑 | 吞吐 | 聚合 TPOT P99 | 短请求 TPOT P99 | 短请求 TTFT P99 |
|---|---:|---:|---:|---:|
| 4×DP | 64.0 tok/s | 309 ms | 303 ms | 14.2 s |
| 2P+2D | 44.0 tok/s | 169 ms | 156 ms | 32.0 s |
| 1P+3D | 28.3 tok/s | 111 ms | 110 ms | 62.5 s |

它证明了 PD 能隔离 decode，但也暴露出三个口径问题：

1. 40 个请求同时 burst，PD 的 P 队列被瞬间压满，TPOT 改善和 TTFT 排队恶化同时发生；
2. `max_new_tokens` 只是上限，EOS 使各组实际输出量不同（负载 A 为 4248～4995 token），
   原始吞吐不完全是等工作量比较；
3. 聚合分位数混合了长短请求，没有直接回答“短交互请求在长 prefill 干扰下能否守住 SLO”。

V3 因而以 **短请求 SLO goodput** 为主指标，以总吞吐为约束；burst 只保留为过载/恢复测试，
不再作为唯一主结论。

---

## 二、待证明的四条结论

### H1：PD 隔离优势（主结论）

在长 prefill 持续注入的混合流量下，2P+2D 相比 4×DP：

- 短请求 TPOT P99 至少降低 30%；
- 短请求 SLO goodput 更高；
- 总 output throughput 保持 4×DP 的至少 60%；
- 不通过失败、提前 EOS 或少生成 token 换取结果。

### H2：PD 数据面的新增优化确实降低 TTFT

在相同 2P+2D 拓扑下，persistent SHM Ring、chunked transfer、计算/传输 overlap 和可选
INT8 wire transfer 应逐步降低长请求 TTFT，并减少短请求在 P 队列中的等待。

### H3：长上下文 decode kernel 与拓扑收益可叠加

FlashDecoding/split-K 分别在 4×DP 和 2P+2D 下改善长上下文 TPOT；kernel 优化不能被误写成
PD 拓扑收益，二者必须独立 A/B。

### H4：框架优势有明确边界

纯短请求或 decode-bound 负载下，4×DP 应保持吞吐优势；PD 只在 prefill 干扰和严格流式
延迟 SLO 下体现价值。报告必须保留这些负结果。

---

## 三、P0：压测前必须补齐的口径能力

以下能力没有完成前，不产出新的对外性能数字。

当前代码已完成 trace adapter、逐请求 `ignore_eos`、分 class/SLO 指标、进程内 4×DP backend、
持久请求负载均衡及跨 worker 并行 decode 的前置实现；**本次按要求未执行测试，仍须通过 3.5 的
四卡门禁，不能把“已实现”写成“已验证”。**

### P0.1 Trace workload

为 benchmark 增加 JSONL trace adapter，每条请求包含：

```json
{"id":"short-0","class":"short","prompt_tokens":2048,
 "max_new_tokens":128,"arrival_offset_ms":1200,"ignore_eos":true,"seed":42}
```

必须支持每请求独立的：

- `class`（short/long/balanced/real）；
- prompt token 目标；
- `max_new_tokens`；
- arrival offset；
- `ignore_eos`；
- seed。

性能 synthetic 负载统一 `ignore_eos=true`，保证各拓扑执行完全相同的 decode token 数；
真实质量负载保留 EOS。

### P0.2 HTTP 与进程内输入完全一致

修复 `run_http_benchmark` 未应用 `max_prompt_tokens` 的差异。生成 trace 后保存：

- tokenizer revision；
- 每条实际 re-encode token 数；
- prompt token ID 的 SHA256；
- 每条完整 JSON record 的 SHA256；
- 全 trace SHA256。

4×DP、PD 和 vLLM 必须读取同一个冻结 trace；不允许各自重新随机生成。

### P0.3 分 class 指标与 SLO goodput

JSON 除总体指标外，增加 `by_class.short/long/...`：

- request/output-token throughput；
- TTFT、TPOT、latency P50/P95/P99；
- queue/admission/prefill/transfer/decode 分解（能取得的部分）；
- 成功数、超时数、OOM、preemption/replay；
- SLO 达标请求数与 goodput。

计时口径固定如下，禁止再用 terminal event 代替最后一个 token：

- `ttft_ms`：客户端线程实际开始提交到首 token，只反映 engine path；
- `client_queue_ms`：trace 计划到达时刻到客户端线程真正开始的排队；
- `e2e_ttft_ms`：trace 计划到达时刻到首 token，正式 SLO 使用此项；
- `tpot_ms`：`(last_token_at - first_token_at) / (completion_tokens - 1)`；
- `release_tail_ms`：最后 token 到 terminal event，单独暴露 KV release/RPC/清理尾部；
- `itl_ms`：逐 token 间隔分布；engine-only 额外保存对应的 `decode_batch_sizes`。多 GPU 后端记录
  请求实际所在 P/D worker 的物理 batch width，不得用 coordinator 全局 batch 冒充单卡 batch；
- `per_worker` 的 key 使用 `prefill:N`、`decode:N` 或 `collocated:N`，避免 P0 和 D0 都叫
  `worker_id=0` 时合并统计。

`route_counts`继续表示成功请求以兼容历史结果，同时必须报告`route_counts_all`和
`route_failure_counts`。任一请求失败时`throughput_valid=false`，该轮output tok/s不得进入主表。

主 SLO 固定为：

- short TTFT ≤ 5 s；
- short TPOT ≤ 200 ms（至少 5 token/s 的流式速度）；
- 请求生成完整 128 token；
- 无错误。

同时报告相对指标：`mixed / short-only` 的 TTFT/TPOT inflation，避免结论依赖任意绝对阈值。

### P0.4 可复现性元数据

每个结果写入：git commit、dirty 状态、完整 CLI/env、CUDA/PyTorch/Triton/FlashAttention 版本、
GPU 型号/时钟/温度/topology、P2P 状态、模型与 trace hash。每轮保留 worker stderr 和 proxy stats。

### P0.5 本次前置修改后的四卡门禁

正式矩阵前必须依次通过，任何一项失败都先修复，不允许继续批量跑数：

1. `ignore_eos=true` 的请求即使首 token 为 EOS，也必须生成完整目标长度，结果中
   `finish_reason=length`、`completion_tokens=target_new_tokens`；
2. 用 8 条同长度 burst 请求检查 4×DP，`per_worker` 必须为每卡 2 条，worker log 中四个进程均有
   reserve/prefill/decode；这是持久 active-request 均衡的最小正确性门禁；
3. 用至少两卡同时 decode，结合 worker 时间戳或 Nsight Systems 确认不同 GPU 的 decode 区间重叠；
   不能只看总吞吐推断并行；
4. 构造交错 request 顺序（例如 worker 0/1/0/1），确认返回 token 与 request ID 一一对应，且无
   `PartialDecodeError`、错位或漏 token；
5. trace 生成后立即保存同名 `.meta.json`，重放前校验 `token_ids_sha256`、`record_sha256` 和
   `trace_sha256`；JSONL 一旦生成禁止手改，任何改动都必须重新生成；
6. `--trace` 重放统一使用 `--arrival-pattern burst` 且不再传 `--request-rate`，因为到达时刻已经
   固化在 `arrival_offset_ms`；否则会叠加第二套到达延迟；
7. `--warmup N` 只使用独立的 synthetic warmup，不消耗正式 trace；结果的 `requests` 必须仍等于
   trace 行数（W1 为 72）；
8. 检查每卡显存余量、KV free blocks、state slots、ring slots 和进程存活。W1 从
   `--cache-tokens 131072` 起步，不使用默认 65536；若 memory planner 下调，记录实际值；
9. 同一 seed 的 D0/P0/P1 必须读取同一个 trace 文件，结果 metadata 中 trace SHA256 必须完全一致；
10. 用带30ms同步`release()`的假backend验证`release_tail_ms`增长而TPOT不变；32-token与
    128-token跨负载解释必须同时报告ITL、物理decode batch size和release tail；同时提交第二条
    请求，确认第一条的release RPC未阻塞第二条decode；
11. 用阻塞long-PD和可立即完成short-D的并发测试验证两者使用独立executor；1P+3D下short不能
    因单个long占用P worker而等待同一个host prefill线程；
12. open-loop正式结果要求`client_queue_ms p99`接近0。若使用线程客户端，concurrency必须覆盖
    最大在途请求；否则改用异步提交，不能把线程池排队隐藏在TTFT之外；
13. 当前机器没有可用 NVIDIA 驱动且本次未执行四卡性能测试，因此以下命令只是四卡机的开跑模板，
    不是已验证结果。第一次四卡gate通过后再批量跑5 seeds。
14. P/D prefill executor 的 outstanding future 数不得超过对应物理 worker 数；排队等待 P 槽的
    long 不能提前 reserve D 侧 KV。用“阻塞第一个long + 排队第二个long + 可运行short”门禁确认
    第二个long尚未admit而short已经完成；
15. 重复调用同一 P-short 的 `admit()`只能发出一次reserve RPC；并发 P dispatch 必须通过原子
    claim分散到空闲P worker，不能多个线程同时观察旧pending值后扎堆同一卡；
16. `scripts/v3_gate_check.py` 的P0 hash门禁必须实际启动2P+2D，不能把四卡列表截成两卡后误跑
    1P+1D；`--datasets`必须作用于子命令而非继续读取脚本默认路径。
17. `shm-ring`门禁必须覆盖两个独立P进程并发向同一个D namespace发送，而不只是1P+1D smoke；
    连续复用slot时每个request digest/payload必须一一对应，无transfer timeout，结束后所有请求均
    可释放。2P+2D出现长时间四卡0%利用率且无结果JSON时按数据面死锁处理，禁止记作性能结果。

---

## 四、硬件、模型和统一配置

### 4.1 主环境

- 4×RTX 3090 24GB，无 P2P；
- CUDA 12.8、同一 Python 环境；
- 主模型 Qwen3.5-4B BF16；
- KV cache INT8；
- FlashAttention 和 split-K Paged Attention 默认开启；
- CUDA Graph 使用当前默认策略（shape 观察 16 次后捕获）；
- 每个正式配置先完成独立 warmup，不把 Graph/Inductor/FlashAttention 冷编译计入稳态；
- 冷启动 TTFT 另列，不与稳态混合。

### 4.2 主实验固定项

```text
block_size              = 256（FlashAttention paged KV 的真实物理 page 要求）
kv_quant                = int8
prefix_cache            = off（主干扰实验）
host_prefix_cache       = off（主干扰实验）
arrival                 = trace replay
seeds                    = 42, 43, 44, 45, 46
repetitions              = 5
```

`prefill_chunk_size` 先在 calibration seed=41 上扫描`{4096,8192,16384,32768}`，按短请求
SLO与总吞吐的帕累托点选定一次，随后冻结，不能在正式对比中按拓扑分别调参。
`max_step_tokens`固定为8192：同步单卡路径仍使用统一token budget；异步多worker路径只按一个
chunk（且不超过该量子）计admission cost，并由P/D物理executor槽限制并发，不再用完整32K
prompt和active decode数共同决定long能否提交。

### 4.3 W1 trace 与首轮开跑模板

先冻结 seed=42 的 W1。`SHORT_RATE` 替换为 W0 calibration 得到的目标 offered load；长请求到达
单位是毫秒。生成动作只加载 tokenizer，不启动 GPU runtime。

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic \
  --trace-out traces/w1_128_seed42.jsonl \
  --num-long 8 --long-tokens 32768 --long-new-tokens 16 \
  --long-arrival-offsets-ms 5000,20000,35000,50000,65000,80000,95000,110000 \
  --num-short 64 --short-tokens 2048 --short-new-tokens 128 \
  --short-trace-request-rate SHORT_RATE --seed 42
```

32-token 版本使用完全相同的参数，仅替换输出文件和 short 输出长度：

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic \
  --trace-out traces/w1_32_seed42.jsonl \
  --num-long 8 --long-tokens 32768 --long-new-tokens 16 \
  --long-arrival-offsets-ms 5000,20000,35000,50000,65000,80000,95000,110000 \
  --num-short 64 --short-tokens 2048 --short-new-tokens 32 \
  --short-trace-request-rate SHORT_RATE --seed 42
```

生成后记录打印出的 `trace_sha256`，确认 JSONL 中 `prompt_tokens` 是实际重编码长度；
`requested_prompt_tokens` 只是目标值。跨引擎公平性以实际 `prompt_tokens` 和
`token_ids_sha256` 为准，不以请求目标值冒充实际长度。

D0 首轮使用进程内 engine-only 4×DP，避免 HTTP/proxy 成为 baseline 的固定开销：

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic \
  --trace traces/w1_128_seed42.jsonl --dp-devices 0 1 2 3 \
  --concurrency 72 --warmup 8 --arrival-pattern burst \
  --kv-quant int8 --prefix-cache-blocks 0 --cache-tokens 131072 --block-size 256 \
  --prefill-chunk-size FROZEN_CHUNK --max-step-tokens 8192 \
  --worker-log-dir results/v3/w1_d0_seed42_workers \
  --output results/v3/w1_d0_seed42.json --seed 42
```

P0 首轮使用同一进程内 benchmark coordinator 和 2P+2D worker，不经过 HTTP：

```bash
python -m hydraserve benchmark MODEL DATASETS --dataset synthetic \
  --trace traces/w1_128_seed42.jsonl --adaptive --force-pd-tokens 1 \
  --prefill-short-policy never \
  --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware \
  --concurrency 72 --warmup 8 --arrival-pattern burst \
  --kv-quant int8 --prefix-cache-blocks 0 --cache-tokens 131072 --block-size 256 \
  --pd-transfer-backend shm-ring --pd-transfer-target-mb 8 \
  --pd-transfer-inflight 2 --shm-ring-slots 3 --shm-ring-slot-mb 64 \
  --prefill-chunk-size FROZEN_CHUNK --max-step-tokens 8192 \
  --worker-log-dir results/v3/w1_p0_seed42_workers \
  --output results/v3/w1_p0_seed42.json --seed 42
```

`MODEL`、`DATASETS`、`SHORT_RATE`、`FROZEN_CHUNK`必须显式替换，禁止原样运行。
P1 仅在 P0 正确性通过后增加 `--pd-transfer-quant int8`，其他参数不变。

2026-08-23的首次四卡复核中，修复后的D0在8192与65536下均为72/72且性能收敛，但P0曾触发
双P并发认领同一SHM slot的协议死锁。该问题已增加MPSC原子认领和多进程回归；恢复矩阵前只用
上述P0命令复跑seed42。验收条件为72/72、生成结果JSON、无transfer timeout、无持续四卡0%空转。
在该单点门禁通过前不要运行P1、多seed或把旧P0挂死记录纳入对比。

---

## 五、数据集与流量设计

### W0：短请求隔离基线

- 64 个 distinct 2K prompt；
- 每条严格输出 128 token；
- 无长请求、无前缀共享；
- Poisson 到达率扫描：先测 4×DP 饱和点 `lambda_max`，再取
  `0.3/0.6/0.8/0.95 × lambda_max`。

作用：得到没有长 prefill 干扰时的 TTFT/TPOT 和最大容量，作为 inflation 分母。

### W1：交互式 prefill 干扰（主负载，双输出长度）

- 8 个 distinct 32K 长 prompt，严格输出 16 token；
- 64 个 distinct 2K 短 prompt，分别冻结严格输出 **32 token** 与 **128 token** 的两份 trace；
- 短请求按 Poisson 到达；长请求在 trace 的 5/20/35/50/65/80/95/110 秒注入；
- 测 `0.6/0.8/0.95 × W0 lambda_max` 三档短请求负载；
- prefix cache 关闭。

设计原因：长请求负责制造 prefill 干扰，但只短暂参与 decode，避免 8×32K KV 长时间占据
decode batch，把“prefill 隔离”与“长 KV scan”混成一个瓶颈。

容量审计必须按 block 计算，而不是只看`cache_tokens`：block=256、cache=131072时每个D worker
只有512 blocks；32K+16输出的long需要129 blocks，2K+32/128输出的short均需要9 blocks。
因此单卡4条long已需516 blocks，理论上必定OOM；调度器必须限制未取得P/D prefill物理槽的请求
提前reserve KV，并在结果中保存实际并发KV占用。不得通过只统计成功token或让OOM提前缩短wall
time制造吞吐优势。

32-token 组用于放大 prefill-interference 与路由/迁移固定开销，128-token 组用于保留历史的
decode-heavy 边界。两组使用相同 prompt、到达时刻和 seed，仅 `max_new_tokens` 不同，均完整报告，
不允许只保留胜出的一组。

### W2：V2 可比 burst

原样保留 V2 负载 A：8×32K + 32×2K、每条严格输出 128 token、C16、burst、seed 42、
5 warmup、prefix cache off。用于与 commit `a8f747a` 的七组历史结果对齐；由于修复了 EOS 和
HTTP prompt cap，V3 数字必须标记为“口径修正版”，不能直接覆盖旧 JSON。

### W3：balanced 边界

- 80 个 distinct prompt，4K～8K 均匀分布；
- 每条严格输出 256 token；
- Poisson `0.6/0.8/0.95 ×` 短基线容量；
- prefix cache 关闭。

预期：PD 的 TPOT 可能更稳，但总吞吐不一定占优。

### W4：decode-bound 负载

- 64 个 distinct 2K prompt；
- 每条严格输出 512 token；
- 无长 prefill；
- Poisson 和 burst 各一组。

预期：4×DP 吞吐领先。该组用于确定 PD 的反例边界，禁止从最终报告删除。

### W5：共享前缀/会话负载

生成三份 token 数完全一致的 32K trace：0%、50%、80% block-aligned 公共前缀，divergent tail
保持 distinct。每份 32 请求、输出 32 token，分别测：

1. cache off；
2. GPU radix L1；
3. GPU L1 + Host block-radix L2；
4. L2 容量不足时的 eviction/reload。

记录 hit tokens、suffix bytes、Host/GPU cache occupancy、TTFT、传输字节和失败数。该组独立于
W1，不允许用共享前缀降低主拓扑实验的实际工作量。

### W6：真实数据外部有效性

- ShareGPT：至少 256 条，保留真实 prompt 长度与 EOS；
- LongBench：按 8K/16K/32K 分桶，每桶至少 30 条；
- GSM8K：用于生成质量，不作为长上下文性能主负载。

真实数据只验证趋势和质量，不替代 exact-token synthetic 的归因实验。

---

## 六、DP 与 PD 配置

### D0：4×DP baseline

- 四个独立 collocated worker，各占一张 GPU；
- 正式主 baseline 使用 `--dp-devices 0 1 2 3` 的 engine-only coordinator，不经过 HTTP；
- coordinator 以“已分配且尚未 release 的请求数”为持久负载，等负载时 round-robin；不能使用仅在
  RPC 调用期间非零的 pending 数，否则 burst 会集中到 GPU 0；
- 每轮审计 `per_worker`、worker log、prompt tokens 和长请求分布；
- decode 必须由 coordinator 同时派发到各 worker，不允许逐 worker 同步等待；
- 另跑一次严格 2-long-per-GPU 的平衡控制组，确认动态分配不是结论来源；
- 所有 worker 使用与 PD worker 相同的 KV 精度、chunk、Graph 和 kernel 开关。

HTTP proxy 只保留为线上服务路径或与只提供 HTTP API 的外部引擎对接，不作为 HydraServe
D0/P0/P1 主对比的数据面。若 vLLM/SGLang 只能通过 HTTP 压测，必须额外报告 transport-only
空载开销，并明确该组不是严格 engine-only 延迟对比。

### P0：2P+2D lossless（主 PD）

```text
prefill_devices         = 0,1
decode_devices          = 2,3
pd_schedule             = kv-aware
force_pd_tokens         = 1（主拓扑实验强制所有请求走 PD）
pd_transfer_backend     = shm-ring
pd_transfer_quant       = off（lossless 主结果）
pd_transfer_target_mb   = 8
pd_transfer_inflight    = 2
shm_ring_slots          = 3
shm_ring_slot_mb        = 64
worker_log_dir          = enabled
```

### P1：2P+2D INT8 wire（优化 PD）

与 P0 相同，仅开启 `--pd-transfer-quant int8`。只有在精度门禁通过后才进入主图；否则作为实验性
结果单列，不能替代 lossless 结果。

### P2：1P+3D（极致流式 SLO 点）

P 卡 0，D 卡 1/2/3，其余与 P0 相同。预期 TPOT 最低、TTFT 和总吞吐较差，用于画帕累托边界，
不作为默认推荐配置。固定角色消融必须增加：

```text
conditional_pd_tokens   = 0
force_pd_tokens         = 1
prefill_short_policy    = never
```

### P2-C：1P+3D conditional（long PD、short D-collocated）

```text
prefill_devices         = 0
decode_devices          = 1,2,3
conditional_pd_tokens   = 8192
force_pd_tokens         = 0
prefill_short_policy    = never
prefill_preempt_max_ops = 8
```

该组是确定性路由，不读取在线成本模型：2K short 必须留在 D，32K long 必须走 P→D。正式结果审计
`conditional_short_collocated` / `conditional_long_pd` route reason，任何其他 reason 都视为配置错误。

### P2-WC：1P+3D conditional + P work-conserving

与 P2-C 唯一差异为 `--prefill-short-policy work-conserving`。P 空闲时可接 short；long 到达后，
已绑定 short 的后续 decode 在 long prefill chunk 边界插入，每个边界最多执行 8 个短操作。
报告 `hydraserve_prefill_short_collocated_total` 和
`hydraserve_prefill_chunk_preemptions_total`，用于证明收益确实来自 P 复用/抢占。

### P3：2P+2D adaptive（生产策略）

使用冻结的 router profile，`force_pd_tokens=8192`：长请求强制 PD，短请求由成本模型决定；必须
报告 route reason、P/D queue、drift/fail-closed 次数。它与强制 PD 分开，避免把路由器收益和
PD 执行收益混为一谈。

---

## 七、正式实验矩阵

### M1：主拓扑/SLO 矩阵

| 组 | Workload | 配置 | 目的 |
|---|---|---|---|
| 1 | W0 | D0 | 4×DP 无干扰基线 |
| 2 | W0 | P0 | 2P+2D 固有开销 |
| 3 | W1 | D0 | DP 的 prefill 干扰 |
| 4 | W1 | P0 | lossless PD 隔离主结果 |
| 5 | W1 | P1 | INT8 wire 最优 PD |
| 6 | W1 | P2 | 1P+3D 极致 SLO |
| 7 | W1-32/W1-128 | P2-C | 确定性 long-PD / short-D |
| 8 | W1-32/W1-128 | P2-WC | P 卡 work-conserving + chunk 抢占 |
| 9 | W1 | P3 | adaptive 生产策略 |
| 10 | W3 | D0/P0 | balanced 边界 |
| 11 | W4 | D0/P0 | decode-bound 反例 |

W1 的三档到达率均运行 5 个 seed。主图横轴为总 output tok/s，纵轴为 short TPOT P99；第二张图
横轴为 offered request rate，纵轴为 short SLO goodput。

### M2：V2 七组复现

严格复跑 `BENCHMARK_PLAN_V2.txt` 的七组，但启用 P0 口径修复。输出同时给出：

- 历史 `a8f747a` 数字；
- 当前代码同配置数字；
- 当前代码关闭对应优化的同代码 A/B。

这一矩阵回答“本轮升级后是否整体进步”，M1 回答“框架优势在哪个生产场景成立”。

### M3：kernel 与 PD 数据面归因

#### K1：decode attention

W1、D0/P0 各跑：

- split-K 默认；
- `HYDRASERVE_PAGED_ATTENTION=reference`。

#### K2：prefill attention

32K/64K、C1 和 C4，分别：

- FlashAttention 默认；
- `--no-flash-attention`。

#### T1：PD transfer 阶梯

固定 1P+1D 和 2P+2D，各测 8K/32K/64K：

1. one-shot `shm` + chunked off；
2. persistent `shm-ring` + chunked off；
3. persistent ring + chunked on；
4. persistent ring + chunked on + INT8 wire。

只改变一个变量，记录 TTFT、output tok/s、传输字节、ring wait、in-flight 深度和 overlap 比例。

### M4：前缀缓存

按 W5 的 0/50/80% 三档运行 cache off、GPU L1、GPU+Host L2；报告冷/热两次结果，不能只报告
warm hit。Host radix 的收益必须以 suffix bytes 和实际 TTFT 同时证明。

### M5：开源对比

- vLLM：同一 Qwen3.5-4B checkpoint、4×DP、INT8/FP8 KV、prefix cache off、相同冻结 trace、
  `ignore_eos`、同 offered load；
- SGLang：仅在同 checkpoint 和同精度能稳定运行时加入；
- 若开源 PD 后端在无 P2P 的 3090 拓扑上不能运行，明确写“不具备相同硬件前提”，不填推测值；
- 比较 raw throughput、TTFT/TPOT、SLO goodput 和失败率，不只挑 HydraServe 有利指标。

HydraServe 的主张应表述为：相对成熟 4×DP 引擎，绝对吞吐可能仍有差距；其差异化价值是混合
状态 PD、消费级无 P2P 数据面，以及 prefill 干扰下的流式 SLO 隔离。

---

## 八、精度与正确性门禁

### 8.1 Kernel

- BF16/FP32 reference 对齐：Paged Attention、split-K、FlashAttention paged KV、GDN recurrent/
  conv、fused projection；
- 覆盖 2K/8K/32K/64K、batch 1/4/8、不同 page table width；
- 报告 max/mean absolute error，不只写测试通过。

### 8.2 传输

- lossless Ring 必须 token/logits/state 对齐；
- INT8 wire 报告 KV reconstruction error、最终 logits error、固定生成 token agreement；
- GSM8K 至少 200 条，报告 exact-match 与 lossless 的绝对差值；LongBench 报告选定子集分数。

### 8.3 稳定性

最优 D0、P0、P1 各做 30 分钟、80% 饱和负载 soak：

- 无 OOM、死锁、worker restart、状态错位；
- 显存无持续增长；
- Ring slot/in-flight 最终归零；
- 请求成功率 100%，preemption replay 输出一致。

---

## 九、统计与报告规则

1. calibration seed 与正式 seed 分离；参数只能在 calibration 上选择一次；
2. 每个正式点 5 个 seed，实验顺序随机化，组间清理进程并确认显存释放；
3. 报告 5 次 run-level 中位数、范围和 bootstrap 95% CI；
4. P99 必须同时报告样本数，长请求只有 8 条时以 max/P95 为主，不夸大 P99；
5. 总吞吐只在相同完整输出 token 数且`throughput_valid=true`时比较；不同32/128-token负载不得
   直接横比output tok/s，任何EOS、失败、超时都使该轮headline吞吐无效；
6. 冷启动、稳态、cache cold、cache warm 分开；
7. 不跨机器比较绝对值；vLLM/SGLang 与 HydraServe 必须同机交错运行；
8. 原始 JSON、trace、环境 manifest、worker log 和绘图脚本全部归档。
9. 主SLO使用`e2e_ttft_ms`和不含release的`tpot_ms`；`ttft_ms`、`client_queue_ms`、
   `release_tail_ms`作为归因项同时报告。

---

## 十、最终验收与可写结论

### 10.1 主验收

在 W1 的至少两个 offered-load 档位：

- P0/P1 的 short TPOT P99 比 D0 低 ≥30%；
- short SLO goodput 高于 D0；
- P0 总吞吐 ≥D0 的 60%；
- P1 相比 P0 TTFT 或吞吐获得稳定收益，且通过精度门禁；
- short TTFT P99 不因 P 队列失控超过 5 s；若失败，必须先修调度，不能靠只报 TPOT 掩盖。

### 10.2 边界验收

- W4 中明确给出 4×DP 的吞吐优势；
- W3 中说明 PD 是否值得；
- W2/V2 burst 中同时报告 PD 的 TPOT 改善和 TTFT 排队代价。

### 10.3 简历/项目描述允许写法

只有满足上述门禁后，才可写：

> 在 4×RTX 3090 无 P2P 环境、同一混合 trace 与完整输出工作量下，2P+2D 通过混合状态 PD
> 隔离长 prefill 对短请求 decode 的干扰，使短请求 TPOT P99 降低 X%，SLO goodput 提升 Y%，
> 同时保持 Z% 的 4×DP 吞吐；persistent SHM Ring、chunked overlap 和 INT8 wire 分别贡献……

如果未满足 TTFT/SLO goodput 门禁，只能写“降低 TPOT，但增加 TTFT/牺牲吞吐”，不能称为完整
帕累托改进。

---

## 十一、执行顺序与时间预算

1. **本地 P0 harness（0.5～1 天）**：trace、ignore-EOS、HTTP token 对齐、by-class/SLO 指标；
2. **四卡门禁（0.5 h）**：拓扑、worker health、正确性、显存；
3. **calibration（1～2 h）**：chunk/token budget 与 W0 饱和点；
4. **M1 主矩阵（4～6 h）**：三个负载档、5 seed；
5. **M2 V2 复现（2～3 h）**；
6. **M3/M4 归因（3～5 h）**；
7. **开源对比与多模型选点（3～5 h）**；
8. **soak（并行计时约 1.5 h）**；
9. **汇总（0.5 天）**：帕累托图、SLO 曲线、误差表、负结果与简历数字。

完整矩阵约 14～22 小时四卡机时；若时间有限，最小闭环为 P0 + W0/W1 的 D0/P0/P1 +
V2 组 1/2 + vLLM W1，共约 6～8 小时。
