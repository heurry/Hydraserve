# H1 调度问题分析:为什么 H1-4GPU 落后 DP4(V5 M1)

> 背景:V5 M1 真实混合 RAG(48 short 1-4K + 16 long 8-16K)。DP4 每 seed 64/64 成功、
> short SLO goodput 中位 48 tok/s;H1-4GPU(1P+3D)同样 64/64 成功但 short goodput
> 只有 17.6 tok/s,long TTFT 4x 于 DP。功能正常,纯调度/结构问题。

## 1. H1 准入/路由的负载信号(实测核对结论)

`MultiWorkerGenerationBackend`(`multi_worker.py`)的路由**没有任何 token 加权负载账本**:

| 信号 | 类型 | 位置 |
|---|---|---|
| `_prefill_available()` | 二元:prefill worker 是否健康 | multi_worker.py:1865 |
| `_prefill_long_inflight[i]` | 0/1:是否正在 prefill 一个 long | :476 |
| `_prefill_pending[i]` | 计数:in-flight 操作数 | :476 |
| `decode_load` | `max(1-kv_free/kv_total, 1-state_free/state_total)` 占用率 | serving_loop.py:101 |
| `_prefill_capacities` | 剩余 KV 块/state 槽(仅可行性过滤) | :456 |
| round-robin | 轮转兜底 | :1691 |

`len(request.token_ids)` 在 multi_worker.py 只用于阈值(short_cutoff / conditional_pd_tokens)
和路由元数据,**不参与负载比较**。

## 2. DP 的调度(collocated_multi.py:363,对照基准)

```python
def score(i):
    return (
        self._prefill_tokens[i] > 0,     # ① 有 outstanding prefill 排后
        self._prefill_tokens[i],         # ② outstanding prefill TOKEN 数最少 ←核心
        self._assigned_work[i],          # ③ token 加权(prompt+output)live work
        self._pending[i],                # ④ in-flight RPC
        capacity.decode_load,            # ⑤ decode 占用
        self._assigned[i], round_robin,
    )
```
- `_request_work = len(prompt) + max_new_tokens`(:320):每请求折算成 token 工作量。
- `_prefill_tokens[i]` 在 prefill 完成时精确增减(`_mark_prefill_complete`:418)。
- 16 个 long 的 prefill 按 token 摊到 4 张对称卡 → 无单卡瓶颈、负载均衡。

## 3. H1 失衡的两处根因

### 3a. short 抢 hybrid prefill 槽 —— 只比 decode_load(`_pick_serve_prefill_worker`:1635)

`admit()`(:549)对每个 short 先尝试塞 hybrid(:584-626),判定:
```python
if prefill_short_policy=="work-conserving" and short<cutoff and _prefill_available():
    index = _pick_serve_prefill_worker(...)   # 只看 decode_load
    ...
    route_reason = "prefill_worker_collocated"
```
`_pick_serve_prefill_worker` 在 `hybrid_load > competing_decode_load` 时才拒绝收 short。
`hybrid_load/decode_load` 都是 KV/state **占用率快照**,不反映:
- **token 成本**(2K short 与 16K long 同权重);
- **排队的 long**(下一时刻要 prefill 的 long 没进任何信号);
- **hybrid 的角色不对称**(decode 是兼职,主业 long prefill)。

结果:decode worker 一忙,short 就涌向此刻"占用率低"的 hybrid → 占掉唯一 prefill 槽 →
后续 long 排队。V5 配置 `conditional-pd-tokens 6144` 走确定性路由,**连 cost router 的
token 感知成本估计都没用上**(`router.decide` 只在 conditional_pd_tokens=0 时触发)。

### 3b. 单 prefill 引擎串行化全部 long(`_hybrid_prefill_slot_available`:1577)

```python
return any(healthy and role is HybridRole.DECODE ...)   # 一次只能接一个 long
```
1P+3D 下 16 个 8-16K long 的 prefill 全部串行在 GPU0。M1 里这是硬天花板,任何
short 调度都救不了 —— 这也是 DP 不存在的问题(每卡自带 prefill)。

### 3c. decode worker 上 short 与 long decode 混排

`conditional_short_collocated` 的 short 由 decode worker collocated prefill+decode,
但同一 worker 还要给 PD long decode。s42 里 `decode:1` 上 6 个 short p50 达 13.3s。

## 4. s42 路由分布佐证

| 路由 | 数量 | 说明 |
|---|---|---|
| `prefill_worker_collocated` | 18 | 占 prefill 槽的 short,p50 TTFT 5.8s |
| `conditional_short_collocated` | 30 | decode worker 上的 short,0.7~13.3s 不等 |
| `conditional_long_pd` | 13 | PD long |
| `hybrid_queue_overflow` | 3 | 回退 collocated |

## 5. 建议(不是"有 long 等就不收 short"的补丁)

1. 给 H1 补 DP 式 **token 加权账本**:按 worker 记 outstanding prefill token + token 加权
   assigned work;short→hybrid 的判定换成 DP 的 `score()` 式多级比较,并计入排队 long;
2. 正视 1P 的结构瓶颈:M1(25% 8-16K long)对单 prefill 卡就是超载,应对比 2P+2D
   (`--prefill-devices 0 1 --decode-devices 2 3`)或把该结论写入 H1 的边界(计划 §8.2 B1 意图);
3. 若 H1 目标是 short SLO 优先,可把 `prefill-short-policy` 设为 `never`,让 hybrid 专职 long,
   重新对比 —— 但那牺牲了 worker 利用率,是否值需要单独验证。

## 复现材料

- 最小死锁复现(已修复后通过):`traces/v5/tiny_seed42.jsonl`,命令见 `scripts/run_h1_4g_v5.sh`。
- 完整结果:`results/v5_dp4/`、`results/v5_h1_4g/`、`results/v5_h1_2g/`。
