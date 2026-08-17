# HydraServe 压测结果（2026-08-17）

> 基线 commit `a8f747a`，4×RTX 3090 24GB，Qwen3.5-4B BF16 + INT8 KV。
> 数据文件见 `results/Load*.json`。方案与三项前置改动见
> [`BENCHMARK_PLAN_V2.txt`](./BENCHMARK_PLAN_V2.txt) / [`BENCHMARK_V2_CHANGES.txt`](./BENCHMARK_V2_CHANGES.txt)。

## 一、结果总表

| 组 | 拓扑 | 负载 | FD | 吞吐(tok/s) | TPOT p50 | TPOT p99 | TTFT p99 | 成功 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4×DP  | A | on  | 64.0  | 103 | 309 | 27.1s | 40/40 |
| 2 | 2P+2D | A | on  | 44.0  | 144 | 169 | 47.2s | 40/40 |
| 3 | 1P+3D | A | on  | 28.3  |  92 | 111 | 85.8s | 40/40 |
| 4 | 4×DP  | A | off | 56.7  | 150 | 360 | 27.6s | 40/40 |
| 5 | 2P+2D | A | off | 40.0  | 200 | 226 | 44.2s | 40/40 |
| 6 | 4×DP  | B | on  | 103.7 | 100 | 467 | 13.4s | 40/40 |
| 7 | 2P+2D | B | on  | 72.3  | 165 | 181 | 38.2s | 40/40 |

## 二、关键结论

1. **PD 隔离价值成立（组 1 vs 2）**：2P+2D 用 31% 吞吐换 45% TPOT P99 改善（309→169ms），帕累托改进成立。
2. **prefill 并行度（组 2 vs 3）**：1P+3D 是极致 SLO 点（吞吐 28.3、TPOT 111ms），两者均为帕累托点。
3. **FlashDecoding（组 1 vs 4、组 2 vs 5）**：关 FD 使 TPOT P99 分别 +16%（DP）/+34%（PD），与拓扑优化独立叠加。
4. **负载 B（组 6 vs 7）**：balanced 负载下 PD 仍显著改善 TPOT（467→181ms，-61%），吞吐损失 30%。

帕累托（负载 A，FD on）：

```
TPOT P99(ms)
310 |  · 4×DP (64.0, 309)
170 |        · 2P+2D (44.0, 169)
110 |               · 1P+3D (28.3, 111)
    |_________________________
     28    44    64   吞吐(tok/s)
```

## 三、本次会话发现并修复的问题

1. **多卡 PD 崩溃 `index_copy_(): index out of bounds`**：CUDA graph 捕获了每次 `batch()` 新建的动态 `slot_ids` 张量，replay 读到已释放内存。修复：`slot_ids` 纳入静态缓冲（`model/runtime.py`）。
2. **serve 并发 `decode state does not match its pool owner`**：`decode()` 失败回滚 `restore()` 用浅拷贝替换池化状态对象，破坏状态池身份校验。修复：池化状态跳过对象替换（`engine/serving_loop.py`）。
3. **serve int8 OOM**：`--kv-quant int8` 的 scale-scatter `.item()` host-sync 使 CUDA graph 捕获 100% 失败并泄漏数 GB。修复：int8 时跳过 graph + 失败捕获后 `graph.reset()`/`empty_cache()`（`model/runtime.py`）。
4. **4×DP 只有一卡在跑**：proxy 依赖远程 `/health` 选后端，burst 下惊群全打第一个后端。修复：本地 in-flight 计数（`scripts/dp_proxy.py`）。
5. **TPOT 测成 0**：serve SSE 是 close-delimited 响应，proxy `read(64KB)` 阻塞到连接关闭、批量缓冲全部 token。修复：逐行 `readline` 转发（`scripts/dp_proxy.py`）。
6. **router 全 collocated**：`--adaptive` 无 force-PD 开关，长 prompt 也判 collocated。修复：新增 `--force-pd-tokens` CLI（`__main__.py`）。
7. **状态池内存规划偏差**：`plan_paged_kv_blocks` 默认 `state_slots=1`，低估状态池占用。修复：传入实际 `max_state_slots`/`workspace` 容量（`engine/pd_service.py`、`__main__.py`）。

另有三项前置改动与三个补充项（synthetic dataset、dp_proxy、kv-aware 路由、per-worker 聚合、`--dp-proxy` HTTP 模式、decode FlashDecoding 开关确认），详见 `BENCHMARK_V2_CHANGES.txt`。
