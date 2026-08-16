# 64K 四拓扑帕累托实验 — 2026-08-16

4×RTX 3090, Qwen3.5-4B BF16 + INT8 KV, SHM FULL 传输, FlashAttention 开。
数据来源:`pareto2/`(工作树 `fix-prefix-remap`)。

## 0. 本轮代码改动(相对 949ec2e)

| 改动 | 文件 | 效果 |
|---|---|---|
| 前缀同前缀重映射不再抛错 | `cache/prefix_cache.py` | burst 下重复缓存共享前缀不再 `ValueError`(问题 1) |
| prefill 执行器并行度 = nP | `engine/serving_loop.py` + `multi_worker.py` | 原来 `max_workers=1` 串行,nP 从不并发(问题 2 根因) |
| 长请求强制走 PD | router profile `force_pd_tokens=4096` | 默认成本先验是 PARTIAL 时代的过时值,长请求全被判 collocated |

## 1. 实验设计(公平性修正后)

- **序列**:72 条(8 长 64K + 64 短 2K)随机打乱(seed 42),长短自然分布
- **DP** = 4 卡 round-robin 分片 × concurrency 4 = **16 并发**
- **2P+2D / 3P+1D** = concurrency 16,同一打乱序列
- 统一:`--kv-quant int8 --prefix-cache-blocks 4000 --prefill-chunk-size 65536 --arrival-pattern burst --max-new-tokens 128`
- burst = 72 条 t=0 全提交 + 滑动窗口(concurrency 个槽位滚动补位),非严格批次

## 2. 结果

| 拓扑 | 吞吐 tok/s | wall | TPOT p99 | 结果 |
|---|---|---|---|---|
| **4×DP** | **50.1**(聚合) | 183.8s(max) | ~457ms | 每卡 18/18 |
| **2P+2D** | 31.1 | 296.6s | 447ms | 72/72 |
| **3P+1D** | — | — | — | **72/72 全部 OOM**(GPU 3) |

4×DP 分片(长/短分布 2/1/3/2 长):gpu0 15.0 / gpu1 20.7 / gpu2 12.5 / gpu3 16.4 tok/s。

## 3. 长/短 TPOT 分解(关键)

| | 长请求 TPOT p99 | 短请求 TPOT p99 |
|---|---|---|
| 2P+2D(8 长 ÷ 2 D = 每 D ~4 长) | 436ms | 445ms |
| DP gpu0(2 长) | 223ms | 298ms |
| DP gpu2(3 长) | 454ms | 454ms |

**结论:TPOT 由 decode batch 里的长请求数量主导。** 每个 D 卡 decode batch 挤 N 个 64K
长请求时,每步 attention 扫 N×64K token,batch 内每个 token(不分长短)都被拖到 ~450ms。
64K 短请求被长请求"同 batch 干扰"是 TPOT p99 高的直接机制(与本地单卡 40ms→662ms 一致)。

## 4. 结论

1. **吞吐**:4×DP(50)> 2P+2D(31),因为 64K+128 输出是 decode 受限(9216 输出 token),DP 有 4 卡 decode、2P+2D 只有 2 卡(P 卡 prefill 完闲置)。
2. **3P+1D 撞内存墙**:单 D 卡装不下 8×64K KV,印证 OPT-05;2P+2D 是 64K 内存可行组合。
3. **TPOT**:PD 隔离没带来 p99 收益(447 vs 457),因为 decode batch 长请求干扰是主因,不是 prefill 干扰。

## 5. 下一步(W4/W5/W6)

- **W4(P 卡服务短请求 + prefill-load 路由)**:让 2P+2D 的 P 卡在 decode 阶段也接短请求,
  decode 并行度 2→4 卡 → 吞吐逼近 4×DP;**短请求 TPOT ~450→~40ms**。但 p99 仍由 8 长主导,不会明显降。
- **W5**(prefill 队列优先级):兜底被迫上 P 卡的短请求体验。
- **W6**(continuation chunk KV-tile 复用):长请求 decode 每步加速,是**真正降 p99** 的杠杆。
