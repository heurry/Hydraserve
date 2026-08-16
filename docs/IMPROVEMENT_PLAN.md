# 统一池(nP+mD)改进方案 — 2026-08-16

目标与判据见 [docs/TEST_PLAN_unified_pool.md](TEST_PLAN_unified_pool.md):
**2P+2D+INT8 在 distinct 混合负载下达到吞吐 ≥18 tok/s 且 TPOT p99 <400ms**
(4×DP 是 21.4/822ms,1P+3D 是 7.0/296ms,目标点两者都到不了)。

## 1. 当前状态(基于 de0f0ee 代码分析)

| 项 | 状态 |
|---|---|
| nP 基础:prefill_devices、n 个 P 进程、round-robin、per-worker 恢复 | ✅ 已实现(阶段 1) |
| prefix cache 递归遍历修复 + prefix_cache_stats 上报 | ✅ 已实现 |
| Bug A:第 2 个 prefill worker 启动崩溃(2P+2D 退化 1P+2D) | ⚠️ 待修 |
| Bug B:prefix cache hit_rate=0(publish 疑似截断) | ⚠️ 待修 |
| P 卡服务短请求 + prefill-load 感知路由 | ❌ 阶段 2 未做 |
| B2 / B3 / extract_kv-BF16 等性能欠账 | ❌ 未做 |

## 2. 工作项(按依赖排序)

### W1 [P0] 修复 Bug A:第 2 个 prefill worker 启动崩溃

**诊断结论(代码核对后)**:

1. 启动握手**存在且会 surface 错误**:构造函数等待全部 worker 的 `ready`,
   收到 `startup_error` 会 raise 并 `close(force=True)`——所以云端观察到的
   "2P+2D 退化 1P+2D"意味着崩溃发生在 **ready 之后**(或看到的是恢复循环反复
   失败);ready 之后的崩溃不经过任何消息通道,进程直接死;
2. mailbox 命名**核对无冲突**:`dst_gpu` 是逻辑 id(decode index + 1),收发两侧
   一致,key = request_id 全局唯一,两个 P worker 不会撞名;
3. 最可疑的 ready 后死因:**并发长请求下的显存峰值**(extract_kv 的 `.float()`
   把 KV 临时放大 2×,双 P 同时处理 128K 请求时单卡瞬时 ~8GB 额外分配)+ 云端
   历史残留进程 OOM。

**已修(本地,2026-08-16)**:

- `--worker-log-dir`(serve/benchmark):每个 worker 进程的 stderr 重定向到
  `{dir}/{prefill,decode}-{index}.log`,下次云端复现能拿到崩溃 traceback;
- 启动握手错误带 worker index 上下文("prefill worker N failed startup: ...");
- `dst_gpu` 逻辑 id 约定写入注释;PDWorkerConfig/PDClusterConfig 增加
  `worker_log_dir` 字段(2 进程模式与 nP 模式都接线);
- 相关测试 32 个通过(test_multi_worker / test_pd_service / test_pd_worker /
  test_transfer)。

**待云端复现**:干净 4 卡上跑 2P+2D,加 `--worker-log-dir ./worker_logs`,
若第 2 个 P 再崩,把 `prefill-1.log` 尾部贴回;重点验证 OOM 假设
(可先用 64K 数据集降低峰值)。

验收:云端 2P+2D `/health` 显示 2P+2D 全健康。

### W2 [P0] 修复 Bug B:prefix cache hit_rate=0 ✅ 已修复并验证

**根因有两个**(2026-08-16 定位):

1. 代码:频率门禁按"完整 prompt 序列"计频,distinct 尾部的请求共享前缀永远
   过不了 `minimum_frequency` 门禁;
2. 数据集:生成脚本把互异头部("Document {i}")放在**开头**,第 0 块即分叉,
   radix 树块级共享为零(云端与本地 v2 数据集同病)。

**修复**:①频率键改为共享前缀(树路径证明的 prior sighting 计入频次,无共享
路径时记录完整序列);②被拒 sighting 也挂树节点(第二次可见),`_prune_blockless_leaves`
按 `max_frequency_entries` 预算修剪无块叶子;③准入先试完整条目、再回退共享前缀
条目;④`_cached_blocks` 记账改为"挂 KV 时计数";⑤数据集 v3:共享前缀前置、
互异尾部后置。

**验证**:本地单卡 64K 标准负载(8 长共享 52220 token = 3263 块 + 64 短共享
133 块),`--prefix-cache-blocks 4000 --prefix-cache-min-frequency 2`:
**hit_rate = 0.972(70 hits / 2 misses),cached_blocks = 3270**,71/71 成功;
对照修复前 hit_rate = 0。相关测试 42 个通过(含 2 个新回归测试)。

### W3 [P0] 四拓扑帕累托实验(依赖 W1+W2)

- 数据集:**64K 上限**(4B 在 24GB 上跑 128K 太紧:单 chunk 激活 ~7GB + KV 4GB/请求,
  且 65-80s 的单请求 prefill 让实验 wall time 爆炸;64K 下 prefill ~27s、KV 2GB/请求
  BF16、1GB INT8,拓扑自由度大得多)。**8 长(64K)+ 64 短(2K),条数比 1:8**
  (token 占比 79%/21%,长请求压力足够暴露干扰,短请求量足够提供 decode 负载);
  内容 distinct + 80% 共享前缀;生成脚本复用 `/tmp/gen_mixed_local.py`(O(1) 编码);
- 统一配置:`--kv-quant int8`、`--prefix-cache-blocks` ≥ 共享前缀块数(64K 下共享
  前缀 ≈ 4000 块;注意 128K 时 80%×cache=7000 块小于 8192 块条目,entry-fraction
  会误拒,64K 后此约束消失)、warmup 1、burst、128 输出;
- KV 预算(64K,2GB/请求 BF16 / 1GB INT8):2P+2D 每 D 4 长 = 8GB+8.5GB 权重
  ≈ 16.5GB,**BF16 即够,INT8 更宽**;3P+1D 每 D 8 长 = 8GB+8.5GB ≈ 16.5GB
  (INT8),**首次内存可行**——64K 让 3P+1D 也成为可测点;
- 四组:4×DP(基线重测)、**2P+2D**、3P+1D(按 2026-08-16 决定,不再跑 1P+3D——
  其 128K 混合负载数据已作为参照点存在);
- 产出:(吞吐, TPOT p99) 散点 + 短请求 TTFT 分位数。

### W4 [P1] 阶段 2:P 卡服务短请求 + prefill-load 感知路由

- P worker 增加 collocated 执行路径(复用 decode worker 的 admission/state 语义);
- router 增加 `prefill_load` 输入(registry 已收集容量,需暴露 in-flight 长 prefill 数与预计排队);
- 路由规则:短请求避开"正在跑长 prefill"的卡;长请求保持 nP round-robin(或改最短队列)。

### W5 [P1] prefill 队列优先级(可与 W4 合并)

被 force 到 PD 的短请求在 prefill worker 内**永远优先于长 prompt**(当前是 FIFO)。
与 W4 互为补充:W4 让短请求绕开 P 卡,W5 兜底它们被迫上 P 卡时的体验。

### W6 [P2] B2:continuation chunk KV-tile 复用(优先级上调)

理由(本地数据支撑):chunk 粒度 = 短请求插队延迟。当前 131072 单 chunk 的 80s 大
kernel 阻塞短请求 decode(单卡混合实测 TPOT p50 662ms 的机制);chunk 4096 时短请求
最多等一个块,但 continuation 走 O(n²) 慢 22×。修好 B2,128K 才能用 4K-16K 分块
且吞吐损失 <20%——长上下文交互性是它的直接收益。

### W7 [P2] extract_kv 保留 BF16 传输(PD 逼近 DP)

log.md §5 已定位:`.float()` 把 BF16 KV 放大 FP32(1GB→2GB)再拷贝;改 uint16 原样
传输(QUANTIZED 路径仍走 FP32)。小改动,PD 32K TTFT 可再降 ~4.6s。

### W8 [P3] B3:GDN chunkwise 递推(研究性)

128K 单请求 prefill 65-80s 的主因(24 层 × 128K 步顺序递推);chunkwise 状态转移
矩阵化可并行。所有拓扑同时受益,但工作量和风险最大,放最后。

## 3. 里程碑

| 里程碑 | 内容 | 出口 |
|---|---|---|
| M1 | W1 + W2 | 2P+2D 可跑、prefix cache 生效 |
| M2 | W3 | 四拓扑数据 → 回答"统一池能否拿到帕累托点" |
| M3 | W4 + W5 | 短请求端到端体验(绕行 + 优先级) |
| M4 | W6-W8 | 长上下文性能深水区 |

## 4. 风险与注意

- W1 复现必须在干净 GPU 上(先查残留进程),worker stderr 要捕获到文件;
- INT8 KV 只有正确性验证、无精度数据——M2 实验报告必须标注;
- distinct 数据集生成用 O(1) 编码的构建方式(本地 `/tmp/gen_mixed_local.py` 已验证);
- 云端与本地环境版本差(triton 3.7.1 vs 3.0.0)影响绝对数字 ~20%,跨机对比只比趋势;
- W3 的 4×DP 基线要重测(开 prefix cache + INT8 后与原基线 21.4 不可直接比)。
