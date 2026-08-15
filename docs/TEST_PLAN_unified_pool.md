# 统一池(nP+mD)压测设计

目标:证明"统一池"(每张卡同时具备 P / D / DP 能力,调度器逐请求分配)在 4×3090 上
相对 4×DP 的帕累托改进——即**吞吐接近 DP 量级的同时,decode 尾延迟接近 PD 量级**。

## 一、"比 4×DP 好"的判据(先定义清楚)

4 张卡上,统一池**不可能在吞吐上严格超过 4×DP**(4×DP 把 4 张卡全用于 prefill+decode,
最大并行度;统一池拆卡后 prefill 并行度天然减半)。所以正确目标是**帕累托改进**:

| 拓扑 | 吞吐 | TPOT p99 |
|---|---|---|
| 4×DP(基线) | 21.6 tok/s | 822ms ❌ |
| 1P+3D(现状) | 7.0 tok/s ❌ | 296ms |
| **统一池(目标)** | **≥18 tok/s** | **<400ms** |

**成功判据:`吞吐 ≥18 tok/s 且 TPOT p99 <400ms`** —— 同时摆脱 4×DP 的尾延迟塌方和
1P+3D 的吞吐塌方,拿到两个极端都到不了的帕累托点。

## 二、核心假设(本次要验证)

统一池能逼近 DP 吞吐的机理:4×DP 的 prefill(算力密集,~65s)与 decode(带宽密集)在
同一张卡上交替执行,短请求 decode 被长 prefill 阻塞(实测 TPOT p99 822ms)。统一池把
decode 隔离到 D 卡后,D 卡永不被 prefill 阻塞,有望回收这部分被干扰吃掉的吞吐。

**关键工作比(prefill:decode)决定 n/m 配比**:
- prefill 总量 = 8×65s = 520s;
- decode 总量 ≈ 8 长×128 token×261ms + 32 短×128×~40ms ≈ 430s;
- 两者量级接近 → **均衡切分(2P+2D)比偏 P(3P+1D)更合理**。

## 三、数据集(必须 distinct + 开 prefix cache)

1. **distinct 长请求**:8 个 128K prompt 内容真正不同(不是 12 句循环拼接)。注意:上批
   数据虽用 `Document {i}` 前缀让 token 序列从第 2 个 token 起就不同(prefix cache 只会
   匹配 ~1 个 token),但内容高度重复,对 decode 行为不真实 —— 所以改为真正 distinct。
2. **开 `--prefix-cache-blocks`**,容量 = **80% × (cache-tokens / block-size)**,模拟生产
   的前缀缓存占比;
3. **INT8 KV**(`--kv-quant int8`):让 D 卡 KV 预算放宽(2GB/请求 vs 4GB),2P+2D 每 D 卡
   装 4 长 = 8GB + 权重 ~8.5GB ≈ 16.5GB,余量充足。

## 四、第一约束:KV 容量(在时间约束之前)

`config.py:120-121`:`kv_bytes_per_token_bf16 = 8 层 × 4 KV heads × 256 dim × 4 字节
= 32 KB/token`。故 128K = 4GB/请求(BF16)/ 2GB/请求(INT8)。

| 拓扑 | 单 D 卡长请求数 | BF16 KV | INT8 KV |
|---|---|---|---|
| 3P+1D | 8 | 32GB ❌ | 16GB+8.5GB=24.5GB 紧 |
| 2P+2D | 4 | 16GB+8.5GB=24.5GB 紧 | 8GB+8.5GB=16.5GB ✅ |

**结论:2P+2D + INT8 是最内存可行的起点;3P+1D 需 KV eviction/offload,不值得先做。**

## 五、OPT-05 前置(不实现没法测 2P+2D)

1. **多 prefill worker 池**:长请求在 nP 间负载均衡,decode 从任意 P 收状态;
2. **P 卡可服务 collocated 短请求 + prefill-load 感知路由**:router 的 `decode_load`
   输入扩展为 prefill+decode 双向负载;
3. **D 卡 KV 预算**作为 nP/mD 配比与 admission 的第一约束(worker registry 已按容量
   评分,天然兼容)。

## 六、验证产出

同一份 distinct 数据集(开 prefix cache 80% + INT8 KV),四个拓扑各出一组
`(吞吐, TPOT p99)`,画二维散点。成功 = 2P+2D 落在"吞吐 ≥18 且 TPOT p99 <400"的
右上角空档(4×DP 和 1P+3D 都到不了)。

## 七、实施状态(2026-08-15)

- ✅ **阶段 1 代码已实现**:`PDClusterConfig.prefill_devices`(nP+mD)、n 个 prefill
  worker 进程、round-robin 负载均衡、per-worker 故障恢复。全量测试通过(2 个 flaky
  GPU OOM 测试除外,系残留进程占用显存所致)。
- ⚠️ **已知 bug(未修)**:spawn 第 2 个 prefill worker 后进程未存活(只起来 1 个
  prefill + 2 个 decode,GPU 1 空着)。诊断已确认 `prefill_devices` 配置与 argparse
  解析正确,但第二个 prefill worker 启动阶段崩溃,根因待定位(怀疑与两个 prefill
  worker 共享 decode SHM namespace 有关)。2P+2D 实测因此退化成 1P+2D。
- ⚠️ **prefix cache 去重未生效**:128K 长请求(80% 共享前缀)实测 `hit_rate=0`,
  `cached_blocks=567`(远小于共享前缀的 ~6554 块),`publish_prefix` 传参疑似被截断,
  独立 bug 待查。另外 prefix cache 的 `_count_evictable`/`_leaves`/`_nodes` 三个
  递归遍历在长 prompt 下会触发 `RecursionError`,已改为迭代(已修)。
- 📌 **4×DP 基线(开 prefix cache + 80% 共享前缀)**:20.9 tok/s、TPOT p99 ~810ms。
  (注意:此前无 prefix cache 的基线是 21.6 tok/s / 822ms,二者接近,因为 prefix
  cache 去重未生效。)

