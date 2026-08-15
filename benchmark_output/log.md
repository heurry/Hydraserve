# HydraServe 环境搭建与长上下文压测记录

日期:2026-08-15。机器:4× RTX 3090 24GB,无 P2P(`nvidia-smi topo -p2p r` 全 `CNS`)。

## 1. 环境对齐(与真实跑通版本一致)

| 包 | 目标版本 | 实际 | 说明 |
|---|---|---|---|
| torch | 2.9.1+cu128 | 2.9.1+cu128 ✅ | 从 PyTorch cu128 源装,替换 AutoDL 预装的 2.1.2+cu121 |
| triton | 3.7.1 | 3.7.1 ✅ | 装 flash-attn 时会回退 3.5.1,需重装回 3.7.1 |
| tokenizers | 0.22.2 | 0.22.2 ✅ | 从 0.23.1 降级 |
| numpy | 2.3.5 | 2.2.6 ⚠️ | 2.3.5 需 Python≥3.11,本机 3.10.8,用 3.10 下最新 2.x |
| safetensors | 0.8.0 | 0.8.0 ✅ | |
| pytest | 9.1.1 | 9.1.1 ✅ | |
| flash-attn | (用户要求装) | 2.8.3.post1 ✅ | 源码编译 `FLASH_ATTN_CUDA_ARCHS=80`,sm_86 走 sm_80+PTX JIT |

其他:阿里云 pip 源被 IP 拉黑(403),全程用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

全量测试:218 passed / 23 skipped。

## 2. 关键发现:Triton 版本漂移

- 代码混用 `tl.range`(仅 2.1.0 有)与 `tl.rsqrt`/`tl.exp2`(≥2.2.0 才有)。
- `main.md` 写 `triton==2.1.0`、`pyproject.toml` 写 `triton>=2.1`,均为过时/错误。
- 实际需要 triton 3.5+(3.5.1 与 3.7.1 都验证有 `tl.range/rsqrt/exp2`),最终对齐 3.7.1。

## 3. B1 性能修复(logits 浪费)

**问题**:`runtime.py` 的 `forward()` 对每个 prefill chunk 都算 `[chunk, vocab]` 的 FP32 logits,vocab=248320,chunk=4096 时单份 ~4 GiB;prefill 循环只保留最后一份,中间 chunk 纯浪费。`chunk_size=32768` 时 logits 达 15.16 GiB → OOM。

**修复**(3 处 Edit,见 `git diff hydraserve/model/runtime.py`):
1. `forward()` 加 `compute_logits` / `last_position_only` 参数。
2. `compute_logits=False` 时跳过 lm_head;`last_position_only=True` 时只算 `hidden[:, -1:]`。
3. `prefill()` 循环:中间 chunk `compute_logits=False`,末 chunk `last_position_only=True`。

所有调用方都用 `logits[:, -1]`,向后兼容;`test_model_adapter` 期望 `(1,1,vocab)` 不破坏。

## 4. 实测数据

### 32K TTFT(DP collocated)

| 配置 | TTFT p50 | 说明 |
|---|---|---|
| 无 flash-attn, chunk 4096(LongBench 30K) | 124478 ms | 修复前基线(自适应路由全走 collocated) |
| 有 flash-attn, chunk 4096 | 268321 ms | B1 修复前,chunk 2+ 走 Triton O(n²) |
| 有 flash-attn, chunk 32768 | **11250 ms** | B1 修复后,整个 32K 走 FA ≈ **24× 提速** |

TPOT p50 稳定 ~77-79 ms(不受影响)。

### chunk 上限

- `chunk_size=65536`(64K 单块):`Triton Error [CUDA]: invalid argument`(kernel 参数超限,非 OOM)。
- 结论:**32768 是单 chunk 安全上限**,64K/128K 用 32768 分块。

### PD 实测

- PD 静态模式默认 `SharedMemoryTransferBackend.transfer_mode = PARTIAL_TRANSFER`(backend.py:129,硬编码)。
- PARTIAL 语义:只传 GDN 循环状态(53.48MB),decode 端**重算 full-attention KV**。
- 实测 PD @ 32K:TTFT p50 = **313686 ms**(313.7s),route 确认 `pd_disaggregated`;对照 DP @ 32K TTFT 11250 ms → **PD 慢 28×**。

### PD 慢的根因与修复

**根因**:`pd_service.py:240` 创建 decode worker 的 runtime 时**硬编码 `use_flash_attention=False`**(原注释:"prefill-only 的 FA 包不在 decode worker 上要求")。但 PARTIAL 模式下 decode 端重算就是一次完整 prefill,FA 被关掉后 full-attention 层掉进 O(n²) 的 Triton paged-attention,313s 就是这么来的。

**修复**(一行):`use_flash_attention=False` → `use_flash_attention=config.use_flash_attention`,并更新注释。

**关于"只重算 full-attention KV、不重跑 GDN 层"(用户方案 1)的可行性**:经读代码确认不可行——full-attention 层的 Q/K/V 来自前面所有层(含 GDN)流下来的 hidden,GDN 层的 `core` 输出(由 delta rule 算出)必须先算出来才能喂给 full-attention 层。重算必然要跑 GDN,只能靠 FA 加速。

## 5. FULL 传输改动(去掉重算)

**根因**:`SharedMemoryTransferBackend.transfer_mode` 硬编码 `PARTIAL_TRANSFER`,而重算=完整 O(n²) prefill,永远贵于 O(n) 的 KV 传输。

**改动**:`backend.py` 给 `SharedMemoryTransferBackend` 加 `mode` 参数(默认 `FULL_TRANSFER`),`transfer_mode` 返回 `self._mode`。`extract_kv`/`install_kv`/`pipeline.send` 的 FULL/QUANTIZED 链路本就完整,只需切换默认。同步修 `tests/test_transfer.py` 的 PARTIAL 专用测试(显式 `mode=PARTIAL_TRANSFER`)。

**PD 32K 实测链**:

| 配置 | TTFT |
|---|---|
| PARTIAL + FA 关(初始) | 313.7s |
| PARTIAL + FA 开 | 22.7s |
| **FULL 传输(不重算)** | **15.9s** |
| 对照 DP | 11.25s |

PD 从 28× DP 降到 **1.4× DP**。剩余 ~4.6s 主要是 `extract_kv` 里 `.float()` 把 BF16 KV 放大成 FP32(1GB→2GB)的 GPU↔CPU 拷贝,后续可优化(保留 BF16 传输)。

## 6. grid 修复(causal conv 的 gridDim.y 超限)

**根因**:`gdn.py` 的 `_causal_conv_kernel` 用 3D grid `(batch, sequence, channels)`,把 `sequence` 放 gridDim.y,CUDA 的 gridDim.y 上限是 65535。sequence=65536 时超限 → `invalid argument`(不是 OOM,也不是 B2)。

**修复**:grid 改为 `(batch*sequence, channels)`,`program_id(0)` 拆出 `batch_id=idx//sequence, token=idx%sequence`(sequence 摊到 gridDim.x,上限 2^31)。

**效果**:64K 单 chunk(65536)从不可用 → **27.4s**(对照 chunk 32768 双 chunk 的 607.7s,22×)。128K 单 chunk(131072)最初 OOM,根因是 `--cache-tokens` 设成 262144 过度分配 8GB KV;改对 `--cache-tokens 140000` 后单 chunk 跑通 **65.6s**。至此 32K/64K/128K 全部单 chunk 走 FA,B2(continuation chunk)对这三个长度被绕开,TTFT 近似线性(11.25s / 27.4s / 65.6s),瓶颈转为 B3(GDN 顺序递推)。

## 7. 32K/64K/128K DP vs PD 扫描(concurrency=1)

| 上下文 | DP TTFT | PD TTFT | PD/DP |
|---|---|---|---|
| 32K | 12.3s | 17.0s | 1.38× |
| 64K | 27.8s | 36.2s | 1.30× |
| 128K | 65.4s | 74.8s | **1.14×** |

**结论**:PD/DP 单调下降、趋向 1 但 concurrency=1 下永远 >1。FULL 传输的 O(n) 开销(extract/transfer/install)始终存在,只是随 prefill 占比增大而相对变小。**PD 的优势不在此维度**,而在高并发下的 prefill/decode 隔离(需并发扫描才可见)。

**128K OOM 修复**:PD 128K 的 prefill worker 因 PyTorch 缓存分配器碎片化 OOM("3.97 GiB reserved but unallocated"),加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后跑通(74.8s)。

## 8. 待办 / 结论

- B1(logits 浪费)已修,32K prefill 提速 24×。
- B1.5(FA 开在 decode worker)已修。
- FULL 传输已改并验证,PD 32K 313.7s → 15.9s。
- grid 修复已做,64K 单 chunk 607.7s → 27.4s(22×);32K/64K/128K 全部单 chunk 走 FA。
- extract_kv 改 BF16(uint16 原样传输),PD 传输减半(mode-aware,QUANTIZED 仍走 FP32)。
- B2(continuation chunk KV-tile 复用)对 32K/64K/128K 已被单 chunk 绕开;>128K 才需要。
- B3(GDN 顺序递推并行化)未动,当前 TTFT 的主要瓶颈。
- **下一步:并发扫描(固定上下文 × concurrency 4/16/64),测 TPOT P99 / 吞吐 / decode 抖动,找 PD > DP 的临界区。**
- 可选优化:extract_kv 保留 BF16(去 FP32 放大),PD 可进一步逼近 DP。

## 9. INT8 KV 压缩(为 256K)

**实现**:每 token 每 head 对称 int8 量化,存储 int8 数据 + float32 scale。改动:`paged_kv.py`(存储/量化写/反量化读 + `_quantize_kv`/`_scatter_scales`)、`memory_planner.py`(int8 字节数)、`__main__.py`(`--kv-quant int8` flag + 4 处 PagedKVCache/plan 接线)、`pd_service.py`(PDWorkerConfig.kv_quant + 2 处接线)。`layer_cache`/`read` 在边界反量化返回 BF16,注意力内核零改动。

**效果与发现**:
- 8K INT8 实测正确(2.8s TTFT,16 token 正常)。
- KV 从 BF16 8.6GB → int8 4.4GB(256K),省 ~4GB。
- **但 256K 单 chunk + INT8 仍 OOM(峰值 ~25GB)——真正的瓶颈是中间激活(MLP 中间层 ~12GB),不是 KV**。
- 256K 只能 chunked(chunk 65536)+ INT8 装下(约 16GB),但 chunk 2+ 走慢 Triton(B2),预计 ~30 分钟。

**结论**:INT8 解决了 KV 这一半,但单 chunk 256K 还需压缩/分块激活(或接受 chunked 的 B2 慢速)。

## 10. 混合负载 4×DP vs 1P+3D(128K 长 + 2K 短)

| 指标 | 4×DP collocated | 1P+3D adaptive |
|---|---|---|
| 总吞吐 tok/s | **~21.6**(5.4×4) | 7.0 |
| TPOT P99 | ~800ms | **296ms** |
| TTFT50 | 短 2.5s / 长 65s | 81.6s |

**结论**:DP 吞吐赢 3×,PD 尾延迟赢 2.7×。根因——DP 把长 prefill 并行到 4 卡(每卡 2 个 ≈130s);PD 的 8 个长 prefill 全压在 1 张 prefill 卡串行(8×65s=520s),prefill 卡成吞吐瓶颈,但短请求 decode 隔离(TPOT99 296ms vs 800ms)。

**这印证了整体判断:4 卡规模下 DP(collocated)是大多数场景的默认正确选择,PD 只在有严格 decode 延迟 SLO 且能接受 ~3× 吞吐损失时才值得。PD 的真正价值在 100+ GPU 的大规模部署(README 定位)。**

## 11. 并发扫描(1卡 DP vs 1P+1D,8K 上下文)

| 并发 | DP TPOT P99 | PD TPOT P99 | DP TPOT P50 | PD TPOT P50 | DP tok/s | PD tok/s |
|---|---|---|---|---|---|---|
| 1 | 33.0ms | 32.5ms | 32.2 | 32.3 | 18.0 | 16.8 |
| 4 | **109.3ms** | **37.3ms** | 86.4 | 36.3 | 30.4 | **34.7** |
| 32 | **825.9ms** | **40.2ms** | 498.4 | 35.8 | 34.5 | 34.8 |

**结论(与 §10 一致,更细粒度)**:PD 的价值在 **decode 尾延迟的隔离**——高并发下 DP 的 prefill 阻塞 decode,TPOT P99 从 33ms 飙到 826ms(32 并发 20× 恶化),而 PD 稳定在 ~37-40ms。吞吐几乎持平。**交叉点约在并发 4;低并发(1)时 PD 因传输开销略慢、无隔离收益。**
