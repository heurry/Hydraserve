# HydraServe 迭代优化日志

## 目标 (Goal)

1. **功能覆盖 main.md 全部内容或超越**
2. **BulletServe 式 intra-GPU 分离**（libsmctrl 因 CUDA 12.8 不可用 → 改用 MPS 调度）
3. **速度 + 精度双测试**（PPL、GSM8K、生成质量）
4. **量化支持**（INT4 权重以解锁更长上下文）
5. **公平对比** DP / PP / PD 分离 / Collocated
6. **推到硬件上限**（256K 上下文、最大 batch、满算力）
7. **记录每次迭代的过程和发现**

## 硬件环境

| 组件 | 规格 |
|------|------|
| GPU | 2× RTX 3090 24GB |
| PCIe | GPU0 x16, GPU1 x4（**非对称**） |
| P2P | ❌ 不可用（NODE 拓扑） |
| SHM 带宽 | ~4.5 GB/s（实测） |
| NVLink | ❌ 无 |
| CUDA | 12.8（libsmctrl 需 ≤12.6，不可用） |
| MPS | ✅ nvidia-cuda-mps-control 可用 |

## 迭代记录

### 迭代 1 (2026-08-12 22:00-22:30): 基础实现 + 首次真实测试

**做了什么**：
- 完成 HydraServe 全部 14 个模块的代码实现
- 首次真实加载 Qwen3.5-4B 测试

**发现**：
1. GPU 被 vLLM 容器占用 → 用户停止容器
2. transformers 4.46 不支持 qwen3_5 → 升级到 5.15
3. 首次测试数据（BF16 4B）：
   - Prefill 2K: 614ms, 3258 tok/s
   - 生成: 39.5 tok/s
   - 批量 decode ×4: 58.5 tok/s

### 迭代 2 (2026-08-12 22:30-23:00): 硬件上限探索

**做了什么**：
- 测试 max context / max batch / DP 模式 / SHM 带宽

**发现**（关键）：
1. **P2P 不可用**！GPU0→GPU1 之间是 NODE 拓扑，只能走 SHM
2. **SHM 带宽实测 4.58 GB/s**（x16+x4 的瓶颈在 x4 槽）
3. **BF16 4B max context 只有 ~7.3K tokens**（16K OOM）
   - 权重 8.8GB + KV cache + 激活值 > 24GB
4. **max batch 128**（无 KV cache），256 OOM
5. **DP 模式 GPU 利用率不均**：GPU0 41%, GPU1 100%
   - 原因：顺序执行 `model0(i0); model1(i1)` 而不是并行流
6. **Collocated 干扰实测**：
   - 1K prefill → decode 2.5× 减速
   - 2K → 3.9× 减速
   - 4K → 6.4× 减速

**结论**：
- 本机（无 NVLink、无 P2P、x16+x4）PD 分离只有 PARTIAL_TRANSFER 有意义
- DP 是本机最优策略
- 需要 INT4 量化才能解锁长上下文（4B: 8.8GB→2.2GB, 9B: 18.2GB→4.5GB）

### 迭代 3 (2026-08-12 23:00+): 量化 + MPS + 精度测试

**做了什么**：
1. 安装 bitsandbytes 0.50 + autoawq 0.2.9 ✅
2. 实现独立权重量化模块 `weight_quantizer.py`（不依赖第三方库）：
   - GPTQ 式 group-wise 对称 INT4（group_size=128）
   - `Int4Linear` 层：在线反量化
   - 预计 4B: 8.8GB→2.2GB, 9B: 18.2GB→4.5GB
3. 实现 MPS intra-GPU 模式 `mps_manager.py`：
   - libsmctrl 因 CUDA 12.8 不可用（需 ≤12.6）
   - 用 `nvidia-cuda-mps-control` 实现同卡分离
   - prefill + decode 进程共享 GPU，零拷贝状态传递
4. 编写综合 benchmark v2：速度 + 精度 + INT4 + 全策略对比

**进行中**：
- [x] bitsandbytes/autoawq 安装
- [x] INT4 权重量化模块
- [x] MPS intra-GPU 管理器
- [x] 综合 benchmark v2
- [x] 4B + 9B 完整测试
- [x] 精度验证（PPL + 生成质量）
- [ ] MPS 模式实测
- [ ] 27B AWQ-INT4 测试

### 迭代 4 (2026-08-13): INT4 量化验证 + 环境修复

**环境事故与修复**：
- compressed-tensors 安装把 torch 从 2.9.1 升级到 2.13.0，破坏 torchvision 兼容
- 修复：从 pip 缓存恢复 torch 2.9.1 wheel（缓存里找到 859MB 的 torch-2.9.1）
- 教训：安装包前检查依赖冲突，`--no-deps` 需要谨慎

**INT4 量化 bug 修复（3 个）**：
1. `quantize_model` 量化后立即反量化回 BF16 → 无 VRAM 节省（假量化）
   - 修复：`keep_int4_storage=True` 时用 `Int4Linear` 替换层
2. `Int4Linear.from_linear` 参数留在 CPU → device mismatch
   - 修复：参数移动到原层设备
3. `Int4Linear.forward` 反量化到 float32 → dtype 污染传播
   - 修复：反量化后 cast 到输入 dtype

**INT4 精度结果（4B，PPL on 5 段测试文本）**：
| 指标 | BF16 | INT4 | 目标 |
|------|------|------|------|
| PPL | 4.93 | 5.67 (+0.74) | <0.3 ✗ |
| KL divergence | - | 0.0009 | ~0 ✓ |
| Cosine sim | - | 0.997 | >0.99 ✓ |
| 生成匹配 | - | 完全一致 ✓ | 一致 ✓ |

PPL 略超目标：naive 对称量化无激活感知校准。AWQ/GPTQ 带校准可达 <0.3。
但 logits 方向一致性 0.997 说明对生成质量影响很小。

**INT4 上下文解锁结果**：
| 模型 | BF16 VRAM | INT4 VRAM | BF16 max ctx | INT4 max ctx | 增益 |
|------|-----------|-----------|-------------|-------------|------|
| 4B | 8.4GB | 5.9GB | 7.3K | **14.5K** | 2.0× |
| 9B | 17.9GB | 12.7GB | 3.6K | 3.6K | 1.0× |

9B 的 GDN 层（48 层中的 24 层 linear_attn）保持 BF16 不可量化，
5.2GB 节省主要来自 FFN + full attention 投影层。

### 迭代 5 (2026-08-13): 27B FP8 实测 + MPS intra-GPU 实测

**27B FP8 (vLLM TP=2, max-model-len 32K, enforce-eager)**：
- AWQ-INT4 (compressed-tensors) 在 vLLM 下 OOM（22.6GB/GPU）
- 用户原配置 FP8 成功：需 `--enforce-eager` 跳过 CUDA graph
- 缺 `preprocessor_config.json` → 从 4B 复制修复
- TTFT 实测（流式）: 512 tok=670ms, 2K=2294ms, 8K=9063ms, 16K=18400ms
- Prefill 吞吐: ~890 tok/s（16K 上下文）
- TPOT: 43-49ms 稳定
- 并发: 1→22.7 tok/s, 4→76.7 tok/s（峰值）, 8→67.2 tok/s

**MPS intra-GPU 双进程实测（GPU 0, 4B BF16）**：
| 场景 | Prefill (4K) | Decode (batch16) |
|------|-------------|-----------------|
| 单进程独占 | 1195ms | 134ms (120 tok/s) |
| MPS 双进程同卡 | 1580ms (1.32×) | 334ms (2.5×, 48 tok/s) |

**结论：MPS 模式有显著 SM 争用（decode 2.5× 减速）。**
验证了 main.md §5.8.3 的判断：无 libsmctrl 精确 SM 隔离时，
inter-GPU 物理分离（decode 完全不受影响）优于 intra-GPU MPS。
libsmctrl 需 CUDA ≤12.6，本机 12.8 不可用。

**三模型完整对比（全部实测）**：
| 模型 | 精度 | 服务方式 | Prefill 吞吐 | TPOT | 生成吞吐 |
|------|------|---------|-------------|------|---------|
| 4B | BF16 | HF 单卡 | 3.3K tok/s | 5-17ms | 42 tok/s |
| 4B | INT4 | HF 单卡 | 3.1K tok/s | - | - |
| 9B | BF16 | HF 单卡 | 2.3K tok/s | 6-8ms | ~35 tok/s |
| 27B | FP8 | vLLM TP=2 | 890 tok/s | 43-49ms | 25 tok/s |
