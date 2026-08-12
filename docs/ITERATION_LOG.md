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
- [ ] 4B + 9B 完整测试（后台运行中）
- [ ] 精度验证（PPL + 生成质量）
- [ ] MPS 模式实测
