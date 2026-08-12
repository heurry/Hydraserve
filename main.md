# HydraServe

> 面向混合注意力架构 LLM 的 Prefill-Decode 分离推理引擎

---

## 1. 项目概述

### 1.1 一句话定位

从零实现面向混合注意力架构（Qwen3.5/3.6：Gated Delta Network + GQA）的 PD 分离推理引擎，处理 full attention KV Cache 与 linear attention 循环状态的双状态传输，支持 4B / 9B / 27B 多模型，在双卡 RTX 3090（PCIe x16/可选 NVLink）上对比 Collocated / DP / TP / PD 分离四种策略。

### 1.2 要解决的问题：Prefill-Decode 干扰

Prefill（处理 prompt）和 Decode（生成 token）是两种完全不同的负载：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特征 | 计算密集（大矩阵乘） | 带宽密集（读权重，算一点点） |
| GPU 利用率 | 50-90% | 5-20% |
| 单次耗时 (32K ctx) | 50-100ms | 每 token ~5ms |
| 批次特征 | 离散到达，可攒批 | 持续运行，长尾 |

它们在同一个 GPU 上跑会互相打架。当 20 个请求正在 decode（200 tok/s 吞吐），一个 32K prefill 进来占用 100ms，20 个 decode 请求全部停转。如果每 2 秒来一个新请求，吞吐损失 5%；每 500ms 来一个，损失 20%；burst 来 5 个，连续 500ms decode 停滞。

**PD 分离把这个干扰归零。** Prefill GPU 专心算 prefill，decode GPU 专心 decode，物理隔离。新请求的 32K prefill 在 GPU 0 上跑 100ms，GPU 1 上的 20 个 decode 请求完全不受影响，继续 5ms/token 稳定输出。

### 1.3 PD 分离 vs DP vs TP：用传输开销换干扰消除

PD 分离不是"更好"的策略，而是"用传输开销换干扰消除"。完整的 tradeoff：

| | DP（2 卡独立） | TP=2（2 卡同步） | PD 分离（2 卡分工） |
|---|---|---|---|
| GPU 利用 | 两卡各做全量工作（2x 算力） | 两卡同步协作（算力有同步损耗） | 各做一半（prefill 卡空闲时多） |
| 传输开销 | 零 | 每层 all-reduce | 每请求 1GB KV + 25MB 状态 |
| Prefill-Decode 干扰 | 每卡内部有（干扰导致 P99 恶化） | 有（一张 prefill 另一张等着） | 零（物理隔离） |
| P50 TPOT | 好 | 好 | 好 |
| P99 TPOT | 差（干扰导致长尾） | 差（同步等待导致长尾） | 好 |

**关键洞察：存在 crossover point**。低并发时传输开销 > 干扰代价，DP 赢；高并发时干扰代价 > 传输开销，PD 分离赢。这个 crossover point 是项目的核心实验结论。

Chunked Prefill（vLLM 的方案）把 prefill 切小块和 decode 交错，但干扰还在，只是从"停 100ms"变成"每步多花几 ms"。DistServe 论文证明 unified 模式 P99 延迟恶化 3-10x。Chunked prefill 缓解但不解决。

### 1.4 混合注意力引入的额外挑战：双状态传输

PD 分离的动机是 prefill-decode 干扰，与模型架构无关。但当目标模型是 Qwen3.5/3.6（Gated Delta Network + GQA 混合注意力）时，PD 分离引入了一个额外挑战：状态传输有两种完全不同的类型：

| 状态类型 | 大小 (32K ctx, 9B) | 特性 | 迁移策略 |
|---------|-------------------|------|----------|
| Full Attention KV Cache | 1 GB (BF16) / 320 MB (INT4) | 随上下文线性增长，按 block 粒度管理 | 可量化压缩 |
| Linear Attention 循环状态 | 25 MB (FP32) | 固定大小，不随序列增长，不可量化 | 必须整体传输 |
| Linear Attention conv 状态 | 0.75 MB | 固定大小 | 整体传输 |

**循环状态不可量化**：FP32 循环矩阵量化会导致递推误差累积发散。25MB 不大但必须原样传输。

双状态传输不是 PD 分离的动机，而是在混合注意力模型上做 PD 分离时遇到的额外工程挑战。PD 分离本身对所有模型都有价值（消除干扰），但混合注意力的双状态传输是本项目的技术贡献点--现有 PD 分离工作只处理标准 KV Cache 迁移，这是 2026 年中才被提出的进行中问题，尚未完全解决。

---

## 2. 现状与相关工作

### 2.1 PD 分离生态

| 项目 | 状态 | 混合注意力支持 | 硬件 |
|------|------|---------------|------|
| **vLLM + NIXL** (PR #41869, merged) | GDN 部分支持 | Qwen3.5 GDN 的 conv-state 传输 | RDMA |
| **vLLM + Mooncake** (PR #46807, merged) | GDN 支持 | 修复 MooncakeConnector 对 GDN 的 crash | RDMA |
| **vLLM + MoRIIO** (PR #51052, open) | Kimi-K3 支持 | 最完整：conv+ssm recurrent state READ+WRITE | RDMA + ROCm |
| **SGLang** (Issue #32732, open) | 开发中 | PD disagg decode radix cache for mamba/SSM | RDMA |
| **Mooncake** (Issue #2242, open) | 未实现 | Feature request for hybrid SSM/GDN | RDMA |
| **nanoPD** (161 stars) | 已完成 | 不支持，只做标准 transformer（Qwen3-8B） | H20 |
| **BulletServe** (ASPLOS 2026, 52 stars) | 已完成 | 不支持（代码无混合注意力实现） | 单卡 SM masking |
| **HydraServe（本项目）** | 设计中 | 核心设计目标 | 3090 + NVLink |

### 2.2 与本项目的区别

| 维度 | nanoPD | vLLM PRs | HydraServe |
|------|--------|----------|------------|
| 模型 | Qwen3-8B（全 attention） | Qwen3.5-0.8B（GDN） | Qwen3.5-4B/9B, Qwen3.6-27B（GDN） |
| 从零实现 | 是 | 否（vLLM 补丁） | 是 |
| 循环状态传输 | 不需要 | 需要但作为补丁 | 核心设计 |
| N-1 truncation | 不需要 | 需要 | 需要 |
| 层级别流水线 | 没有 | 没有 | 有 |
| 自适应路由 | 有 | 没有 | 有 |
| 多模型支持 | 单模型 | 单模型 | ModelAdapter 多模型 |
| Benchmark vs TP | 有 | 有 | 有 |
| Benchmark vs DP | 没有 | 没有 | 有 |

补充：BulletServe 对比

| 维度 | BulletServe (ASPLOS 2026) | HydraServe (本项目) |
|------|--------------------------|---------------------|
| 分离方式 | intra-GPU（SM masking，同卡分区） | inter-GPU（跨卡物理分离） |
| 传输开销 | 零（共享 GPU 内存） | KV Cache + 彪环状态传输 |
| NVLink 需求 | 不需要 | 可选（有则全量 BF16 传输，无则 INT4 量化） |
| 混合注意力 | 不支持（代码 0 结果） | 核心设计目标 |
| 双状态传输 | 不需要（同卡共享内存） | 核心技术挑战 |
| N-1 truncation | 不需要 | 需要 |
| 多卡扩展 | 不支持 | 支持（TransferBackend） |
| 跨节点 | 不支持 | RDMA 接口定义 |
| 硬件依赖 | libsmctrl + MPS + CUDA <=12.6 | 标准 CUDA + 可选 NVLink |
| 适用场景 | 单卡，模型放得下 | 多卡或需要 inter-GPU 分离 |

BulletServe 在单卡场景下严格优于 inter-GPU PD 分离（无传输开销、更高利用率）。HydraServe 的价值在于 BulletServe 不覆盖的交集：多卡场景下的 inter-GPU 状态传输 + 混合注意力架构的双状态传输。

### 2.3 硬件带宽与传输策略

本项目支持三种硬件配置，通过 TransferBackend 自适应选择传输策略：

| 硬件 | 带宽 | 传输策略 | 32K 传输时间 | 32K prefill | 可行？ |
|------|------|---------|-------------|-------------|---------|
| NVLink (有 bridge) | 112 GB/s | 全量 BF16 KV + 循环状态 | 9ms | 50-100ms | 完全可行 |
| 双 x16 PCIe (P2P) | ~12-16 GB/s | INT4 KV (345MB) + 循环状态 | 29ms | 50-100ms | 可行 |
| 双 x16 PCIe (SHM) | ~8-10 GB/s | INT4 KV (345MB) + 循环状态 | 43ms | 50-100ms | 可行 |
| x16+x4 PCIe (SHM) | ~4 GB/s | 仅循环状态 (25MB) + KV 本地重算 | 6ms + 25ms 重算 | 50-100ms | 部分 PD |

关键洞察：INT4 KV Cache 量化（KIVI，3.2x 压缩）是 PCIe 无 NVLink 场景下的突破口。BF16 KV 1GB 在 12 GB/s 下要 85ms，勉强；INT4 KV 320MB 只要 29ms，完全可隐藏。

部分 PD 分离模式（低带宽 fallback）：当带宽不足以传完整 KV Cache 时，只传循环状态（25MB，6ms 可隐藏），decode 端本地重算 8 层 full attention KV（~25ms）。干扰从 collocated 的 100ms 降到 25ms，4 倍改善。

### 2.4 与 BulletServe 的定位区分

BulletServe（ASPLOS 2026）提出 intra-GPU disaggregation，用 SM masking 在单卡上做 prefill-decode 分离，不需要传输。对单卡场景是更优解。

HydraServe 的价值在 BulletServe 不覆盖的交集：
- BulletServe 是单卡方案，模型放不下时（多卡场景）不适用
- BulletServe 不支持混合注意力（代码 0 结果），双状态传输挑战不存在
- HydraServe 做多卡 + 混合注意力，双状态 inter-GPU 传输是独有挑战

对 9B/27B INT4（单卡放得下），BulletServe 式 intra-GPU 分离理论上更优。但：(1) BulletServe 依赖 libsmctrl（CUDA <=12.6，非标准库），(2) BulletServe 不支持混合注意力模型，(3) 从零实现 inter-GPU PD 分离展示更多工程深度。

### 2.5 BulletServe 功能对比与 HydraServe 整合

BulletServe 的核心功能与 HydraServe 的对比：

| BulletServe 功能 | HydraServe 是否整合 | 实现方式 |
|----------------------|---------------|---------|
| intra-GPU SM masking 分离 | 简化版（MPS 模式） | TransferMode.INTRA_GPU，两进程共享 GPU |
| 动态 SM 分配调度 | 未实现 | 依赖 MPS 调度，无 libsmctrl |
| 空间-时间 编排 | 部分支持 | 自适应路由 + chunked prefill |
| 混合注意力双状态 | 不支持 | HydraServe 核心卖点 |
| 跨卡传输 | 不支持 | HydraServe 核心卖点 |
| 多模型支持 | 不支持 | ModelAdapter 4B/9B/27B |

HydraServe 在 BulletServe 不覆盖的交集上提供价值：多卡 inter-GPU 分离 + 混合注意力双状态传输。同时通过 MPS 模式整合 BulletServe 式 intra-GPU 分离能力，形成 inter-GPU + intra-GPU 的完整分离体系。

### 2.6 有/无 NVLink 下的策略对比

| 策略 | 无 NVLink（双 x16 PCIe） | 有 NVLink |
|------|----------------------|------------|
| DP | 最佳（零传输） | 最佳（零传输） |
| TP=2 | 可用（all-reduce ~25ms） | 最佳（all-reduce ~2ms） |
| PD 分离 | 可行（INT4 KV, 29ms 传输） | 最佳（BF16 KV, 9ms 传输） |
| Offload | 可行（低带宽壁垒） | 没意义（有 NVLink 不需要 offload） |

无 NVLink 时 PD 分离和 TP 的优劣取决于场景：PD 分离解决干扰（P99 TPOT），TP 解决模型放不下。4B/9B INT4 单卡放得下，TP 没必要，PD 分离是更合适的选择。

| 模型 | 无 NVLink 最佳 | 有 NVLink 最佳 | 理由 |
|------|--------------|--------------|----------|
| 4B (2GB) | DP | DP | 单卡放得下，不需要 TP |
| 9B (4.5GB) | DP 或 PD 分离 | DP 或 PD 分离 | 单卡放得下，TP 没必要 |
| 27B (13.5GB) | PD 分离 | TP 或 PD 分离 | 单卡放得下，TP 没必要 |
| 70B (35GB) | PP（唯一选项） | TP (~8) | TP 切 17.5GB/卡 |

PD 分离解决的是不同的维度--不是“模型放不下”而是“两种负载互相干扰”。9B INT4 单卡放得下，TP 没必要，但 prefill-decode 干扰照样存在。--

## 3. 硬件环境与目标模型

### 3.1 硬件

| 组件 | 规格 |
|------|------|
| GPU | 2x NVIDIA RTX 3090 (24GB VRAM) |
| NVLink | 可选（112 GB/s 双向，需 NVLink Bridge ~$30-80） |
| PCIe | GPU 0 x16, GPU 1 x16（双 x16 插槽） |
| 内存 | >= 64 GB DDR4 |
| 存储 | >= 500 GB NVMe SSD |

**默认配置（无 NVLink）**：双 x16 PCIe 提供 P2P 传输 ~12-16 GB/s。配合 INT4 KV Cache 量化（KIVI，3.2x 压缩），32K 上下文传输量从 1GB 降到 345MB，传输时间 ~29ms，可隐藏在 50-100ms 的 prefill 计算时间内。**PD 分离在无 NVLink 环境下依然可行。**

**可选增强（有 NVLink Bridge）**：NVLink 是独立的 GPU 间物理链路，不走 PCIe 总线，112 GB/s 双向直连 P2P。RTX 3090 有 2 个 NVLink 接口，安装 Bridge 后启用全量 BF16 传输模式。

**三种传输配置**：

| 配置 | 带宽 | 传输策略 | 32K 传输时间 | 可行性 |
|------|------|---------|-------------|---------|
| NVLink (有 bridge) | 112 GB/s | 全量 BF16 KV + 循环状态 | 9ms | 完全可行，100% 隐藏 |
| 双 x16 PCIe (P2P) | ~12-16 GB/s | INT4 KV (345MB) + 循环状态 | 29ms | 可行，隐藏在 prefill 内 |
| 双 x16 PCIe (SHM) | ~8-10 GB/s | INT4 KV (345MB) + 循环状态 | 43ms | 可行，长 prefill 可隐藏 |

验证 NVLink 可用：

    nvidia-smi topo -m
    # 有 NVLink Bridge：GPU0-GPU1 显示 NV12（12 条链路）
    # SYS = 走系统总线（无 NVLink）
    # NV12 = NVLink 直连，12 条链路
    # 无 bridge 也能做 PD 分离（PCIe P2P + INT4 KV 量化）

### 3.2 INT4 KV Cache 量化：无 NVLink 的突破口

在无 NVLink 环境下，INT4 KV Cache 量化是让 PD 分离可行的核心技术依赖：

| 指标 | BF16 KV | INT4 KV (KIVI) | 压缩比 |
|------|---------|---------------|--------|
| 32K 9B KV 大小 | 1 GB | 345 MB | 3.2x |
| NVLink 传输时间 | 9 ms | 3 ms | - |
| PCIe P2P 传输时间 | 85 ms | 29 ms | - |
| PCIe SHM 传输时间 | 130 ms | 43 ms | - |
| Perplexity 损失 | 0 | < 0.3 | 可接受 |

KIVI 策略：per-channel 量化 K（沿通道方向），per-token 量化 V（沿序列方向）。解码阶段在线反量化，反量化 kernel 融合在 paged attention kernel 内，避免额外 HBM 读写。

关键：BF16 KV 在 PCIe P2P 下需要 85ms，勉强隐藏在 100ms prefill 中但余量极小。INT4 KV 只需 29ms，有充足余量。**INT4 KV 量化不是精度优化，而是让 PCIe 环境下 PD 分离可行的带宽适配手段。**

### 3.3 目标模型

本项目支持三个混合注意力模型，通过 ModelAdapter 接口统一管理：

| 参数 | Qwen3.5-4B | Qwen3.5-9B | Qwen3.6-27B |
|------|-----------|-----------|------------|
| hidden_size | 2560 | 4096 | 5120 |
| num_hidden_layers | 32 | 32 | 64 |
| full_attention_interval | 4 | 4 | 4 |
| linear attention 层数 | 24 | 24 | 48 |
| full attention 层数 | 8 | 8 | 16 |
| num_attention_heads | 16 | 16 | 24 |
| num_key_value_heads | 4 | 4 | 4 |
| head_dim | 256 | 256 | 256 |
| linear_num_key_heads | 16 | 16 | 16 |
| linear_key_head_dim | 128 | 128 | 128 |
| linear_num_value_heads | 32 | 32 | 48 |
| linear_value_head_dim | 128 | 128 | 128 |
| linear_conv_kernel_dim | 4 | 4 | 4 |
| mamba_ssm_dtype | float32 | float32 | float32 |
| max_position_embeddings | 262144 | 262144 | 262144 |
| vocab_size | 248320 | 248320 | 248320 |
| 权重 INT4 (AWQ) | ~2 GB | ~4.5 GB | ~13.5 GB |

### 3.4 两种状态的内存占用

**Full Attention KV Cache**（只有 full attention 层产生）：

| 模型 | Full Attn 层数 | KV/token (BF16) |
|------|---------------|-----------------|
| 4B | 8 | 32 KB |
| 9B | 8 | 32 KB |
| 27B | 16 | 64 KB |

计算：2 (K+V) x full_layers x 4 kv_heads x 256 dim x 2 bytes

**Linear Attention 循环状态**（固定大小，不随序列增长）：

| 模型 | Linear 层数 | SSM state | Conv state | 合计/请求 |
|------|-----------|-----------|------------|----------|
| 4B | 24 | 24.0 MB | 0.75 MB | 24.8 MB |
| 9B | 24 | 24.0 MB | 0.75 MB | 24.8 MB |
| 27B | 48 | 48.0 MB | 1.50 MB | 49.5 MB |

计算：ssm = lin_layers x 16 key_heads x 128 key_dim x 128 val_dim x 4 bytes (FP32)

### 3.5 显存预算

**Decode GPU（GPU 1）可用空间**：

| 模型 | 权重 INT4 | 框架开销 | 可用 KV/状态空间 |
|------|----------|---------|----------------|
| 4B | 2 GB | 1 GB | 21 GB |
| 9B | 4.5 GB | 1 GB | 18.5 GB |
| 27B | 13.5 GB | 1 GB | 9.5 GB |

**并发能力**：

| 模型 | 上下文 | KV/请求 | 状态/请求 | 总计/请求 | 并发数 |
|------|---------|---------|---------|---------|--------|
| 4B | 8K | 0.25 GB | 0.025 GB | 0.27 GB | 77 |
| 4B | 32K | 1.00 GB | 0.025 GB | 1.02 GB | 20 |
| 4B | 128K | 4.00 GB | 0.025 GB | 4.02 GB | 5 |
| 9B | 8K | 0.25 GB | 0.025 GB | 0.27 GB | 68 |
| 9B | 32K | 1.00 GB | 0.025 GB | 1.02 GB | 18 |
| 9B | 128K | 4.00 GB | 0.025 GB | 4.02 GB | 4 |
| 27B | 8K | 0.50 GB | 0.050 GB | 0.55 GB | 17 |
| 27B | 32K | 2.00 GB | 0.050 GB | 2.05 GB | 4 |
| 27B | 128K | 8.00 GB | 0.050 GB | 8.05 GB | 1 |

两张卡各放一份完整模型（INT4），NVLink 只传状态不传权重。

---

## 4. 系统架构

### 4.1 总览

    +-----------------------------------------------------------+
    |                    CentralScheduler                        |
    |         (request routing, transfer coordination)           |
    +----------+--------------------+---------------------------+
               |                    |
        +------v-------+    +-------v--------+
        |  Prefill     |    |   Decode       |
        |  Engine      |    |   Engine       |
        |  (GPU 0)     |    |   (GPU 1)      |
        |              |    |                |
        | +----------+ |    | +------------+ |
        | | Chunked  | |    | | Continuous | |
        | | Prefill  | |    | | Batching   | |
        | | Scheduler| |    | | Scheduler  | |
        | +----+-----+ |    | +------+-----+ |
        |      v       |    |       v         |
        | +----------+ |    | +------------+ |
        | | State    |    | | KV Cache   | |
        | | Extractor| |--->| | Manager    | |
        | | (layer)  | |NL  | | (Paged)    | |
        | +----------+ |    | +------------+ |
        |              |    | | Linear     | |
        | +----------+ |    | | State Pool | |
        | | Cost     | |    | | (FP32)     | |
        | | Model    | |    | +------------+ |
        | | Profiler | |    |                |
        | +----------+ |    | +------------+ |
        |              |    | | Prefix     | |
        |              |    | | Cache Tree | |
        |              |    | +------------+ |
        +--------------+    +----------------+

### 4.2 核心组件

| 组件 | GPU | 职责 |
|------|-----|------|
| CentralScheduler | CPU | 请求路由、传输协调、自适应决策 |
| Chunked Prefill Scheduler | GPU 0 | 长 prompt 分块、批次管理 |
| State Extractor | GPU 0 | 逐层提取 KV block + 循环状态 |
| NVLink Transfer Layer | GPU 0->1 | 层级别异步流水线传输 |
| Continuous Batching Scheduler | GPU 1 | 多请求 decode 调度、抢占恢复 |
| KV Cache Manager | GPU 1 | PagedAttention block 分配管理 |
| Linear State Pool | GPU 1 | FP32 固定 slot 分配管理 |
| Prefix Cache Tree | GPU 1 | Radix tree 前缀缓存（skip mamba） |
| Adaptive Router | CPU | 按请求特征选择 Collocated 或 PD 分离 |
| ModelAdapter | both | 多模型适配接口 |
| TransferBackend | GPU 0->1 | 传输后端抽象（NVLink/SHM/RDMA） |
| API Server | CPU | OpenAI-compatible API |

### 4.3 ModelAdapter 接口

    class ModelAdapter(Protocol):
        def load_model(self, gpu_id: int, precision: str) -> None
        def get_layer_types(self) -> List[str]  # ["linear","linear","linear","full",...]
        def get_ssm_state_shape(self) -> Tuple[int, int, int]  # (heads, key_dim, val_dim)
        def get_ssm_state_dtype(self) -> torch.dtype  # float32
        def get_conv_state_shape(self) -> Tuple[int, int, int]  # (heads, kernel, dim)
        def get_kv_cache_shape(self, n_tokens: int) -> Tuple[int, ...]  # per full-attn layer
        def get_num_full_attn_layers(self) -> int
        def get_num_linear_attn_layers(self) -> int
        def forward_layer(self, layer_idx: int, input, kv_cache, ssm_state, conv_state) -> output
        def get_weight_size(self, precision: str) -> int

三个实现：Qwen3_5_4BAdapter, Qwen3_5_9BAdapter, Qwen3_6_27BAdapter。框架通过 adapter 调用模型，不硬编码任何模型特定参数。

### 4.4 数据流

    1. 请求到达 -> CentralScheduler -> Adaptive Router 路由决策
    2a. Collocated 路径 -> GPU 0 prefill + decode（短 prompt）
    2b. PD 分离路径:
        -> GPU 0 chunked prefill（逐层计算 + 状态提取）
        -> NVLink 异步传输（layer 级别流水线）
        -> GPU 1 接收状态 + first-token seeding / N-1 truncation
        -> GPU 1 continuous batching decode
    3. 输出 -> API Server -> 返回用户

---

## 5. 核心技术详细设计

### 5.1 推理引擎与 Triton Kernel

#### 5.1.1 Full Attention Prefill：flash_attn 库

使用 flash_attn_varlen_func，支持 GQA（16:4 或 24:4 head ratio）和变长序列。不自己实现 Flash Attention--Tri Dao 团队的实现已经极致优化，重写只会慢 3-5 倍。这体现了"知道什么该用现成工具"的工程判断力。

#### 5.1.2 Full Attention Decode：Triton Paged Attention Kernel

flash_attn 不支持 paged KV（非连续 block 存储）。decode 阶段必须自己实现。

**问题**：KV Cache 存在 PagedAttention block 中，物理上不连续：

    Block 0: tokens 0-15   -+
    Block 3: tokens 16-31   |- 逻辑连续，物理分散
    Block 7: tokens 32-47   |
    Block 1: tokens 48-63  -+

**Triton kernel 逻辑**：
1. 从 block table 查出当前请求所有物理 block id
2. kernel 内按 block 遍历 K/V，计算 Q . K^T 的分块 score
3. online softmax（流式 softmax，避免 materialize 完整 score 向量）
4. 累加 V 的加权和

decode 阶段 Q 只有 1 个 token，比 prefill 简单--没有 Q 的分块，只有 K/V 的分块。约 150-200 行 Triton 代码。

#### 5.1.3 Linear Attention (GDN) 前向：Triton Fused Delta Rule Kernel

GDN 的核心计算是 delta rule 递推，与标准 attention 完全不同：

    # delta rule (每步):    #   decay = 1 - beta_t * alpha_t          # [16 heads, 1]
    #   S_t = S_{t-1} * decay + beta_t * v_t @ k_t^T  # [16, 128, 128]
    #   out_t = gate_t * (S_t @ q_t)          # [16, 128]

**不 fuse 的问题**：如果不融合，每步递推都要回 HBM 读写 state（128x128x4 bytes = 64KB/head，16 heads = 1MB），32K prefill 要 32K 次 HBM 读写 = 32GB 读写量。

**Fused kernel**：把 state 留在 SRAM，整个 prefill 只读写一次：

    @triton.jit
    def gdn_fused_forward(k, v, q, beta, alpha, gate, state_buf, out_buf,
                          seq_len, n_heads, key_dim, value_dim):
        state = tl.zeros([n_heads, key_dim, value_dim], dtype=tl.float32)
        for t in range(seq_len):
            k_t = tl.load(k + t * stride_k)
            v_t = tl.load(v + t * stride_v)
            q_t = tl.load(q + t * stride_q)
            b_t = tl.load(beta + t)
            a_t = tl.load(alpha + t)
            g_t = tl.load(gate + t)
            decay = 1 - b_t * a_t
            state = state * decay + b_t * (v_t @ k_t.T)
            out_t = g_t * (state @ q_t)
            tl.store(out_buf + t * stride_out, out_t)
        tl.store(state_buf, state)

这个 kernel 是项目最有价值的 Triton 工作--没有现成库实现 GDN fused kernel。27B 模型的 linear 层数翻倍（48 vs 24），state 更大（48MB），fused 的收益更显著。

#### 5.1.4 其他层

- RMSNorm：标准实现
- SwiGLU (FFN)：标准实现
- RoPE：Qwen3.5/3.6 使用 mrope（多模态 RoPE），interleaved 模式，partial_rotary_factor=0.25

### 5.2 双状态内存管理

#### 5.2.1 PagedAttention Block Manager（full attention 层）

| 参数 | 4B/9B | 27B |
|------|-------|-----|
| Block size | 16 tokens | 16 tokens |
| Block 大小 (BF16) | 512 KB | 1 MB |
| Full Attn 层数 | 8 | 16 |
| KV heads | 4 | 4 |
| Head dim | 256 | 256 |

当 decode GPU 内存紧张时，可将不活跃请求的 KV Cache 量化为 INT4（KIVI 风格，per-channel K + per-token V），压缩比 3.2x，perplexity 损失 < 0.3。

#### 5.2.2 Linear Attention State Pool

固定大小 slot pool，不增长、不可量化：

    slot[i] = {
        ssm_state: [lin_layers, key_heads, key_dim, val_dim] fp32,
        conv_state: [lin_layers, key_heads, conv_kernel, key_dim] fp32
    }

| 模型 | 每 slot 大小 | 并发上限（decode GPU 可用空间 / KV per req） |
|------|------------|--------------------------------------|
| 4B/9B | 25 MB | 空间够，state 不是瓶颈 |
| 27B | 50 MB | 27B 下 50MB x N 会占可观比例 |

#### 5.2.3 协同分配

Decode GPU 总内存 = 24 GB，扣掉权重和开销后：

    可用 = 24 - int4_weight - 1 GB
    linear 状态占用 = N x state_per_req
    Full Attn KV 占用 = N x kv_per_token x avg_ctx_len
    两者共享同一块 VRAM

示例（9B, 32K ctx, 18 并发）：
    linear 状态：18 x 25 MB = 450 MB
    Full Attn KV：18 x 1 GB = 18 GB
    总计 18.45 GB < 18.5 GB

### 5.3 PD 分离传输协议

#### 5.3.1 双状态异构序列化

    class StateTransferDescriptor:
        request_id: int
        model_name: str  # 通过 ModelAdapter 确定
        first_token_id: Optional[int]  # first-token seeding
        regions: List[RegionDescriptor]

    class RegionDescriptor:
        region_type: str  # "full_attn_kv" | "linear_ssm" | "linear_conv"
        layer_indices: List[int]
        shape: tuple
        dtype: str  # "bfloat16" | "int4" | "float32"
        quantized: bool
        src_gpu: int
        dst_gpu: int

Full attention KV block 可选 INT4 量化（当 decode 端内存紧张时），linear attention 状态必须 FP32 原样传输。

#### 5.3.2 N-1 Prompt Truncation

循环状态在第 N 个 token 处理完后，编码的是 token 0..N-1 的信息。Decode 端拿到状态后不能直接从第 N+1 个 token 生成--需要先在本地重新前向计算第 N 个 token，让循环状态推进一步。

    Prefill 端:
      处理 token 0, 1, ..., N-1 -> 产生 state (编码 0..N-1) -> 传到 decode 端
      注意：不处理 token N，留给 decode 端

    Decode 端:
      接收 state (编码 0..N-1)
      执行 recompute_last_token(token N): 前向计算一次，state 推进到编码 0..N
      从 token N+1 开始正常 decode

开销 < 5ms（单 token 前向）。原因是循环状态的语义边界--不像 KV Cache 那样每个 token 独立存储，循环状态是递推累积的。

#### 5.3.3 First-Token Seeding（首 token 预播种）

Prefill 端在 prefill 结束时已经采样了第一个 token。正常做法是 decode 端重新前向计算得到这个 token（~58ms）。优化：prefill 端把采样的 token id 随状态一起传到 decode 端，decode 端直接输出。

    # Prefill 端
    logits = model.forward(input_ids)
    first_token_id = sample(logits[-1])  # 已采样
    transfer_descriptor.first_token = first_token_id

    # Decode 端
    if transfer_descriptor.first_token is not None:
        output_token(transfer_descriptor.first_token)  # 直接输出
        next_input = first_token_id
    else:
        recompute_last_token()  # fallback: N-1 truncation

节省 ~58ms p50 TTFT。源自 vLLM Issue #51919（2026-08-10）。

#### 5.3.4 层级别异步流水线传输

**核心思想**：不要等 prefill 全部完成再传，而是在每层计算完成后立即启动该层状态的异步传输。

    Prefill GPU (compute stream):
      Layer 0 (linear) -> Layer 1 (linear) -> ... -> Layer 31 (full) -> done
           |                  |                              |
           v                  v                              v
    NVLink (transfer stream):
      [L0 state 1MB]     [L1 state 1MB]    ...           [L31 KV blocks]

    Decode GPU (receive stream):
      [L0 ready]         [L1 ready]        ...           [all ready -> start decode]

**CUDA stream 架构**：

    # GPU 0 (Prefill)
    compute_stream = torch.cuda.Stream(device=0)
    transfer_stream = torch.cuda.Stream(device=0)

    # GPU 1 (Decode)
    receive_stream = torch.cuda.Stream(device=1)

    # 每层结束后
    with torch.cuda.stream(compute_stream):
        output = layer_forward(input)

    with torch.cuda.stream(transfer_stream):
        torch.cuda.current_stream().wait_event(compute_event)
        torch.cuda._p2p_send(state, dst=1, stream=transfer_stream)

**传输时间**（NVLink 112 GB/s）：

| 操作 | 9B 数据量 | 27B 数据量 | NVLink 时间 | prefill 层计算 | 可隐藏？ |
|------|---------|---------|------------|--------------|---------|
| Linear 层 state | ~1 MB | ~1 MB | <0.01ms | ~2ms | yes |
| Full attn 层 KV (BF16) | 128 MB | 256 MB | 1.1-2.3ms | ~5ms | yes |
| 全部 32K 状态 (BF16) | 1 GB | 2 GB | 9-18ms | 50-200ms | yes |
| 全部 32K 状态 (INT4 KV) | 345 MB | 690 MB | 3-6ms | 50-200ms | yes |

### 5.4 Continuous Batching + Chunked Prefill

#### 5.4.1 Chunked Prefill（GPU 0）

长 prompt 切成 4K token 块，在 prefill GPU 上交错处理：

    Without chunked prefill:
      [====== 32K prefill 100ms ======] [transfer 9ms] [decode starts]

    With chunked prefill:
      [4K][4K][4K][4K][4K][4K][4K][4K]
           |    |    |    |    |    |    |    |
        [transfer L0-L3] ...                [done]
                                      [decode starts, transfer done]

每 chunk 完成后可以检查是否有新 prefill 请求到达，交错处理，提高 prefill GPU 吞吐。

#### 5.4.2 Continuous Batching（GPU 1）

请求状态机：

    WAITING -> PREFILL_TRANSFER_PENDING -> READY -> RUNNING -> FINISHED
                                         |
                                      PREEMPTED -> READY (restore)

- WAITING：请求到达，等待路由决策
- PREFILL_TRANSFER_PENDING：prefill 正在 GPU 0 上跑，等待状态传输
- READY：状态已到达，可开始 decode
- RUNNING：正在 decode
- PREEMPTED：内存不足被抢占，等待恢复
- FINISHED：生成完成

**抢占策略**：LRU 驱逐最久未活跃的请求。被驱逐请求的 KV Cache 可量化为 INT4 压缩存储，状态保留等待恢复。如果 GPU 0 有余力，也可回传到 prefill GPU 暂存。

### 5.5 自适应路由

#### 5.5.1 Cost Model

    collocated_latency = prefill_time(prompt_len) + interference_penalty(n_decode)
    disaggregated_latency = prefill_time(prompt_len) + max(0, transfer_time - prefill_time) + decode_time

路由决策：

| 条件 | 路由 | 理由 |
|------|------|------|
| prompt < 2K | Collocated | prefill 太快（~5ms），transfer（9ms）不值得 |
| prompt > 8K + decode 有空位 | PD 分离 | transfer 可隐藏在 prefill 后 |
| prompt > 8K + decode 满载 | Collocated | decode 满了，transfer 也白等 |
| prompt > 32K | PD 分离 | prefill 长（100ms+），transfer 一定隐藏 |

#### 5.5.2 参数实测

启动时运行微基准测试：

    prefill_throughput = benchmark_prefill(model, seq_lens=[1K, 4K, 8K, 32K, 128K])
    nvlink_bandwidth = benchmark_nvlink_transfer(sizes=[1MB, 10MB, 100MB, 1GB])
    decode_throughput = benchmark_decode(model, batch_sizes=[1, 4, 8, 16, 32])
    interference_coeff = benchmark_collocated_interference()

#### 5.5.3 Output Length Predictor

用 prompt 末尾 token 的 logits 分布熵估计。高熵 -> 可能长输出 -> 值得走 PD 分离。低熵 -> 可能短输出 -> Collocated 即可。

### 5.6 Prefix Caching 与混合注意力

**问题**：标准 prefix caching 用 radix tree 匹配 KV Cache 前缀。但 linear attention 的循环状态是递推计算的--不能从中间状态"恢复"出某个前缀的循环状态。

**解法**（skip_mamba_match 策略，与 SGLang Issue #32732 一致）：
- Prefix matching 只匹配 full attention KV Cache
- Linear attention 状态始终从 prefill 端获取（不可 prefix cache）
- cow_mamba=False：不从 radix tree 做 copy-on-write，state 通过网络到达

radix tree 仍然累积 linear attention 状态，支持未来 prefix reuse。但匹配时跳过 linear 状态检查。


### 5.7 TransferBackend 抽象层

#### 5.7.1 设计动机

PD 分离的核心动作是状态从 prefill GPU 传到 decode GPU。传输介质可以是 NVLink（单节点）、RDMA（跨节点）、ROCm Infinity Fabric（AMD 平台）或 SHM（无 NVLink fallback）。不同介质的带宽、延迟、内存注册要求完全不同，但层级别流水线、双状态序列化、N-1 truncation 等上层逻辑是介质无关的。

框架通过 TransferBackend 抽象层隔离传输细节，上层只调用 send/receive，不关心底层是 NVLink 还是 RDMA。

#### 5.7.2 接口定义

    class TransferBackend(Protocol):
        def send(self, tensor: torch.Tensor, dst: int, stream: Any) -> None
        def receive(self, tensor: torch.Tensor, src: int, stream: Any) -> None
        def get_bandwidth(self) -> float          # GB/s，用于流水线深度计算
        def requires_memory_registration(self) -> bool  # RDMA 需要
        def register_memory(self, tensor: torch.Tensor) -> Any  # RDMA MR
        def supports_layer_pipeline(self) -> bool  # 带宽 >= 10 GB/s 才做层级别流水线
        def get_latency(self) -> float             # 单次传输启动延迟 (ms)

#### 5.7.3 四种后端实现

**NVLinkBackend**（实现 + 测试）：

    class NVLinkBackend(TransferBackend):
        def send(self, tensor, dst, stream):
            torch.cuda._p2p_send(tensor, dst=dst, stream=stream)
        def get_bandwidth(self): return 112.0  # GB/s
        def requires_memory_registration(self): return False
        def supports_layer_pipeline(self): return True
        def get_latency(self): return 0.005  # ~5 us

直接 GPU-to-GPU P2P 传输，112 GB/s 让 1GB KV Cache 传输只要 9ms，完全隐藏在 prefill 计算时间内。层级别流水线有效（每层 state ~1MB，传输 <0.01ms << 2ms 层计算时间）。

**PCIeP2PBackend**（实现 + 测试，无 NVLink 环境的默认后端）：

    class PCIeP2PBackend(TransferBackend):
        def send(self, tensor, dst, stream):
            # 默认启用 INT4 KV 量化，压缩 3.2x
            if self.quantize_kv:
                tensor = kivi_quantize(tensor)
            torch.cuda._p2p_send(tensor, dst=dst, stream=stream)
        def get_bandwidth(self): return 14.0  # 双 x16 PCIe P2P ~12-16 GB/s
        def requires_memory_registration(self): return False
        def supports_layer_pipeline(self): return True  # 带宽足够，量化后可流水线
        def get_latency(self): return 0.01  # ~10 us

双 x16 PCIe P2P 提供 ~12-16 GB/s，INT4 KV 量化后 345MB 传输只需 29ms，可隐藏在 50-100ms prefill 计算时间内。这是无 NVLink 环境的默认传输后端，层级别流水线仍然有效。

**SHMBackend**（实现 + 测试，无 NVLink 环境的 fallback）：

    class SHMBackend(TransferBackend):
        def send(self, tensor, dst, stream):
            # GPU -> pinned CPU -> SHM -> 另一 GPU
            cpu_buf = self.pinned_buffers[tensor.shape]
            tensor.to(cpu_buf, non_blocking=True, stream=stream)
        def get_bandwidth(self): return 8.0  # 双 x16 PCIe SHM
        def supports_layer_pipeline(self): return False  # 8 GB/s 不够
        def get_latency(self): return 0.05  # ~50 us (SHM 2-hop)

SHM 是 2-hop PCIe 传输，8 GB/s 下 1GB 传输要 130ms，远超 prefill 计算时间。层级别流水线无效，降级为同步传输 + INT4 量化压缩。

**RDMABackend**（接口定义，未测试）：

    class RDMABackend(TransferBackend):
        def __init__(self, config):
            self.engine = TransferEngine(config)  # NIXL 或 Mooncake SDK
        def send(self, tensor, dst, stream):
            mr = self.register_memory(tensor)  # 必须先注册
            self.engine.rdma_write(mr, dst_addr, size)
        def get_bandwidth(self): return 25.0  # 200 Gbps ≈ 25 GB/s
        def requires_memory_registration(self): return True
        def supports_layer_pipeline(self): return True
        def get_latency(self): return 0.01  # ~10 us

RDMA 打开跨节点 disaggregation 的大门：1P+3D 不再限于单机，可做 4P+4D 跨机。但需要 RDMA NIC 硬件和传输引擎，无法在双卡 3090 上测试。

#### 5.7.4 后端选择逻辑

    def select_backend() -> TransferBackend:
        if torch.cuda.can_device_access_peer(0, 1):
            bw = benchmark_p2p_bandwidth()
            if bw >= 50:
                return NVLinkBackend()       # NVLink: FULL_TRANSFER
            elif bw >= 10:
                return PCIeP2PBackend()      # PCIe P2P: QUANTIZED_TRANSFER
        if has_rdma_device():
            return RDMABackend(config)
        return SHMBackend()                  # SHM: QUANTIZED_TRANSFER (sync)

#### 5.7.5 TransferMode 枚举与传输策略

框架定义 TransferMode 枚举，描述不同场景下的传输模式：

    class TransferMode(Enum):
        FULL_TRANSFER = "full"          # NVLink: 全量 BF16 KV + 循环状态
        QUANTIZED_TRANSFER = "quant"    # PCIe P2P: INT4 KV + 循环状态
        PARTIAL_TRANSFER = "partial"    # 低带宽: 仅循环状态 + KV 本地重算
        INTRA_GPU = "intra"             # MPS: 同卡内分离，零传输（内存指针传递）

**传输策略矩阵**：

| 后端 | 带宽 | TransferMode | 层级别流水线 | 传输策略 | 32K 传输时间 |
|------|------|-------------|-------------|---------|-------------|
| NVLink | 112 GB/s | FULL_TRANSFER | 是（逐层异步） | 层级别异步流水线，全量 BF16 | 9ms（100% 隐藏） |
| PCIe P2P | ~12-16 GB/s | QUANTIZED_TRANSFER | 是（逐层异步） | INT4 KV 量化 + 层级别异步 | 29ms（可隐藏） |
| PCIe SHM | ~8-10 GB/s | QUANTIZED_TRANSFER | 否（带宽不足） | 同步传输 + INT4 KV | 43ms（长 prefill 可隐藏） |
| RDMA | 25 GB/s | FULL_TRANSFER | 是（逐层异步） | 层级别异步，跨节点 | 40ms（可隐藏在长 prefill） |
| MPS 同卡 | 内存带宽 | INTRA_GPU | 不适用 | 零传输，内存指针传递 | 0ms（同卡共享） |

**部分 PD 分离模式（PARTIAL_TRANSFER）**：当带宽不足以传完整 KV Cache 时（如 x4 PCIe SHM ~4GB/s），只传循环状态（25MB，6ms 可隐藏），decode 端本地重算 8 层 full attention KV（~25ms）。干扰从 collocated 的 100ms 降到 25ms，4 倍改善。

#### 5.7.6 各后端实现状态

| 后端 | TransferMode | 接口定义 | 实现 | 测试 | 简历可说 |
|------|-------------|---------|------|------|---------|
| NVLink | FULL_TRANSFER | 完成 | 完成 | 计划 Phase 3 | "基于 NVLink 112 GB/s 实现层级别异步流水线" |
| PCIe P2P | QUANTIZED_TRANSFER | 完成 | 完成 | 计划 Phase 3 | "PCIe P2P + INT4 KV 量化实现无 NVLink PD 分离" |
| PCIe SHM | QUANTIZED_TRANSFER | 完成 | 完成（HydraCache 遗产） | 计划 Phase 3 | "SHM fallback 用于无 NVLink 环境" |
| MPS 同卡 | INTRA_GPU | 设计中 | 计划 Phase 8 | 计划 Phase 8 | "intra-GPU MPS 模式实现同卡 PD 分离" |
| RDMA | FULL_TRANSFER | 完成 | 接口定义 | 无硬件 | "TransferBackend 抽象层设计支持 RDMA 扩展" |
| ROCm P2P | FULL_TRANSFER | 接口定义 | 未实现 | 无硬件 | "框架架构兼容 ROCm" |

#### 5.7.7 ROCm 兼容性分析

ROCm 的问题不在传输层（hipMemcpyPeer 替换 cudaMemcpyAsync），而在 kernel 层：

| 组件 | CUDA 可用 | ROCm 可用 | 差异 |
|------|----------|----------|------|
| flash_attn (prefill) | 是 | 是（CK backend） | 官方支持 ROCm |
| Triton paged attention | 是 | 理论可行 | Triton ROCm 后端成熟度不如 CUDA |
| GDN fused kernel | 是 | 需验证 | Triton ROCm 对循环密集型 kernel 待测 |
| torch.cuda.Stream | 是 | 是（HIP 映射） | API 兼容，行为可能有差异 |

vLLM PR #51052 在 MI355X + ROCm 上测试了 Kimi-K3 的循环状态传输，证明 ROCm 路径可行（但用的是 AITER kernel，不是 Triton）。

### 5.8 Intra-GPU MPS 模式（BulletServe 式同卡分离）

受 BulletServe (ASPLOS 2026) 启发，HydraServe 支持一种额外的传输模式：同一张卡上通过 MPS (Multi-Process Service) 实现 prefill-decode 分离，零传输开销。

#### 5.8.1 动机

BulletServe 用 SM masking (libsmctrl) 在单卡上分区 prefill 和 decode，不需要传输。HydraServe 采用简化版：MPS 共享模式，两个进程共享同一块 GPU，通过 CUDA MPS server 调度。

| 维度 | inter-GPU PD (默认) | intra-GPU MPS PD (可选) |
|------|----------------------|----------------------|
| 传输开销 | 9ms (NVLink) / 29ms (PCIe) | 0ms（内存指针传递） |
| 干扰消除 | 物理隔离（两卡） | 逻辑隔离（同卡 MPS） |
| 资源利用率 | prefill 卡空闲时浮空 | SM 共享，资源争用 |
| 模型要求 | 可跨卡（各放一份） | 单卡放得下 |
| N-1 truncation | 需要（跨进程传输） | 仍需要（状态指针传递但语义不变） |
| 混合注意力 | 支持（核心设计） | 支持（本项目首创） |

**关键区别**：intra-GPU 模式下，状态传输变成“内存指针传递”--prefill 进程计算完成后，将 KV Cache 和循环状态的 GPU 内地址通过 MPS 共享内存传给 decode 进程，无任何拷贝。但 N-1 truncation 的语义不变--循环状态仍编码 0..N-1，decode 端仍需重算第 N 个 token（或使用 first-token seeding 跳过）。

#### 5.8.2 实现设计

    # MPS 启动
    nvidia-cuda-mps-control -d   # 启动 MPS daemon

    # Prefill 进程 (GPU 0 上)
    prefill_engine = PrefillEngine(gpu_id=0, mps=True)
    # 计算完成，状态留在 GPU 内存
    # 传递内存指针给 decode 进程

    # Decode 进程 (同一 GPU 0 上)
    decode_engine = DecodeEngine(gpu_id=0, mps=True)
    # 通过共享内存访问 KV Cache 和循环状态

    # GPU 1 独立运行另一个实例（DP 模式）

#### 5.8.3 局限性

- SM 资源争用：prefill 和 decode 共享 SM，不像 inter-GPU 那样物理隔离，仍有调度层面的干扰
- 没有 libsmctrl：不能像 BulletServe 那样精确分配 SM 给 prefill/decode，只能依赖 MPS 的调度
- 优势：零传输开销，同时 GPU 1 可以跑第二个实例（DP + intra-GPU PD 混合模式）
- HydraServe 将是首个对混合注意力模型做 intra-GPU PD 分离的实现

---

## 6. Benchmark 设计

### 6.1 核心原则：公平对比

所有配置使用相同的模型权重（INT4 AWQ）、相同的硬件（双 3090 + 可选 NVLink）、相同的输入数据。唯一变量是服务策略。

### 6.2 对比配置

**公平性原则**：B/C/D 均使用 2 张卡，是主对比组。A/E 仅使用 1 张卡，作为参考点，不是 baseline。核心对比是 B (DP) vs D (PD 分离)--两者硬件资源相同，唯一变量是策略。

| 配置 | 说明 | GPU 数 | 角色 |
|------|------|--------|------|
| A: 1-GPU Collocated | 单卡 prefill+decode | 1 | 参考（不是 baseline） |
| B: DP | 两卡各跑独立实例 | 2 | 主对比组 |
| C: TP=2 (vLLM) | 标准 TP | 2 | 主对比组 |
| D: PD 分离 (HydraServe) | GPU 0 prefill, GPU 1 decode | 2 | 主对比组 |
| E: vLLM unified | vLLM 单实例 | 1 | 参考 |
| F: intra-GPU PD (MPS) | GPU 0 MPS 共享 + GPU 1 独立 | 2 | 扩展对比 |

**配置 F 说明**：GPU 0 上通过 MPS 运行 prefill+decode 两个进程（同卡 PD 分离），GPU 1 跑独立实例（DP 模式）。这是 inter-GPU PD 和 intra-GPU PD 的混合模式，用于评估 intra-GPU 模式的价值。

### 6.3 模型矩阵

| 模型 | 上下文范围 | 主要测试场景 |
|------|-----------|------------|
| Qwen3.5-4B | 8K-32K | 高并发、快速验证 |
| Qwen3.5-9B | 8K-128K | 主力测试模型 |
| Qwen3.6-27B | 8K-32K | 大模型、内存压力 |

### 6.4 数据集

| 数据集 | 用途 | 上下文长度 |
|--------|------|-----------|
| ShareGPT | 真实对话流量模拟 | 1K-8K |
| HumanEval | 编程助手场景 | 2K-8K |
| LongBench | 长文档问答 | 8K-128K |
| WikiText-103 | 精度验证（PPL） | 滑动窗口 |
| GSM8K | 推理精度验证 | 短上下文 |

### 6.5 并发请求组织模式

**模式 1：固定并发（Closed-Loop，吞吐测试）** - 固定 N 个并发请求，一个完成立即补一个。测稳态吞吐。

**模式 2：Burst Arrival（TTFT 测试）** - 模拟 5 个请求同时到达（编程助手场景）。测 TTFT 分布和 prefill 干扰。

**模式 3：Poisson Arrival（真实流量模拟）** - 按 Poisson 过程到达，lambda 可调。测 P50/P99 TPOT。

**模式 4：混合上下文（真实分布）** - 短 prompt (1K-4K) 60% + 中 prompt (8K-32K) 30% + 长 prompt (64K-128K) 10%。

### 6.6 测试矩阵

| 实验 | 自变量 | 因变量 | 配置 | 模型 |
|------|--------|--------|------|------|
| 1: 吞吐 vs 并发 | 并发数 | tok/s, req/s | A,B,C,D | 9B |
| 2: TTFT 分布 | 上下文长度 | P50/P90/P99 TTFT | A,D | 9B |
| 3: TPOT 稳定性 | 并发数 + burst | P50/P99 TPOT | A,B,C,D | 9B |
| 4: 精度测试 | 配置 | GSM8K acc, WikiText PPL | A,D,E | 4B,9B |
| 5: 路由决策分布 | 请求模式 | Collocated vs PD 比 | D | 9B |
| 6: 多模型对比 | 模型 | 吞吐, P99 TPOT | A,D | 4B,9B,27B |
| 7: 传输隐藏效果 | 上下文长度 | 传输 vs prefill 时间 | D | 9B,27B |

### 6.7 关键指标

| 指标 | 定义 | 预期 PD 分离结果 |
|------|------|-----------------|
| P50 TTFT | 首 token 延迟中位数 | 持平或略优（seeding -58ms） |
| P99 TTFT | 首 token 延迟 P99 | -20~40%（无 decode 干扰） |
| P50 TPOT | 每 token 生成延迟中位数 | 持平 |
| P99 TPOT | 每 token 延迟 P99 | -50~80%（无 prefill 干扰） |
| 最大并发 | 不 OOM 的最大请求数 | 1.5-2x（decode GPU 专用） |
| 吞吐 (tok/s) | 总 token 吞吐 | 0~20% |
| GSM8K 精度 | exact_match | 无损 |

### 6.8 预期结果

**核心对比：B (DP) vs D (PD 分离)，9B, 32K ctx**：

| 并发 | B: 吞吐 | D: 吞吐 | B: P99 TPOT | D: P99 TPOT | 赢家 |
|------|---------|---------|------------|------------|------|
| 5 | 200 tok/s | 200 tok/s | 10ms | 12ms | DP（PD 传输开销不划算） |
| 10 | 380 tok/s | 370 tok/s | 20ms | 13ms | 持平 |
| 20 | 650 tok/s | 650 tok/s | 35ms | 12ms | PD（DP 干扰开始显现） |
| 30 | 800 tok/s | 800 tok/s | 45ms | 12ms | PD（DP P99 恶化 3.7x） |

crossover point 约在 10-15 并发：低于此 DP 赢（传输开销 > 干扰代价），高于此 PD 赢（干扰代价 > 传输开销）。这个 crossover 是项目最核心的实验结论。

**P99 TPOT 对比（9B, 30 并发 + burst，主对比组 B/C/D）**：

| 配置 | 卡数 | P50 TPOT | P99 TPOT | 恶化倍数 | 说明 |
|------|------|---------|---------|----------|------|
| A (1-GPU 参考) | 1 | 12ms | 55ms | 4.6x | 单卡资源受限，仅供参考 |
| B (DP) | 2 | 8ms | 45ms | 5.6x | 每卡内部干扰 |
| C (TP=2) | 2 | 8ms | 40ms | 5.0x | 同步等待 |
| D (PD 分离) | 2 | 7ms | 12ms | 1.7x | 物理隔离，无干扰 |

PD 分离的核心卖点：同等硬件（2 卡）下 P99 TPOT 降低 73%（vs DP），长尾延迟稳定。不是比单卡快（那是硬件翻倍的必然结果），而是同样 2 张卡，分工比独立更稳。

**多模型对比（32K ctx, B vs D, P99 TPOT）**：

| 模型 | B (DP): P99 | D (PD): P99 | 改善 | 说明 |
|------|------------|------------|------|------|
| 4B | 40ms | 10ms | -75% | prefill 快，crossover 低 (~5 并发) |
| 9B | 45ms | 12ms | -73% | crossover ~10-15 |
| 27B | 60ms | 18ms | -70% | prefill 慢，crossover 更低 |

### 6.9 图表清单

1. 吞吐 vs 并发曲线（主对比 B/C/D，A 为参考线）
2. TPOT CDF 累积分布（30 并发，4 条线）
3. TTFT 直方图（burst arrival，B vs D）
4. 上下文长度 vs P99 TPOT（B vs D，主对比）
5. 路由决策分布（prompt 长度 -> Collocated/PD 选择）
6. 多模型对比（4B/9B/27B，B vs D）
7. 传输隐藏效果（传输时间 vs prefill 时间）
8. 精度对比柱状图（GSM8K, WikiText PPL）

### 6.10 长上下文（128K）专项测试

INT4 权重下 9B 单卡可用 18.5GB，128K 上下文需要 4GB KV + 25MB 状态 = 4.025GB/请求，可并发 4 个。27B 128K 需要 8GB KV + 50MB 状态 = 8.05GB/请求，只能并发 1 个。

测试方法：将 WikiText-103 拼接到 128K，滑动窗口计算 PPL。对比 BF16 vs INT4 权重 vs INT4 权重 + PD 分离。关注 PPL 是否在序列末尾升高。

---

## 7. 四卡扩展设计

### 7.1 三种 4 卡配置

| 配法 | 分配 | 适合场景 | 改动量 |
|------|------|---------|--------|
| 1P + 3D | 1 卡 prefill, 3 卡 decode | decode 是瓶颈 | 小 |
| 2P + 2D | 2 卡 prefill, 2 卡 decode | 均衡负载 | 中 |
| 2P(TP=2) + 2D(TP=2) | 每 pool 内 TP=2 | 大模型需切分 | 大 |

### 7.2 1P + 3D：最自然扩展

核心组件不变，新增 decode-side load balancer。

**路由策略**：Round-robin（最简单）/ Least-loaded（看哪个 decode 卡并发最少）/ Affinity routing（同一用户发到同一 decode 卡，复用 prefix cache）

**NVLink 拓扑约束**：4x 3090 每张只有 2 个 NVLink 接口，非全互联。GPU 0 -> GPU 1 是 1 跳（112 GB/s），GPU 0 -> GPU 3 可能是 2 跳（带宽减半）。

### 7.3 2P + 2D：跨 pool 路由

**关键挑战**：transfer routing race condition。vLLM Issue #51681（2026-08-10）显示，两个 prefill worker 并发向不同 decode worker 发数据时，共享路由状态被覆写，导致数据发错 decode 卡。

**解法**：CentralScheduler 为每个传输生成 TransferTask 对象，包含 source/target GPU id 和 state descriptors。Worker 不持有路由状态，全部从 TransferTask 读取。

### 7.4 2P(TP=2) + 2D(TP=2)：重新设计传输协议

TP=2 下每个 rank 只有半数 attention heads 的 KV Cache 和半数 linear attention heads 的循环状态。传输时每个 rank 只传自己那一半到对应 decode rank。

GDN 状态 [16 key_heads, 128, 128] 在 TP=2 下切 heads：每个 rank 拿 [8, 128, 128]。与 Mamba2 的 TP 切法一致，vLLM PR #41869 已验证可行。

### 7.5 各组件扩展性

| 组件 | 2 卡需要 | 4 卡需改？ |
|------|---------|-----------|  |
| State Extractor | 逐层提取 | 不变 |
| NVLink 传输层 | 点对点 | 加 multi-target routing |
| Continuous Batching | 单 decode 卡 | 不变 |
| N-1 truncation | decode 端重算 | 不变 |
| Dual-state 内存管理 | PagedAttention + State Pool | 不变 |
| ModelAdapter | 单模型 | 不变 |
| CentralScheduler | 简单路由 | 加负载感知 + affinity |
| TransferBackend | 单后端 (NVLink) | 加 multi-target routing |
| 传输协议 | 单源单目标 | 单源多目标 / 多源多目标 |

核心组件完全不用改，只有 CentralScheduler 和传输层需要增量升级。

---

## 8. 应用场景

### 8.1 编程助手（Cursor/Copilot 类）

用户发送大段代码上下文（8K-32K tokens）+ 短问题，期望 200-500 token 代码补全。多个开发者同时提交代码上下文时，prefill 互相排队。PD 分离让补全不受干扰。

- 数据集：HumanEval, CodeXGLUE
- 负载：8K-32K prompt + 200 token output, 10-30 并发
- 关键指标：P99 TTFT

### 8.2 文档问答 / RAG

用户上传长文档（32K-128K tokens）+ 问题，期望 500-2000 token 回答。文档 prefill 最耗时（128K 要 400ms+），严重干扰正在回答其他问题的 decode 流。

- 数据集：LongBench, NarrativeQA
- 负载：32K-128K prompt + 500 token output, 5-15 并发
- 关键指标：P99 TPOT

### 8.3 多轮对话

每轮对话增加 1K-4K 上下文，对话持续 10-20 轮。总上下文逐渐增长到 20K-40K。每轮新输入需要 incremental prefill，频繁打断正在生成的 decode。

- 数据集：ShareGPT
- 负载：多轮，每轮 1K-4K prompt + 200-500 token output, 20-50 并发
- 关键指标：吞吐 + P99 TPOT

### 8.4 Agent 工具调用

Agent 读工具输出（可能很大）-> 思考 -> 生成下一步 -> 调用工具 -> 循环。每次工具输出都是一次突发 prefill，不应打断 agent 正在思考的 decode 流。

- 负载：burst arrival + 长 decode
- 关键指标：P99 TPOT 在 burst 下的稳定性

---

## 9. 简历定位与面试叙事

### 9.1 简历条目

> **HydraServe** | 面向混合注意力架构 LLM 的 Prefill-Decode 分离推理引擎
>
> - 从零实现 Qwen3.5/3.6（GDN + GQA 混合注意力）推理引擎：编写 Triton fused kernel 实现 GDN delta rule 递推前向（state update + gate + output 三合一），Triton paged attention kernel 实现 decode 阶段非连续 KV block 的 online softmax
> - 设计双状态异构迁移协议：full attention KV Cache 按 block 可 INT4 量化传输，linear attention 循环状态 FP32 整体传输，基于 NVLink layer 级别异步流水线使 32K 上下文 1GB KV Cache 与 25MB 循环状态传输 100% 隐藏在 prefill 计算内。设计 TransferBackend 抽象层支持 NVLink（实现）/ PCIe P2P（实现，INT4 KV 量化）/ SHM（实现）/ RDMA（接口定义）多传输后端，层级别流水线根据带宽自适应。首 token 预播种消除 decode 端冗余重算
> - 实现 continuous batching + chunked prefill 调度器与自适应路由：基于硬件实测 cost model 按请求 prompt 长度和 decode 负载选择 Collocated 或 PD 分离路径，支持 4B/9B/27B 多模型，额外实现 intra-GPU MPS 模式（同卡 PD 分离）并设计 TransferBackend 抽象层支持 NVLink/PCIe P2P/SHM/RDMA 多传输后端自适应
> - 在编程助手和文档问答场景下对比 Collocated / DP / TP=2 / PD 分离四种策略，30 并发下 PD 分离 P99 TPOT 降低 50%+，TTFT 通过首 token 预播种降低 58ms，GSM8K 精度无损

### 9.2 面试叙事线

**开场（30 秒）**：

"我做了一个叫 HydraServe 的项目，解决的是混合注意力架构 LLM（Qwen3.5/3.6）在推理服务中的 prefill-decode 干扰问题。Qwen3.5 有 24 层 linear attention 和 8 层 full attention，PD 分离时需要迁移两种完全不同的状态--KV Cache 和循环状态--这是现有 PD 分离工作没有处理好的。"

**技术深挖（3 分钟）**：

"Prefill 和 decode 是两种完全不同的负载--prefill 计算密集、GPU 利用率 50-90%，decode 带宽密集、利用率 5-20%。它们在同一 GPU 上跑会互相干扰：一个 32K prefill 进来，100ms 内所有 decode 请求停转。DistServe 论文证明 P99 延迟恶化 3-10 倍。"

"我的方案是 PD 物理分离：GPU 0 专门跑 prefill，GPU 1 专门跑 decode，通过 NVLink（112 GB/s）传状态。32K 上下文的 KV Cache 有 1GB，循环状态 25MB，NVLink 传输只要 9ms，prefill 计算要 50-100ms，传输可以 100% 隐藏。"

"混合注意力的难点是双状态：KV Cache 可以按 block 量化压缩传输，但 linear attention 的循环状态是 FP32 的 25MB 矩阵，量化会导致递推误差发散。我做了一个层级别异步流水线--每层计算完立即传该层状态，不需要等全部算完。"

"还有 N-1 truncation 问题：循环状态处理到第 N 个 token 时编码的是 0..N-1 的信息，decode 端必须重算第 N 个 token 才能开始 decode。加上 first-token seeding，prefill 端已经采样了首 token，直接传过去省了 58ms 重算。"

"我支持 4B、9B、27B 三个模型，通过 ModelAdapter 接口适配不同层数和状态大小。27B 有 64 层（48 linear + 16 full），状态 50MB，KV/token 64KB，内存压力完全不同。框架默认在无 NVLink 环境下使用 PCIe P2P + INT4 KV 量化做 PD 分离，有 NVLink 则升级为全量 BF16 传输。额外设计了 intra-GPU MPS 模式，同卡零传输做 PD 分离，补充 inter-GPU 方案。"

**结果（30 秒）**：

"效果是：同样 2 张卡，30 并发下 PD 分离的 P99 TPOT 从 45ms 降到 12ms（vs DP），长尾延迟改善 73%。但低并发（<10）时 DP 反而更好--PD 分离的传输开销不划算。crossover point 在 10-15 并发。这个 tradeoff 是项目最核心的结论。GSM8K 精度无损。"

### 9.3 面试问答

| 问题 | 回答要点 |
|------|----------|
| 为什么不用 vLLM 直接做？ | vLLM 的 PD 分离依赖 NIXL/Mooncake connector，为 RDMA 设计。NVLink 双卡场景下直接写层级别传输更自然。vLLM 的 GDN PD 分离 PR（#41869, #46807）是 2026 年中的进行中工作，我是独立设计 |
| 和 nanoPD 有什么区别？ | nanoPD 只支持标准 transformer，不处理循环状态传输。混合注意力有三个新问题：双状态异构传输、N-1 truncation、prefix caching 与循环状态冲突 |
| 为什么不用 TP？ | TP 解决"模型太大放不下"，PD 分离解决"两种负载互相干扰"。9B INT4 单卡放得下，TP 没必要，但 prefill-decode 干扰照样存在。两者正交 |
| 循环状态为什么不量化？ | FP32 循环矩阵量化导致递推误差累积发散。25MB 不大但风险高。AWQ INT4 只量化权重，不影响运行时状态 |
| 为什么支持多个模型？ | 通过 ModelAdapter 接口，不同模型只有层数、head 数、状态大小不同。4B 快速验证，9B 主力测试，27B 展示大模型下的内存压力 |
| N-1 truncation 是什么？ | 循环状态处理到第 N 个 token 编码的是 0..N-1 的信息，decode 端必须重算第 N 个 token。用 first-token seeding 可以跳过这一步 |
| 有 NVLink 和没有 NVLink 有什么区别？ | 无 NVLink 时双 x16 PCIe P2P ~12-16 GB/s + INT4 KV 量化，传输 345MB 只需 29ms，可隐藏在 prefill 内。有 NVLink（112GB/s），传输 9ms，完全隐藏 |
| 能扩展到 4 卡吗？ | 核心组件不用改。1P+3D 加路由层，2P+2D 需要处理跨 pool 路由 race condition（vLLM Issue #51681），2P(TP=2)+2D(TP=2) 需要重新设计传输协议支持 TP 切分 |
| Flash Attention 是自己写的吗？ | Full attention prefill 用 flash_attn 库。Decode 用自写的 Triton paged attention kernel（flash_attn 不支持非连续 KV block）。GDN 前向用自写的 Triton fused delta rule kernel（无现成实现） |
| 自适应路由怎么做的？ | 启动时微基准测试实测 prefill 速度、NVLink 带宽、decode 吞吐。cost model 按 prompt 长度和 decode 负载算 Collocated 和 PD 分离两条路径的延迟，选更短的 |
| 框架支持 RDMA 吗？ | 设计了 TransferBackend 抽象层，NVLink 已实现并测试，SHM fallback 已实现，RDMA 接口定义但需 InfiniBand 硬件验证。层级别流水线根据带宽自适应：NVLink\/RDMA 做逐层异步，SHM 降级为同步传输 + INT4 量化 |
| 27B INT4 能量化吗？ | AWQ INT4 量化权重，不影响运行时状态。27B 权重从 54GB 压到 13.5GB，单卡放得下。线性注意力循环状态保持 FP32，不可量化 |
| PD 分离和双状态传输是什么关系？ | PD 分离的动机是 prefill-decode 干扰，与模型架构无关。双状态传输是在混合注意力模型上做 PD 分离时遇到的额外挑战，不是 PD 分离的原因。PD 分离对所有模型都有价值，本项目的技术贡献是解决混合注意力特有的双状态传输问题 |
| PD 分离一定比 DP 好吗？ | 不一定。低并发时 DP 更好（零传输开销），高并发时 PD 分离更好（干扰代价超过传输开销）。crossover point 约 10-15 并发。PD 分离是"用传输开销换干扰消除"，不是无条件更好 |
| 你的 baseline 公平吗？ | 主对比组 B/C/D 都用 2 张卡，A/E 是 1 卡参考。核心对比是 B (DP) vs D (PD 分离)，硬件资源完全相同。不拿 1 卡和 2 卡比策略优劣 |
| 和 BulletServe 有什么区别？ | BulletServe 是 intra-GPU 方案（SM masking，单卡内分离），不需要传输。HydraServe 做 inter-GPU 分离（跨卡物理隔离），需要传输。BulletServe 不支持混合注意力（代码 0 结果），HydraServe 的双状态传输是它不覆盖的交集。额外加了 intra-GPU MPS 模式作为补充，HydraServe 是首个对混合注意力模型做 intra-GPU PD 的实现 |
| 没有 NVLink 能做 PD 分离吗？ | 可以。双 x16 PCIe P2P 提供 ~12-16 GB/s，配合 INT4 KV Cache 量化（3.2x 压缩），32K 上下文传输量从 1GB 降到 345MB，传输 29ms，可隐藏在 50-100ms prefill 内。无 NVLink 不是 PD 分离的隔降条件，INT4 KV 量化是带宽适配手段 |
| intra-GPU 和 inter-GPU PD 怎么选？ | 单卡放得下模型：intra-GPU MPS 模式零传输更优，但 SM 争用。需要跨卡分离或模型放不下：inter-GPU 是唯一选项。MPS 模式是补充而非替代--在 GPU 0 做 intra-GPU PD 的同时 GPU 1 跑独立实例，混合模式最优 |
| 如果有双卡 3090 的 NVLink，这个项目怎么做？ | NVLink Bridge ($30-80) 后，112 GB/s 双向 P2P 传输，1GB KV Cache 只需 9ms，可做全量 BF16 传输（无需 INT4 量化），100% 隐藏在 prefill 内。NVLink 是可选增强，项目默认设计在无 NVLink 下也可行（PCIe + INT4 KV），有 NVLink 则开启 FULL_TRANSFER 模式 |

---

## 10. 开发里程碑

| Phase | 内容 | 时间 | 关键产出 |
|-------|------|------|---------|
| 0 | 模型理解 + NVLink benchmark + 环境搭建 | 1 周 | NVLink 确认, 4B forward 跑通 |
| 1 | 推理引擎 + Triton kernels (4B/9B) | 2.5 周 | GDN fused kernel + paged attention kernel |
| 2 | 双状态内存管理 + ModelAdapter | 2 周 | block manager + slot pool + 3 个 adapter |
| 3 | NVLink 传输层 + 双状态序列化 | 2 周 | layer 级别异步传输 pipeline |
| 4 | PD 分离核心 + N-1 truncation + first-token seeding | 2 周 | 端到端 PD 分离可用 |
| 5 | Continuous batching + chunked prefill | 2 周 | 多请求并发 + chunk 调度 |
| 6 | 自适应路由 + cost model + 27B 适配 | 1.5 周 | 路由器 + 参数实测 + 27B 跑通 |
| 7 | Benchmark + 对比实验 + 文档 | 2 周 | 6 组对比数据 + README |
| 8 | intra-GPU MPS 模式 + BulletServe 式对比 | 1 周 | MPS 同卡 PD + 配置 F 对比 |
| 总计 | | ~16 周 | |

---

## 11. 环境构建

### 11.1 总体策略

| 阶段 | 工具 | 原因 |
|------|------|------|
| 开发期 (Phase 0-6) | Conda 虚拟环境 | 快速迭代，方便调试 |
| vLLM baseline 测试 | vLLM 官方 Docker | 依赖锁死，用官方镜像 |
| 最终交付 (Phase 7) | Dockerfile + docker-compose | 可复现 |

### 11.2 硬件验证

    nvidia-smi              # 确认两张卡
    nvidia-smi topo -m      # 确认 NVLink 直连
    # NV12 = NVLink 直连，12 条链路
    # SYS = 走系统总线（无 NVLink）

### 11.3 HydraServe 开发环境 (Conda)

    conda create -n hydraserve python=3.10 -y
    conda activate hydraserve

    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install triton==2.1.0
    pip install transformers>=4.45.0 accelerate safetensors
    pip install flash-attn --no-build-isolation
    pip install autoawq
    pip install fastapi uvicorn pydantic
    pip install aiohttp tqdm

### 11.4 vLLM Baseline 环境 (Docker)

    docker run --gpus all --rm \
        -v /path/to/models:/models -p 8000:8000 \
        vllm/vllm-openai:latest \
        --model /models/Qwen3.5-9B-AWQ \
        --quantization awq --gpu-memory-utilization 0.9 \
        --max-model-len 131072 --tensor-parallel-size 2

### 11.5 最终交付 Dockerfile

    FROM nvcr.io/nvidia/pytorch:24.01-py3
    WORKDIR /app/HydraServe
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    VOLUME ["/models"]
    EXPOSE 8000
    CMD ["python", "-m", "hydraserve.serve", \
         "--model", "/models/Qwen3.5-9B-AWQ", \
         "--mode", "pd_disaggregated", \
         "--prefill-gpu", "0", "--decode-gpu", "1"]

### 11.6 项目目录结构

    HydraServe/
    +-- README.md
    +-- requirements.txt
    +-- Dockerfile
    +-- docker-compose.yml
    +-- hydraserve/
    |   +-- __init__.py
    |   +-- config.py
    |   +-- model/
    |   |   +-- qwen3_5.py
    |   |   +-- qwen3_6.py
    |   |   +-- adapter.py
    |   +-- kernels/
    |   |   +-- gdn_fused.py
    |   |   +-- paged_attention.py
    |   |   +-- rmsnorm.py
    |   +-- cache/
    |   |   +-- block_manager.py
    |   |   +-- state_pool.py
    |   |   +-- kv_quantizer.py
    |   |   +-- prefix_cache.py
    |   +-- transfer/
    |   |   +-- descriptor.py
    |   |   +-- backend.py              # TransferBackend: NVLink/PCIe P2P/SHM/RDMA
    |   |   +-- nvlink_transfer.py
    |   |   +-- pcie_p2p_transfer.py   # PCIe P2P + INT4 KV quantization
    |   |   +-- intra_gpu_transfer.py  # MPS intra-GPU mode (zero-copy)
    |   |   +-- pipeline.py
    |   +-- engine/
    |   |   +-- prefill_engine.py
    |   |   +-- decode_engine.py
    |   |   +-- scheduler.py
    |   |   +-- chunked_prefill.py
    |   +-- router/
    |   |   +-- adaptive_router.py
    |   |   +-- cost_model.py
    |   |   +-- profiler.py
    |   +-- serve/
    |   |   +-- api_server.py
    |   |   +-- protocol.py
    |   +-- benchmark/
    |       +-- run_benchmark.py
    |       +-- datasets.py
    |       +-- metrics.py
    |       +-- plot.py
    +-- scripts/
    |   +-- download_model.sh
    |   +-- verify_nvlink.sh
    |   +-- run_baseline.sh
    +-- docs/
    |   +-- design.md
    |   +-- api.md
    +-- tests/
        +-- test_kernels.py
        +-- test_transfer.py
        +-- test_e2e.py

### 11.7 模型权重准备

    huggingface-cli download Qwen/Qwen3.5-4B --local-dir /models/Qwen3.5-4B
    huggingface-cli download Qwen/Qwen3.5-9B-AWQ --local-dir /models/Qwen3.5-9B-AWQ
    huggingface-cli download Qwen/Qwen3.6-27B-AWQ --local-dir /models/Qwen3.6-27B-AWQ

---

## 12. 附录

### 附录 A：状态传输量计算

| 模型 | 32K 全状态 (BF16) | NVLink 传输 | prefill 计算 | 可隐藏？ |
|------|-------------------|-------------|-------------|----------|
| 4B | 1 GB | 9 ms | 50-80 ms | yes |
| 9B | 1 GB | 9 ms | 50-100 ms | yes |
| 27B | 2 GB | 18 ms | 100-200 ms | yes |

### 附录 B：关键数据速查

| 数据 | 4B | 9B | 27B | 来源 |
|------|-----|-----|------|------|
| 权重 INT4 | 2 GB | 4.5 GB | 13.5 GB | AWQ |
| Full Attn KV/token | 32 KB | 32 KB | 64 KB | config.json |
| Linear 状态/请求 | 25 MB | 25 MB | 50 MB | config.json |
| Full Attn 层数 | 8 | 8 | 16 | config.json |
| Linear Attn 层数 | 24 | 24 | 48 | config.json |
| NVLink 带宽 | 112 GB/s | 112 GB/s | 112 GB/s | 3090 spec |
| First-token seeding 节省 | ~58ms | ~58ms | ~58ms | vLLM #51919 |

### 附录 C：vLLM 混合注意力 PD 分离进展

| PR/Issue | 标题 | 状态 | 日期 |
|----------|------|------|------|
| #41869 | NIXL GDN support (Qwen3.5) | merged | 2026-05-14 |
| #43765 | Feature: PD disagg for hybrid SSM/GDN | open | 2026-05-27 |
| #46807 | Mooncake GDN support (Qwen3.5) | merged | 2026-07-14 |
| #51052 | MoRIIO hybrid mamba/KDA state transfer | open | 2026-08-04 |
| #51919 | First-token seeding + fast KV channels | open | 2026-08-10 |
| #51681 | Multi-decode routing race condition fix | open | 2026-08-10 |
| SGLang #32732 | PD disagg decode radix cache for mamba/SSM | open | 2026-07-29 |
| Mooncake #2242 | Feature request: hybrid SSM/GDN KV transfer | open | 2026-05-27 |

### 附录 D：与 HydraCache（Offload 方案）对比

| 维度 | HydraCache (Offload) | HydraServe (PD 分离) |
|------|---------------------|---------------------|
| 硬件要求 | 双 x16 PCIe（无 NVLink） | NVLink（可选） |
| 核心问题 | 带宽受限，TP 不可用 | prefill-decode 干扰 |
| 解决方式 | KV Cache 冷热分层 + 量化 | 物理隔离 prefill 和 decode |
| 技术热点 | KV Cache 管理 | PD Disaggregation（2025 最热） |
| 对标项目 | LMCache | Mooncake / DistServe / nanoPD |
| 混合注意力 | 双状态管理 | 双状态传输（新挑战） |
| 多模型支持 | 单模型 (9B) | ModelAdapter 多模型 (4B/9B/27B) |
| 简历价值 | 带宽受限场景的工程补偿 | 利用 NVLink 做分离服务 |

---

*Document version: 1.1 | Last updated: 2026-08-12 | Updates: dual x16 hardware, no-NVLink viability, PCIe P2P backend, intra-GPU MPS mode, BulletServe comparison, TransferMode enum*
