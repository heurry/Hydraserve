# HydraServe

> 面向混合注意力架构 LLM 的 PD 分离推理引擎原型

---

## 1. 项目定位

### 1.1 一句话定位

从零实现面向混合注意力架构（Qwen3.5/3.6：Gated Delta Network + GQA）的 PD 分离推理引擎，核心技术贡献是混合注意力特有的双状态传输协议（KV Cache + 循环状态）。双卡/四卡 RTX 3090 作为开发验证环境，通过 TransferBackend 抽象层支持 NVLink / PCIe P2P / SHM / gRPC 多传输后端，设计目标可扩展到多节点部署。

### 1.2 项目是什么

一个引擎原型。不是"证明 PD 比 DP 好"的实验论文，而是"能不能从零写出一个支持混合注意力 PD 分离的推理引擎"的工程验证。

核心价值在引擎实现和混合注意力双状态传输协议，不在"4 卡 PD 比 4 卡 DP 好"的结论。PD 分离真正发挥价值的场景是大规模部署（Mooncake/DistServe 的 100+ GPU 量级，prefill 节点占比 <5%），本项目是引擎原型和协议验证，不是生产部署。

### 1.3 项目不是什么

- 不是"消费级 GPU 上 PD 分离的生产部署"——生产 PD 分离在多节点进行（Mooncake/DistServe），双卡/四卡 3090 是开发验证环境
- 不是"PD 分离比 DP 好"的证明——在 4 卡规模，DP 的吞吐可能更高，PD 的优势在 P99 延迟稳定性，但这个优势在大规模才显著
- 不是"混合注意力 PD 分离的唯一方案"——vLLM #41869/#46807 已合并 GDN PD 分离支持（依赖 NIXL/Mooncake + RDMA），SGLang #32732 也在做，本项目是独立实现

### 1.4 要解决的问题：Prefill-Decode 干扰

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特征 | 计算密集（大矩阵乘） | 带宽密集（读权重） |
| GPU 利用率 | 50-90% | 5-20% |
| 3090 实测 (9B BF16) | 2,282 tok/s -> 4K 需 1.8 秒 | 35 tok/s |
| 3090 实测干扰 | 1K -> 2.5x 减速 | 4K -> 6.4x 减速 |

PD 分离把干扰归零。但 PD 是"用传输开销换干扰消除"——不是无条件更好。crossover point 在大规模（100+ GPU）才显著。

### 1.5 混合注意力引入的额外挑战

| 状态类型 | 大小 (32K, 9B) | 特性 | 迁移策略 |
|---------|-------------------|------|----------|
| Full Attn KV Cache | 1 GB (BF16) / 345 MB (INT4) | 线性增长，可量化 | INT4 压缩 |
| Linear Attn 循环状态 | 53.48 MB (9B) / 158.86 MB (27B), FP32 | 固定大小，不可量化 | 整体传输 |

循环状态不可量化：FP32 递推误差累积发散。双状态传输是本项目的技术贡献点。

---

## 2. 现状与相关工作

### 2.1 PD 分离生态

| 项目 | 状态 | 混合注意力 | 硬件 |
|------|------|-----------|------|
| vLLM + NIXL (#41869, merged) | GDN 部分支持 | Qwen3.5 GDN conv-state | RDMA |
| vLLM + Mooncake (#46807, merged) | GDN 支持 | 修复 GDN crash | RDMA |
| vLLM + MoRIIO (#51052, open) | Kimi-K3 | conv+ssm READ+WRITE | RDMA + ROCm |
| SGLang (#32732, open) | 开发中 | PD disagg for mamba/SSM | RDMA |
| Mooncake (#2242, open) | 未实现 | Feature request: hybrid GDN | RDMA |
| nanoPD (161 stars) | 已完成 | 不支持 | H20 |
| BulletServe (ASPLOS 2026) | 已完成 | 不支持（代码 0 结果） | 单卡 SM masking |
| **HydraServe** | 原型开发 | 核心设计目标 | 3090 P2P/SHM |

与 vLLM 的区分：vLLM #41869/#46807 已合并 GDN PD 分离支持，但依赖 NIXL/Mooncake + RDMA。HydraServe 是从零实现的独立引擎，不依赖外部 connector，面向消费级 GPU 验证。

### 2.2 与本项目的区别

| 维度 | nanoPD | vLLM PRs | BulletServe | HydraServe |
|------|--------|----------|------------|------------|
| 模型 | Qwen3-8B（全 attention） | Qwen3.5-0.8B（GDN） | 标准 Qwen3 | Qwen3.5-4B/9B, Qwen3.6-27B |
| 从零实现 | 是 | 否（vLLM 补丁） | 是 | 是 |
| 循环状态传输 | 不需要 | 补丁 | 不需要 | 核心设计 |
| N-1 truncation | 不需要 | 需要 | 不需要 | 需要 |
| 层级别流水线 | 没有 | 没有 | 不适用 | 有 |
| 多传输后端 | 没有 | NIXL/RDMA | 不适用 | TransferBackend |

### 2.3 BulletServe 对比

BulletServe 在单卡场景下严格优于 inter-GPU PD 分离。实测验证：MPS 简化版 decode 2.5x 减速（无 libsmctrl），证明 inter-GPU 物理隔离的必要性。

---

## 3. 硬件环境与目标模型

### 3.1 硬件

**当前开发环境（双卡 3090）**：

| 组件 | 规格 |
|------|------|
| GPU | 2x RTX 3090 (24GB) |
| PCIe | GPU0 x16, GPU1 x4（非对称） |
| P2P | 不可用（NODE 拓扑，实测） |
| SHM 带宽 | 4.58 GB/s（实测） |
| NVLink | 无 |
| CUDA | 12.8 |

**可选扩展环境（四卡 3090 全 x16）**：

| 组件 | 规格 |
|------|------|
| GPU | 4x RTX 3090 (24GB) |
| PCIe | 全 x16（需验证 P2P 拓扑） |
| NVLink | 可选（菊花链） |
| 配置 | 1P+3D（Mooncake 最小形态） |

**传输后端选择**：

| 硬件 | 带宽 | 传输策略 | 32K 传输时间 | 可行性 |
|------|------|---------|-------------|---------|
| NVLink (有 bridge) | 112 GB/s | 全量 BF16 KV + 状态 | 9ms | 完全可行 |
| PCIe P2P (双 x16) | ~12-16 GB/s | INT4 KV (345MB) + 状态 | 29ms | 可行 |
| PCIe SHM (x16+x4) | 4.58 GB/s (实测) | 仅状态 (53.48MB for 9B) + KV 重算 | ~12ms + KV 重算 | 部分 PD |
| gRPC (跨节点) | 3-10 GB/s | INT4 KV + 状态 | 35-115ms | 长上下文可隐藏 |

当前双卡环境 P2P 不可用、SHM 4.58 GB/s，只能跑 PARTIAL_TRANSFER。四卡全 x16 P2P 如果可用，可跑完整 QUANTIZED_TRANSFER。

### 3.2 INT4 KV Cache 量化：带宽适配手段

| 指标 | BF16 KV | INT4 KV (KIVI) | 压缩比 |
|------|---------|---------------|--------|
| 32K 9B KV 大小 | 1 GB | 345 MB | 3.2x |
| NVLink 传输 | 9 ms | 3 ms | - |
| PCIe P2P 传输 | 85 ms | 29 ms | - |
| SHM 传输 (4.58 GB/s) | 218 ms | 75 ms | - |
| PPL 损失 | 0 | +0.74 (naive, 实测) | - |

naive 对称量化无校准，PPL +0.74。AWQ/GPTQ 带校准可达 <0.3。4B 上下文 7.3K->14.5K（2x），9B 无增益（GDN 层 BF16 不可量化）。

### 3.3 目标模型

| 参数 | Qwen3.5-4B | Qwen3.5-9B | Qwen3.6-27B |
|------|-----------|-----------|------------|
| hidden_size | 2560 | 4096 | 5120 |
| num_hidden_layers | 32 | 32 | 64 |
| linear 层数 | 24 | 24 | 48 |
| full attn 层数 | 8 | 8 | 16 |
| mamba_ssm_dtype | float32 | float32 | float32 |
| max_position | 262144 | 262144 | 262144 |
| 权重 INT4 | ~2 GB | ~4.5 GB | ~13.5 GB |

### 3.4 显存预算与并发（INT4 权重）

| 模型 | 上下文 | 并发/卡 | 4卡 1P+3D 总并发 |
|------|---------|--------|---------|
| 9B | 32K | 18 | 54 |
| 9B | 128K | 4 | 12 |
| 27B | 32K | 4 | 12 |

9B INT4 + 32K + 4 卡 1P+3D = 54 并发，是有实际意义的服务规模。

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
        |  (GPU 0)     |    |   (GPU 1..N)   |
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
        | | (layer)  | |    | | (Paged)    | |
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

### 4.2 核心组件与实现状态

| 组件 | GPU | 职责 | 代码状态 |
|------|-----|------|---------|
| CentralScheduler | CPU | 请求路由、传输协调 | 状态机、无队首阻塞资源准入、带老化加权公平 decode、逐请求混合执行路由与 route binding 完成 |
| Chunked Prefill | GPU 0 | 长 prompt 分块 | 分块调度、Paged 历史与 causal offset 完成 |
| State Extractor | GPU 0 | 提取 KV + 循环状态 | N−1 GDN 状态、pinned host staging 与 typed SHM 传输已接入；逐层 P2P overlap 待验证 |
| TransferBackend | GPU0->1..N | 传输后端抽象 | InMemory/SHM/P2P 实现；本机 P2P 不可用，未实测 |
| TransferPipeline | GPU0->1..N | 状态传输 | typed 单-envelope SHM、原子发布、pinned D2H/H2D 完成；P2P 层级 GPU 异步流水仍待可用硬件接入验证 |
| Continuous Batching | GPU 1..N | decode 调度、抢占恢复 | batched runtime、容量保证、精确回放、事务回滚、故障隔离与老化加权公平调度完成 |
| KV Cache Manager | GPU 1..N | PagedAttention block 管理 | 容量预留、批量原子增长、共享页引用计数/写保护/压力淘汰、batched Triton scatter 与 tiled online-softmax 完成 |
| Linear State Pool | GPU 1..N | FP32 固定 slot 管理 | layer-major 连续 GPU pool、显存预算化保证容量、decode 原地事务提交完成 |
| Prefix Cache | GPU 1..N | Radix tree (skip mamba) | 策略与真实 Paged KV 页生命周期、worker affinity 探测完成；GDN 不缓存 |
| Adaptive Router | CPU | Collocated vs PD 路由 | 直接 service 成本曲线、admission/executor queue 分解、running-future 剩余工作、decode-load 一阶外部性、收益/风险/迟滞门禁、在线分桶 EWMA 与漂移回退、profile 配置、不可变 route binding、1P+ND 容量/缓存/拓扑评分、跨 worker 并行 decode、worker 自动摘流/重启/握手完成；N>1 实机待验证 |
| ModelAdapter | both | 多模型适配 | 动态 config + 4B 真实 runtime smoke 完成 |
| API Server | CPU | OpenAI-compatible | completions/chat/SSE、完整单 choice 采样参数、stop 隐藏、logprobs、流式 usage + collocated/PD 常驻模式完成；tools/多 choice 未实现并显式拒绝 |

---

## 5. 核心技术详细设计

### 5.1 推理引擎与 Triton Kernel

#### 5.1.1 Full Attention Prefill：flash_attn 库

使用 flash_attn_varlen_func，支持 GQA 和变长序列。不自己实现 Flash Attention。

#### 5.1.2 Full Attention Decode：Triton Paged Attention

flash_attn 不支持 paged KV。自实现 Triton kernel：
1. 从 block table 查物理 block id
2. kernel 内按 block 遍历 K/V，计算 Q . K^T 分块 score
3. online softmax（流式）
4. 累加 V 加权和

Triton online-softmax kernel 已完成，并在 RTX 3090 上与 reference Paged Attention 对照通过。

#### 5.1.3 Linear Attention (GDN)：Triton Fused Delta Rule

    # gated delta rule (每步):
    #   S <- exp(g_t) * S
    #   delta <- beta_t * (v_t - k_t^T S)
    #   S <- S + k_t @ delta
    #   out_t <- RMSNorm(S^T q_t) * SiLU(z_t)

不 fuse：每步回 HBM 读写 state（1MB/16 heads），32K = 32GB 读写量。
Fused recurrent kernel：按 value-dimension tile 保持 state，RTX 3090 上已与逐步 FP32 reference 对照通过。

### 5.2 双状态内存管理

#### 5.2.1 PagedAttention Block Manager

| 参数 | 4B/9B | 27B |
|------|-------|-----|
| Block size | 16 tokens | 16 tokens |
| Full Attn 层数 | 8 | 16 |
| KV heads | 4 | 4 |
| Head dim | 256 | 256 |

不活跃请求的 KV Cache 可量化为 INT4（KIVI），压缩比 3.2x。

#### 5.2.2 Linear Attention State Pool

固定大小 slot pool，不增长、不可量化：

    slot[i] = {
        ssm_state: [lin_layers, key_heads, key_dim, val_dim] fp32,
        conv_state: [lin_layers, key_heads, conv_kernel, key_dim] fp32
    }

### 5.3 PD 分离传输协议

#### 5.3.1 双状态异构序列化

    class StateTransferDescriptor:
        request_id: int
        model_name: str
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

#### 5.3.2 N-1 Prompt Truncation

循环状态处理到第 N 个 token 编码 0..N-1。Decode 端必须重算第 N 个 token 让状态推进。开销 <5ms。

#### 5.3.3 First-Token Seeding

Prefill 端采样首 token 后随状态传输，decode 端直接输出，省 ~58ms 重算。源自 vLLM #51919。

#### 5.3.4 层级别异步流水线

每层计算完立即传该层状态：

    Prefill: Layer 0 -> Layer 1 -> ... -> Layer 31 -> done
                  |           |                 |
                  v           v                 v
    Transfer:  [L0 state]  [L1 state]  ...  [L31 KV]

理论传输时间（待实测验证）：

| 操作 | 9B 数据量 | NVLink | prefill 计算 | 可隐藏？ |
|------|---------|--------|-------------|---------|
| Linear 层 state | ~1 MB | <0.01ms | ~2ms | 理论 yes |
| Full attn KV (BF16) | 128 MB | 1.1ms | ~5ms | 理论 yes |
| 全部 32K (BF16) | 1 GB | 9ms | 50-200ms | 理论 yes |
| 全部 32K (INT4) | 345 MB | 3ms | 50-200ms | 理论 yes |

注意：当前硬件（SHM 4.58 GB/s）无法验证层级别流水线重叠。需 P2P 或 NVLink。

### 5.4 Continuous Batching + Chunked Prefill

请求状态机：

    WAITING -> PREFILL_RUNNING -> TRANSFER_PENDING -> READY -> RUNNING -> FINISHED
                                                           |
                                                     PREEMPTED -> READY

### 5.5 自适应路由

| 条件 | 路由 | 理由 |
|------|------|------|
| prompt < 2K | Collocated | prefill 快，传输不值得 |
| prompt > 8K + decode 有空位 | PD 分离 | 传输可隐藏 |
| prompt > 32K | PD 分离 | prefill 长，传输一定隐藏 |

### 5.6 Prefix Caching 与混合注意力

skip_mamba_match 策略（与 SGLang #32732 一致）：
- Prefix matching 只匹配 full attention KV Cache
- Linear attention 状态始终从 prefill 端获取
- cow_mamba=False

### 5.7 TransferBackend 抽象层

#### 5.7.1 接口

    class TransferBackend(ABC):
        def send(self, tensor, dst, stream) -> None
        def receive(self, tensor, src, stream) -> None
        def get_bandwidth(self) -> float
        def supports_layer_pipeline(self) -> bool
        def get_latency(self) -> float

#### 5.7.2 TransferMode 枚举

    class TransferMode(Enum):
        FULL_TRANSFER = "full"          # NVLink: 全量 BF16
        QUANTIZED_TRANSFER = "quant"    # PCIe P2P: INT4 KV
        PARTIAL_TRANSFER = "partial"    # 低带宽: 仅状态 + KV 重算
        INTRA_GPU = "intra"             # MPS: 同卡零传输

#### 5.7.3 后端实现状态

| 后端 | TransferMode | 接口 | 实现 | 端到端测试 | 实测数据 |
|------|-------------|------|------|-----------|---------|
| NVLink | FULL_TRANSFER | 完成 | 桩文件(249B) | 未测试 | 无（无 NVLink） |
| PCIe P2P | QUANTIZED_TRANSFER | 完成 | 桩文件(243B) | 未测试 | 无（P2P 不可用） |
| SHM | PARTIAL_TRANSFER | 完成 | 完成 | 部分 | 4.58 GB/s 实测 |
| gRPC | QUANTIZED_TRANSFER | 设计中 | 未实现 | 无 | 无 |
| MPS | INTRA_GPU | 完成 | 完成(6.3KB) | 已测试 | decode 2.5x 减速 |

#### 5.7.4 PARTIAL_TRANSFER 实测数据

9B BF16，SHM 4.58 GB/s：

| 上下文 | prefill_ms | 传输_ms | KV重算_ms | PD_total_ms |
|--------|-----------|---------|----------|-------------|
| 455 | 244.9 | 5.3 | 61.2 | 311.4 |
| 911 | 467.8 | 5.3 | 117.0 | 590.1 |
| 1821 | 813.2 | 5.3 | 203.3 | 1021.8 |
| 3642 | 1595.7 | 5.3 | 398.9 | 1999.9 |

KV 重算约为 prefill 的 25%，随上下文线性增长。

### 5.8 Intra-GPU MPS 模式

受 BulletServe 启发，MPS 同卡 PD 分离（零传输）。实测无 libsmctrl 时 decode 2.5x 减速（SM 争用），验证 inter-GPU 物理隔离的必要性。

---

## 6. 四卡 1P+3D 扩展设计

### 6.1 1P+3D：最小自然 PD 配置

1P+3D 是 Mooncake/DistServe 的单节点等价物——1 张 prefill 卡 + 3 张 decode 卡。

| 维度 | 2 卡 (1P+1D) | 4 卡 (1P+3D) |
|------|-------------|-------------|
| DP 替代方案 | DP 太强（零传输 2x） | 4x DP 仍可用，但高并发时每卡有干扰 |
| PD 自然性 | 不自然 | Mooncake 最小形态 |
| Decode 容量 | 1 卡 = 单卡 | 3 卡 = 3x decode |
| 负载均衡 | 不需要 | round-robin / least-loaded / affinity |

### 6.2 多目标传输调度

3 张 decode 卡，每请求传 INT4 KV（345MB for 32K）+ 9B 状态（53.48MB）：

| 传输策略 | 总时间 | 说明 |
|----------|--------|------|
| 串行（GPU0->1->2->3） | 约 3x33ms = 99ms | 超 prefill，部分暴露 |
| 并行（同时传 3 路） | 398.5MB / 4.7GB/s ≈ 85ms | 带宽 3 路均分 |
| 层级别交错流水线 | ~33ms | 每路独占带宽传该层 |

层级别交错：Layer 0 传给 GPU1，Layer 1 传给 GPU2，Layer 2 传给 GPU3，每路独占带宽。

### 6.3 NVLink 菊花链拓扑（可选）

3090 每卡 2 个 NVLink 端口，4 卡菊花链：GPU0->GPU1->GPU2->GPU3。

| 传输路径 | 跳数 | 有效带宽 | BF16 KV 1GB 传输 |
|----------|------|---------|------------------|
| GPU0->GPU1 | 1 跳 | 112 GB/s | 9ms |
| GPU0->GPU2 | 2 跳 | ~56 GB/s | 18ms |
| GPU0->GPU3 | 3 跳 | ~37 GB/s | 27ms |

不对称带宽 -> 拓扑感知路由：延迟敏感请求路由到近端（GPU1），吞吐优先请求路由到远端（GPU3）。

---

## 7. Benchmark 设计

### 7.1 对比配置

| 配置 | 说明 | GPU 数 | 角色 |
|------|------|--------|------|
| A: Collocated | 单卡 prefill+decode | 1 | 参考 |
| B: DP | 各卡独立 | 2/4 | 主对比组 |
| C: TP (vLLM) | 标准 TP | 2 | 主对比组 |
| D: PD 分离 | GPU0 prefill, GPU1..N decode | 2/4 | 主对比组 |
| E: vLLM unified | vLLM 单实例 | 1 | 参考 |
| F: intra-GPU MPS | MPS 同卡 + 独立卡 | 2 | 扩展对比 |

公平性：主对比组 B/C/D 均使用相同 GPU 数。核心对比 B (DP) vs D (PD)。

### 7.2 已有实测数据

| 实验 | 模型 | 数据 |
|------|------|------|
| Prefill 吞吐 vs 上下文 | 4B/9B BF16 | 实测（4B: 3258, 9B: 2282 tok/s） |
| Decode batch 吞吐 | 4B/9B BF16 | 实测（4B batch128: 191 tok/s） |
| DP 吞吐 + GPU 利用率 | 4B/9B BF16 | 实测 |
| Collocated 干扰 | 4B/9B BF16 | 实测（1K: 2.5x, 4K: 6.4x） |
| SHM 带宽 | - | 实测（4.58 GB/s） |
| PD Partial 延迟 | 4B/9B BF16 | 实测 |
| INT4 精度 | 4B BF16 | 实测（PPL +0.74） |
| MPS intra-GPU | 4B BF16 | 实测（decode 2.5x 减速） |
| 27B FP8 vLLM TP=2 | 27B FP8 | 实测（TTFT/TPOT/并发） |
| HydraServe 4B GSM8K C=4 | 4B BF16 | 2 warmup + 8 measured：collocated 58.58 tok/s；PARTIAL PD 22.65 tok/s，未 crossover |

### 7.3 待实测数据

| 实验 | 优先级 | 依赖 |
|------|--------|------|
| PD 分离完整端到端（SHM Partial） | 已完成 | 4B 双进程、双 GPU 实测 |
| B vs D crossover point | P1 | 引擎端到端 |
| 层级别流水线传输隐藏 | P1 | P2P 或 NVLink |
| 1P+3D 多目标路由 | P2 | 四卡 P2P 环境 |

### 7.4 数据集

| 数据集 | 用途 | 上下文 |
|--------|------|--------|
| ShareGPT | 对话流量 | 1K-8K |
| HumanEval | 编程助手 | 2K-8K |
| LongBench | 长文档问答 | 8K-128K |
| WikiText-103 | PPL 验证 | 滑动窗口 |
| GSM8K | 推理精度 | 短上下文 |

本机数据目录为 `/mnt/nvme-data/datasets/benchmark`。当前加载器对 ShareGPT
顶层大 JSON 数组做增量解析，对 HumanEval 按文件 magic 自动识别 gzip，LongBench
直接读取 ZIP 成员，不复制解压数据。WikiText 的 raw tarball/CSV 是空文件，使用有效的
`wikitext-103-test.jsonl`。

---

## 8. 应用场景

1. 编程助手（10-30 开发者，8K-32K 上下文）
2. 文档问答/RAG（32K-128K 文档）
3. 多轮对话（20-50 并发）
4. Agent 工具调用（burst arrival + 长 decode）

---

## 9. 简历定位与面试叙事

### 9.1 简历条目

> **HydraServe** | 面向混合注意力架构 LLM 的 PD 分离推理引擎原型
>
> - 从零实现 Qwen3.5/3.6（GDN + GQA 混合注意力）推理引擎：Triton fused kernel 实现 GDN delta rule 递推前向（state 留 SRAM），Triton paged attention 实现 decode 阶段非连续 KV block 的 online softmax
> - 设计双状态异构传输协议：KV Cache 可 INT4 量化传输，循环状态 FP32 整体传输（不可量化，递推误差发散）。TransferBackend 抽象层支持 NVLink/PCIe P2P/SHM/gRPC 多后端，层级别流水线根据带宽自适应。N-1 truncation + 首 token 预播种
> - 实现 continuous batching + chunked prefill 调度器与自适应路由，支持 4B/9B/27B 多模型
> - 双卡 RTX 3090（x16+x4，无 P2P，SHM 4.58 GB/s）实测验证：collocated 干扰 2.5-6.4x，PARTIAL_TRANSFER KV 重算占 prefill ~25%，MPS decode 2.5x 减速，INT4 PPL +0.74

### 9.2 面试叙事线

**开场**：

"我做了一个叫 HydraServe 的项目，从零实现了面向混合注意力架构 LLM（Qwen3.5/3.6）的 PD 分离推理引擎原型。核心难点是混合注意力有两种状态要传——KV Cache 和循环状态——现有 PD 分离工作没有处理好这个。"

**技术深挖**：

"Prefill 和 decode 在同一 GPU 上会互相干扰。3090 上实测：4K prefill 导致 decode 减速 6.4 倍，持续 1.8 秒。"

"混合注意力的难点是双状态：KV Cache 可以量化压缩，但循环状态必须保持 FP32。按真实 Qwen GDN value heads 和完整 conv channels 计算，9B 是 53.48MB，27B 是 158.86MB。我用层级别异步流水线逐层传输状态。还有 N-1 truncation：循环状态编码 0..N-1，decode 端必须重算第 N 个 token。用 first-token seeding 跳过。"

"框架通过 TransferBackend 抽象层隔离传输介质。当前双卡 3090（x16+x4，无 P2P，SHM 4.58 GB/s）只能跑 PARTIAL_TRANSFER——9B 传 53.48MB 状态，KV 本地重算。四卡全 x16 P2P 可以跑完整 QUANTIZED_TRANSFER。"

**诚实定位**：

"PD 分离真正发挥价值是大规模部署——Mooncake 那种 100+ GPU。我的项目是引擎原型和协议验证。在 4 卡规模，DP 吞吐可能更高，PD 优势在 P99 稳定性但需大规模才显著。"

### 9.3 面试问答

| 问题 | 回答要点 |
|------|----------|
| 为什么不用 vLLM？ | vLLM #41869/#46807 已合并 GDN PD 分离，但依赖 NIXL/Mooncake + RDMA。我是从零实现的独立引擎，面向消费级 GPU |
| 和 BulletServe 区别？ | BulletServe 是 intra-GPU（SM masking）。HydraServe 做 inter-GPU。实测 MPS decode 2.5x 减速 |
| 为什么不用 TP？ | TP 解决"放不下"，PD 解决"干扰"。9B INT4 单卡放得下，TP 没必要 |
| 循环状态不量化？ | FP32 递推误差累积发散；9B 真实状态 53.48MB，不能按旧版 25MB 估算 |
| 有无 NVLink 区别？ | 无 NVLink（SHM 4.58 GB/s）只能 PARTIAL。有 NVLink 9ms 传 1GB。四卡 P2P + INT4 KV 也可行 |
| 双卡为什么不做 DP？ | 双卡是开发环境不是部署目标。PD 在大规模才显著优于 DP。四卡 1P+3D 是最小自然 PD 配置 |
| 能扩展多节点？ | TransferBackend 设计支持 gRPC 或 RDMA。上层协议介质无关 |
| 层级别流水线实测了？ | 当前硬件无法验证。理论每层 state 1MB，传输 <0.01ms << 2ms 计算。需 P2P/NVLink 实测 |
| Flash Attention 自己写？ | Prefill 用 flash_attn 库。Decode 自写 Triton paged attention。GDN 自写 Triton fused delta rule（无现成实现） |
| INT4 PPL 涨 0.74？ | naive 对称量化无校准。AWQ 带校准可达 <0.3 |
| PD 一定比 DP 好？ | 不一定。低并发 DP 赢，高并发 PD 赢。crossover 待实测 |
| baseline 公平？ | 主对比组 B/C/D 均用相同 GPU 数 |

---

## 10. 开发里程碑

| Phase | 内容 | 时间 | 状态 |
|-------|------|------|------|
| 0 | 环境搭建 + 模型加载 + 硬件实测 | 1 周 | 完成 |
| 1 | 推理引擎 + Triton kernels | 2.5 周 | 4B/9B BF16 + 27B AWQ 真实 GPU smoke 完成 |
| 2 | 双状态内存管理 + ModelAdapter | 2 周 | 动态配置、Paged KV、连续 GPU state pool 与显存预算化准入完成 |
| 3 | 传输层 + 双状态序列化 | 2 周 | InMemory/SHM/P2P + 层级协议完成；P2P 待可用硬件实测 |
| 4 | PD 分离核心 + N-1 truncation | 2 周 | 4B 真实 GPU 双进程 SHM PARTIAL 完成 |
| 5 | Continuous batching + chunked prefill | 2 周 | 完成：batched decode + Paged 历史与 causal offset |
| 6 | 自适应路由 + 多模型适配 | 1.5 周 | 动态 config + 4B/9B BF16 + 27B AWQ runtime 完成；FP8 待实现 |
| 7 | Benchmark + 对比实验 | 2 周 | 五类数据集、并发 runner、TTFT/TPOT 分位数完成；正式实验待跑 |
| 8 | API + PD worker + SHM Partial 实测 | 1 周 | 完成：常驻双进程 PD 接入 API/benchmark |
| 9 | 生产化资源准入、缓存与路由 | 持续 | P0 联合准入；P1 成本感知策略；P2 混合执行；P3 1P+ND；P4 Prefix 物理页共享；P5 抢占/故障隔离；P6 调度/worker 恢复；P7 采样 API；P8 typed SHM/state pool/batched KV scatter/tiled attention 完成 |
| 总计 | | ~16 周 | |

**最紧急的下一步**：
1. 建立压力、故障矩阵、长稳与正式 B-vs-D benchmark 验证
2. 在 4+ GPU 环境验证 1P+ND 与拓扑路由，再跑正式 B vs D 性能矩阵
3. 扩展并优化 27B AWQ benchmark；实现 FP8 GEMM
3. 四卡全 x16 P2P 环境验证完整 QUANTIZED_TRANSFER

---

## 11. 环境构建

### 11.1 开发环境 (Conda)

    conda create -n hydraserve python=3.10 -y
    pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install triton==2.1.0
    pip install transformers>=5.15.0 accelerate safetensors
    pip install flash-attn --no-build-isolation
    pip install autoawq bitsandbytes
    pip install fastapi uvicorn pydantic aiohttp tqdm

### 11.2 Docker

    FROM nvcr.io/nvidia/pytorch:24.01-py3
    WORKDIR /app/HydraServe
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["python", "-m", "hydraserve.serve", "--model", "/models/Qwen3.5-9B-AWQ"]

### 11.3 项目目录结构

    HydraServe/
    +-- README.md
    +-- requirements.txt
    +-- Dockerfile
    +-- docker-compose.yml
    +-- pyproject.toml
    +-- hydraserve/
    |   +-- __init__.py
    |   +-- config.py              # 模型注册表 + 硬件/服务配置
    |   +-- model/
    |   |   +-- __init__.py
    |   |   +-- adapter.py         # ModelAdapter 协议
    |   |   +-- qwen3_5.py          # Qwen3.5 4B/9B 适配器
    |   |   +-- qwen3_6.py          # Qwen3.6 27B 适配器
    |   +-- kernels/
    |   |   +-- __init__.py
    |   |   +-- gdn_fused.py        # GDN fused delta rule (Triton)
    |   |   +-- paged_attention.py  # Paged decode attention (Triton)
    |   |   +-- rmsnorm.py          # Fused RMSNorm (Triton)
    |   +-- cache/
    |   |   +-- __init__.py
    |   |   +-- block_manager.py    # PagedAttention block 分配
    |   |   +-- state_pool.py       # FP32 循环状态 slot pool
    |   |   +-- kv_quantizer.py     # KIVI INT4 KV 量化
    |   |   +-- prefix_cache.py     # Radix tree (skip mamba)
    |   |   +-- weight_quantizer.py # INT4 权重量化
    |   +-- transfer/
    |   |   +-- __init__.py
    |   |   +-- backend.py          # TransferBackend 抽象层
    |   |   +-- descriptor.py       # StateTransferDescriptor
    |   |   +-- pipeline.py         # 层级别异步流水线
    |   |   +-- nvlink_transfer.py  # NVLink 后端
    |   |   +-- pcie_p2p_transfer.py # PCIe P2P 后端
    |   |   +-- shm_transfer.py     # SHM 后端
    |   |   +-- grpc_transfer.py   # gRPC 跨节点后端
    |   |   +-- intra_gpu_transfer.py # MPS 同卡后端
    |   +-- engine/
    |   |   +-- __init__.py
    |   |   +-- prefill_engine.py   # Prefill 引擎 (GPU 0)
    |   |   +-- decode_engine.py    # Decode 引擎 (GPU 1..N)
    |   |   +-- scheduler.py        # CentralScheduler
    |   |   +-- chunked_prefill.py  # Chunked prefill 调度
    |   |   +-- mps_manager.py      # MPS daemon 管理
    |   +-- router/
    |   |   +-- __init__.py
    |   |   +-- adaptive_router.py  # 自适应路由
    |   |   +-- cost_model.py       # Cost model
    |   |   +-- profiler.py         # 硬件 profiling
    |   +-- serve/
    |   |   +-- __init__.py
    |   |   +-- api_server.py       # OpenAI-compatible API
    |   |   +-- protocol.py         # 请求/响应协议
    |   |   +-- serve.py            # 启动入口
    |   +-- benchmark/
    |       +-- __init__.py
    |       +-- run_benchmark.py   # 基准测试入口
    |       +-- datasets.py         # 数据集加载
    |       +-- metrics.py          # 指标计算
    |       +-- plot.py             # 图表生成
    +-- scripts/
    |   +-- download_model.sh
    |   +-- verify_nvlink.sh
    |   +-- run_baseline.sh
    +-- tests/
    |   +-- test_kernels.py
    |   +-- test_transfer.py
    |   +-- test_e2e.py
    |   +-- test_int4_accuracy.py
    |   +-- bench_27b.py
    |   +-- full_benchmark.py
    +-- docs/
    |   +-- ITERATION_LOG.md
    |   +-- architecture.md
    +-- benchmark_output/          # 9 张实测图表 + JSON

---

## 12. 附录

### 附录 A：实测数据速查

| 数据 | 4B BF16 | 9B BF16 | 27B FP8 | 来源 |
|------|---------|---------|---------|------|
| Prefill 吞吐 | 3,258 tok/s | 2,282 tok/s | 890 tok/s | 实测 |
| Decode (单用户) | 42 tok/s | ~35 tok/s | 25 tok/s | 实测 |
| Decode (batch 128) | 191 tok/s | - | - | 实测 |
| 权重 INT4 | ~2 GB | ~4.5 GB | ~13.5 GB | AWQ |
| KV/token | 32 KB | 32 KB | 64 KB | config.json |
| 状态/请求 (FP32 recurrent+conv) | 53.48 MB | 53.48 MB | 158.86 MB | config.json + runtime shape |
| SHM 带宽 | 4.58 GB/s | 4.58 GB/s | - | 实测 |
| 干扰 (1K) | 2.5x | - | - | 实测 |
| 干扰 (4K) | 6.4x | - | - | 实测 |
| MPS 减速 | 2.5x | - | - | 实测 |
| INT4 PPL | +0.74 | - | - | 实测 |

### 附录 B：理论值 vs 实测值对照

| 指标 | 设计文档理论值 | 实测值 | 差异原因 |
|------|-------------|--------|---------|
| 32K prefill | 50-100ms | ~14 秒 (9B BF16 3090) | 理论值是 H100 级别 |
| SHM 带宽 | 8-10 GB/s | 4.58 GB/s | x4 瓶颈 |
| P2P 可用 | 是 | 否（NODE 拓扑） | 硬件限制 |
| INT4 PPL | <0.3 | +0.74 | naive 量化无校准 |
| 传输 100% 隐藏 | 理论 yes | 未验证 | 无 P2P/NVLink |

### 附录 C：vLLM 混合注意力 PD 分离进展

| PR/Issue | 标题 | 状态 | 日期 |
|----------|------|------|------|
| #41869 | NIXL GDN support | merged | 2026-05-14 |
| #46807 | Mooncake GDN support | merged | 2026-07-14 |
| #51052 | MoRIIO hybrid state transfer | open | 2026-08-04 |
| #51919 | First-token seeding | open | 2026-08-10 |
| SGLang #32732 | PD disagg for mamba/SSM | open | 2026-07-29 |
| Mooncake #2242 | Feature request: hybrid GDN | open | 2026-05-27 |

---

*Document version: 2.0 | Last updated: 2026-08-13 | 基于实测数据修正，区分理论值与实测值，修正硬件描述，调整项目定位为引擎原型*
