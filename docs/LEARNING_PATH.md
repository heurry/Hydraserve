# HydraServe 学习文档(自包含)

> 本文件把简历描述背后的知识直接写全:每章先讲清原理,再指向对应代码,最后给出
> 动手练习。链接只作为"想再深挖时"的延伸,不影响本文件独立阅读。

## 目录

1. [GPU 与并行计算基础](#1-gpu-与并行计算基础)
2. [混合注意力架构:从 softmax attention 到 Gated DeltaNet](#2-混合注意力架构)
3. [online softmax 与 Paged Attention](#3-online-softmax-与-paged-attention)
4. [Triton 编程模型与自研算子逐个拆解](#4-triton-编程模型与自研算子逐个拆解)
5. [量化:INT4 AWQ 与 FP8 E4M3FN](#5-量化int4-awq-与-fp8-e4m3fn)
6. [Paged KV 与 recurrent state 管理](#6-paged-kv-与-recurrent-state-管理)
7. [双状态传输协议](#7-双状态传输协议)
8. [Continuous Batching 与调度容错](#8-continuous-batching-与调度容错)
9. [1P+ND 与校准式自适应路由](#9-1pnd-与校准式自适应路由)
10. [基准方法学](#10-基准方法学)
11. [面试自测题](#11-面试自测题)
12. [仓库阅读顺序与术语表](#12-仓库阅读顺序与术语表)

---

## 1. GPU 与并行计算基础

### 1.1 GPU 为什么快

GPU 是**吞吐优先**的众核架构,与 CPU 的"低延迟单线程"思路相反:

- **SIMT**:一个 warp(32 线程)同一条指令作用于 32 份数据;分支发散时两边都执行,收敛时再汇合;
- **延迟隐藏**:一个 warp 等显存时,调度器立刻切换其他 warp,靠大量并行掩盖几百周期的显存延迟;
- **显存层级**(从慢到快、容量从大到小):HBM 全局显存(GB 级,~1 TB/s)→ L2(几十 MB)→ 每 SM 的 shared memory / SRAM(数百 KB,~20 TB/s)→ 寄存器(每 SM 256 KB)。

### 1.2 关键概念速查

| 概念 | 含义 | 在 HydraServe 中的位置 |
|------|------|----------------------|
| Tensor Core | 专做 4×4(或更大)矩阵乘累加的硬件单元 | `tl.dot` 落到 Tensor Core;FP8/INT4 无法原生使用时用 BF16 dot |
| pinned memory | 页锁定主机内存,可被 GPU DMA 直接读写 | SHM 传输的 host staging([transfer/backend.py](../hydraserve/transfer/backend.py)) |
| CUDA stream | 异步执行队列,流间可并行 | 传输与计算重叠的基础 |
| CUDA P2P | 两 GPU 经 PCIe/NVLink 直接互访显存,不经过主机 | `cudaDeviceCanAccessPeer` 能力检测 |
| occupancy | SM 上同时驻留的 warp 占比 | 决定能否隐藏延迟 |

### 1.3 算力密集 vs 带宽密集——本项目一切的起点

把 9B 模型的一层前向拆成两类工作:

- **prefill**(一次处理整个 prompt):每读一字节权重做多次乘加 → **算力密集**,GPU 利用率 50-90%;
- **decode**(一次生成一个 token):batch 很小时,每个 token 都要把全部权重从显存读一遍 → **带宽密集**,利用率 5-20%。

实测(3090,9B BF16):prefill 2,282 tok/s,decode 35 tok/s。二者在同一张卡上交替执行时互相抢占:1K prompt 让 decode 慢 2.5×,4K 慢 6.4×。这就是简历里
"prefill 干扰 decode 2.5–6.4×"的完整含义——也是 PD 分离的动机。

**动手**:算一笔账——9B 模型 18 GB BF16 权重,decode 单 token 需读 18 GB,除以
HBM 带宽(~936 GB/s),得到单 token 理论下限;batch=32 时同一批权重被 32 个
token 共享,带宽利用率如何变化?

---

## 2. 混合注意力架构

### 2.1 softmax attention 与 KV Cache

标准 attention:给定 Q,K,V ∈ R^{T×d},

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

- 计算量 O(T²·d),显存 O(T²);因果掩码让每个位置只看过去;
- 生成阶段每个 token 都要回看全部历史 → 把历史的 K/V 缓存起来就是 **KV Cache**,历史计算不重做;
- 但 KV 随序列长度线性增长:32K context、9B 模型 ≈ 1 GB(BF16)。

### 2.2 GQA(Grouped Query Attention)

KV Cache 的压缩手段之一:**多个 query head 共享一组 KV head**。若 query heads
= 32、kv heads = 8,则每组 4 个 query head 复用同一 K/V,KV 体积 ÷4。
[config.py](../hydraserve/config.py) 的 `num_attention_heads` / `num_kv_heads`
描述的就是这对数字。

### 2.3 线性注意力谱系的思想

softmax attention 是 O(T²);线性注意力把 softmax 核函数分解为特征内积,
使注意力可写成**外积状态**的递推:

$$S_t = S_{t-1} + k_t v_t^\top, \qquad y_t = q_t^\top S_t$$

状态 S 是固定大小的矩阵(与 T 无关)→ O(1) 每步。这条思想线:
RetNet → RWKV → H3 → Mamba → Mamba-2 → DeltaNet → **Gated DeltaNet**。
Qwen3.5/3.6 用 GDN 层替代大部分 attention 层(如 4B:32 层中 24 层 GDN + 8 层 full attention)。

### 2.4 Gated DeltaNet 完整数学(逐行对照代码)

每层输入 hidden(B×T×D)→ 线性投影出 q, k, v, log_decay, beta, gate:

| 参数 | 形状 | 来源 | 约束与作用 |
|------|------|------|-----------|
| q / k | B×T×H×K_d | hidden 投影 | 先 L2 归一化再加权——保证每步写入状态的范数有界,递推不发散(DeltaNet 的核心技巧) |
| v | B×T×H×V_d | hidden 投影 | 写入记忆的内容 |
| log_decay | B×T×H | 投影,**存对数** | exp(log_decay) ∈ (0,1),是"旧状态保留比例";存对数保证符号与值域天然正确 |
| beta | B×T×H | 投影,sigmoid 到 (0,1) | delta 写入强度的门控:接近 0 时本步几乎不更新状态 |
| gate | B×T×H×D(或逐头) | hidden 投影,SiLU | 输出门控:y ← y·silu(gate),控制本步"读出"多少 |

对每个 token t:

1. **L2 归一化**:q ← q/‖q‖·K^{-1/2},k ← k/‖k‖(K = key dim);
2. **因果 depthwise conv + SiLU**:对 q/k/v 各通道做核宽 K 的一维因果卷积再 SiLU
   (公式见 §4.2,等价于每个 token 只看过去 K 个输入,提供局部时序信息);
3. **delta rule 递推**(状态 S ∈ R^{K_d × V_d},FP32):

$$S_t = \underbrace{\exp(\text{log\_decay}_t)\cdot S_{t-1}}_{\text{衰减旧记忆}} + \underbrace{\beta_t\, k_t \,(v_t - \underbrace{S_{t-1} k_t}_{\text{当前预测}})^\top}_{\text{按残差写入}}$$

4. **读取**:y_t = q_t^\top S_t;再与 gate 相乘、接输出投影 + 残差。

三步语义:先**衰减**旧状态 → 用 key **检索**当前预测 S·k → v 与预测的**残差**
(delta)按 β 门控写回状态。残差小说明记忆已能预测当前输入,状态更新自然稀疏
——这是 delta rule 优于纯外积累加的地方。

精确实现见 [kernels/reference.py](../hydraserve/kernels/reference.py) 的
`gated_delta_rule`(每行都写着上面的公式,是最好的教科书)。

### 2.5 chunk 递推与 SRAM 驻留

- prefill 时把 T 个 token 的递推放进**一个 kernel 里循环**,状态矩阵全程留在
  SRAM/寄存器,只在进入和退出时各读写一次显存——这是 [kernels/gdn.py](../hydraserve/kernels/gdn.py)
  的核心,也是简历"循环状态驻留 SRAM 避免显存往返"的出处;
- chunked prefill 切分 prompt 时,GDN 状态在 chunk 边界落显存、下一 chunk 从
  状态继续(conv state 同样携带)——`gated_delta_recurrent(query, ..., state)`
  的 `state` 参数就是跨 chunk 的载体。

### 2.6 为什么"双状态异构":KV 可量化、recurrent state 不可

| 状态 | 大小(32K) | 能否量化 | 原因 |
|------|-----------|---------|------|
| full-attention KV | 9B:1 GB BF16 / 345 MB INT4 | 能 | 每 token 独立存储,量化误差不跨步累积(实测 naive INT4 对称量化 PPL +0.74,带校准 <0.3) |
| GDN 循环状态 | 4B/9B:53.48 MB;27B:158.86 MB | **不能** | FP32 递推每步都读旧状态,量化误差在 T 步内**累积发散** |

因此传输协议必须区别对待两类状态:KV 可压缩传输,循环状态只能 FP32 整体传输。
这是整个 transfer 层的设计原点。

**动手**:用 NumPy 实现 §2.4 的 delta rule(10 行),分别用 FP32 与 int8 存状态
跑 1000 步,观察误差发散;再对照 `tests/test_reference_kernels.py`。

---

## 3. online softmax 与 Paged Attention

### 3.1 朴素 softmax 的问题与 online 递推(完整推导)

**第一层:为什么要减 max。** softmax 直接算 exp(x_i)/Σexp(x_j) 会溢出(FP16 上限
65504,exp(1000) 就爆)。但分子分母同时除以一个常数结果不变:

$$\text{softmax}_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

取 m = max(x) 时所有 exp 参数 ≤0,值域 (0,1],永不溢出——这就是"减 max"。

**第二层:流式难题。** paged attention 的 KV 在非连续物理块里,必须逐块扫描、
扫完即丢。扫完若干块后手里只攒三个量(m_old、l_old、acc):

| 符号 | 含义 | 形状 |
|------|------|------|
| m_old | 已见分数的最大值 | 标量 |
| l_old = Σ exp(x_j − m_old) | 旧分母(按旧 max 缩放) | 标量 |
| acc = Σ exp(x_j − m_old)·v_j | 旧分子:分数加权的 V 之和 | head_dim 向量 |

新 tile 到了,新 max `m_new ≥ m_old`。**旧累加量按旧 max 缩放,不能与新 tile
按新 max 缩放的值直接相加**——基准不一致。

**第三层:换基准的恒等式。** 对任意旧分数 x_j:

$$e^{x_j - m_{new}} = e^{x_j - m_{old}} \cdot e^{m_{old} - m_{new}}$$

于是把旧累加量整体乘上修正因子 `exp(m_old − m_new)` 即可换到新基准:

```
m_new = max(m_old, m_tile)
l_new = l_old·exp(m_old−m_new) + Σ exp(x_tile − m_new)
acc   = acc·exp(m_old−m_new)   + Σ exp(x_tile − m_new)·v_tile
```

三个关键性质:①**修正因子永远 ≤1**(m_new ≥ m_old,exp 参数 ≤0)——旧贡献只会被
按正确比例降权,数值上从不放大误差;②m_new == m_old 时因子 = 1,旧累加量原样
保留(常见情形);③最后输出 `acc / l` 时 max 彻底消掉——**实数意义下与一次性
算出全部分数完全等价,不是近似**。可以理解为流式加权均值:维护的不变量是
`acc/l` 始终等于"已见所有分数的 softmax 加权 V 之和"。

**映射到代码**([kernels/paged_attention.py](../hydraserve/kernels/paged_attention.py)
的 decode kernel):

```python
maximum = -inf; denominator = 0.0; accumulator = zeros
for each 16-token tile:
    score = q·k_tile / √d; score = where(valid, score, -inf)   # exp(-inf)=0 → 无效位置零贡献
    next_maximum = max(maximum, tile_max)
    old_scale = exp(maximum - next_maximum)                    # ← 恒等式因子
    probability = where(valid, exp(score - next_maximum), 0)
    accumulator = accumulator*old_scale + Σ probability·v      # ← acc 递推
    denominator = denominator*old_scale + Σ probability        # ← l 递推
    maximum = next_maximum
result = accumulator / denominator
```

第一块时 m_old=−inf,exp(−inf−m_new)=0,零累加量乘 0 仍为 0,递推自洽。

它是 FlashAttention 分块递推的核心,也是 [kernels/paged_attention.py](../hydraserve/kernels/paged_attention.py)
里 decode kernel 的骨架。

### 3.2 Paged Attention:块表寻址

decode 时 KV 不在连续显存里,而在**物理块**中:逻辑位置 (request, token) →
物理地址 = `block_table[request, token // BLOCK_SIZE]` 的块内偏移
`token % BLOCK_SIZE`。kernel 对 16 token(BLOCK_T=16)为一 tile 扫描,逐 tile
查表取物理块、算分、online-softmax 合并。grid = (batch, query_heads),每个
program 负责一个 (请求, query head) 的输出;GQA 下 `kv_head = q_head // group_size`。

### 3.3 首 chunk 与 continuation chunk 的两条路径

chunked prefill 时:首 chunk 没有历史,可用 FlashAttention varlen(可选的
`flash-attn` 依赖,[kernels/flash_prefill.py](../hydraserve/kernels/flash_prefill.py));
continuation chunk 需要读物理页历史,走自写 Triton Paged online-softmax
([paged_prefill_attention](../hydraserve/kernels/paged_attention.py))——每个
(query token, 长度) 对是一个独立 program,天然支持异构上下文长度。

**动手**:手推 §3.1 的递推式(从两段拼接推起),解释为什么 `exp(m_old − m_new)`
必然 ≤ 1(数值安全)。

---

## 4. Triton 编程模型与自研算子逐个拆解

### 4.1 Triton 心智模型

- 你写的是**一个 program**(处理一块数据的代码),`program_id` 是块坐标,grid 决定铺多少块;
- `tl.arange` 造块内索引、`tl.load/store` 带 mask 处理越界、`tl.constexpr` 让块大小编译期特化;
- `tl.dot` 落到 Tensor Core;循环用 `tl.range` 供编译器调度。

### 4.2 逐个 kernel(建议顺序)

#### 4.2.1 RMSNorm([kernels/rmsnorm.py](../hydraserve/kernels/rmsnorm.py),数学 oracle 在 [reference.py:24-32](../hydraserve/kernels/reference.py#L24-L32))

$$y = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=1}^D x_i^2 + \varepsilon}} \odot w$$

参数逐项解释:

| 参数 | 形状 | 含义 |
|------|------|------|
| x | [rows, D] | 输入 hidden(D = hidden_size,4096/2560/5120) |
| w(weight/scale) | [D] | **可学习向量**,逐维缩放:第 i 维的输出被放大/缩小 w_i 倍 |
| eps | 标量 | 1e-6 小常数,防止整行为零时除零(√0 会 inf) |
| √mean(x²) | 标量/行 | RMS(Root Mean Square)——注意**不减去均值**,这是与 LayerNorm 的关键区别 |

**为什么用 RMS 而不是 LayerNorm 的方差**:不减去均值少一次规约(省算力),而对
transformer 而言均值信息在残差连接里仍然保留,后续权重矩阵能吸收这个偏移,效果
几乎无差(LLaMA 系的共同选择)。

**zero-centered 变体(默认,`zero_centered=True`)**:checkpoint 里存的 w' 是
**增量**而不是缩放本身,运行时 `scale = 1 + w'`。动机:训练端 w' 初始化 0 → 层
初始化为恒等映射 y=x,残差结构收敛更稳——与 LayerNorm 权重初始化为 1 是同一个
思想,只是参数化方式不同(存增量)。gated 版调用时显式用 `zero_centered=False`
(见代码),即 post-attention 路径的语义由 checkpoint 决定。

**实现要点**:整行 x 先 `.float()` 升到 FP32 再平方求和(BF16 平方精度不足)、
`rsqrt` 一次性算出 1/√(·)、乘 w、最后 cast 回原 dtype;每行一个 program,
`BLOCK = next_power_of_2(D)`,num_warps 随宽度调(≥4096 用 8)。

**gated 变体**:`y = RMSNorm(x, w, zero_centered=False) ⊙ SiLU(gate)`——gate 与
x 同形状(另一路投影),SiLU(gate) 是逐元素门控(§2.4 的 gate 同一思想)。
把 RMSNorm 与门控**融合进同一个 kernel**:x 只读写一次显存,省掉中间张量往返。

#### 4.2.2 causal depthwise conv([kernels/gdn.py](../hydraserve/kernels/gdn.py) 的 `_causal_conv_kernel`)

$$\text{out}[t,c] = \text{SiLU}\!\left(\sum_{j=0}^{K-1} w[c,j]\cdot \text{in}[t-K+1+j,\,c]\right)$$

| 参数 | 形状 | 含义 |
|------|------|------|
| x(in) | [B, T, C] | 输入(C = channels) |
| w | [C, K] | 每通道**独立**的 K 长卷积核(K = linear_conv_kernel_dim,4B 为 4) |
| state | [B, C, K] | 最近 K 个输入的历史,跨 chunk 的载体 |

- **因果性**:token t 只看 `[t−K+1, t]` 窗口;窗口越界的部分(t−K+1+j < 0)
  从携带的 state 里读——kernel 里用 `tl.where(from_input, value, prior)` 实现;
- **为什么需要它**:线性注意力/GDN 没有位置编码,conv 提供局部时序平滑与近邻
  信息(与 Mamba 同思路);SiLU 在 conv 之后激活;
- 每个 chunk 结束后有第二个小 kernel(`_causal_conv_state_kernel`)把最后 K 个
  输入写回 state,供下一 chunk 使用。

#### 4.2.3 GDN 递推([kernels/gdn.py](../hydraserve/kernels/gdn.py) 的 `_gdn_recurrent_kernel`)

数学见 §2.4。kernel 视角的要点:grid=(batch, heads, V_d/16),每个 program 负责
一个 (batch, head, value_block);状态块 `[BLOCK_K, BLOCK_V]`(BLOCK_K =
next_pow2(key_dim),BLOCK_V=16)载入寄存器后,**整个 token 循环都在片上执行**,
循环结束才把状态写回显存——循环状态全程驻留 SRAM/寄存器,只在进出时各一次
显存往返(简历"循环状态驻留 SRAM"的出处)。

#### 4.2.4 batched KV scatter([kernels/kv_cache.py](../hydraserve/kernels/kv_cache.py))

每个请求一个 program,把其 prefill 产出的 K/V 写入自己的物理页(逻辑位置 →
块表寻址,§3.2)。**批量化** = 一次 launch 覆盖整批请求,避免 batch 次 kernel
启动与元数据开销:实测比逐请求 launch 快 1.15×/10.25×/42.79×(batch 1/8/32,
微基准)。

#### 4.2.5 Paged decode attention / AWQ / FP8

Paged decode attention 见 §3;AWQ INT4 GEMM 与 FP8 GEMM 见 §5。

### 4.3 通用优化手法小结

1. **融合**:归一化 + 门控、conv + SiLU、反量化 + GEMM,都省中间显存往返;
2. **复用**:GDN 状态片上驻留、RMSNorm 权重整批复用;
3. **批量化**:单次 launch 处理整批,摊薄 kernel 启动与元数据开销(§6.4 的
   连续页表上传同理:页表先在 host 打包,再两个 tensor 上传)。

**动手**:Triton 官方教程 01-06(vector add → fused softmax → tiled GEMM)手敲;
然后把 GEMM 改成 INT4 packed 输入,与 [kernels/awq.py](../hydraserve/kernels/awq.py) 对照。

---

## 5. 量化:INT4 AWQ 与 FP8 E4M3FN

### 5.1 统一视角:权重 = 整数值 × 缩放

$$w[i,j] \approx \big(q[i,j] - z\big[i,\lfloor j/g\rfloor\big]\big)\cdot s\big[i,\lfloor j/g\rfloor\big]$$

参数逐项解释:

| 参数 | 含义 |
|------|------|
| q | 量化后的整数(INT4:0~15) |
| s(scale) | 每组的缩放因子(浮点)——把整数还原到真实量级 |
| z(zero-point) | **整数 0 对应的原值偏移**。非对称量化:原值域不一定对称,用 z 平移让 0 无损;对称量化 z=0(值域中心就是 0) |
| g(group_size) | 共享同一对 (s, z) 的输入通道数。g 越小精度越高、缩放参数越多;128 是精度/开销的常用平衡点 |

| 类型 | 存储 | 还原公式 | 粒度 |
|------|------|---------|------|
| 对称 INT4 | 4 bit | w ≈ q·s | 每组共享 s |
| 非对称 INT4(AWQ) | 4 bit + zero-point | w ≈ (q − z)·s | 组粒度(本仓库 g=128) |
| FP8 E4M3FN | 1+4+3 bit | 见 §5.3 | 128×128 block-wise scale |

以 g=128 为例:权重矩阵的每 128 个输入通道共享一个 scale 和一个 zero-point——
一个 4096×2560 的矩阵需要 4096×(2560/128)= 81,920 个 (s,z) 对,参数开销
≈ 量化后权重的 6%,换取与 BF16 接近的精度。

### 5.2 AWQ kernel 拆解([kernels/awq.py](../hydraserve/kernels/awq.py))

- 存储:两个 int4 拼一个字节;4 个字节(8 个 int4)拼一个 int32 word;
- kernel 中:BLOCK_M=16, BLOCK_N=32, BLOCK_K=32,4 warps;
  `quantized = (packed_word >> ((i % 8) * 4)) & 0xF` 逐 nibble 提取;
- 每 128 输入通道读一次 scale 与 zero-point,即时 `(q − z)·s` 还原为 BF16,
  再 `tl.dot`(BF16 输入、FP32 累加)——
  **反量化在 GEMM 内即时完成,不物化完整反量化矩阵**(27B 因此能在 22 GiB 内完成 64 层);
- AWQ 思想本身(activation-aware weight quantization,按激活重要度保护权重)见
  论文 2306.00978;本仓库只用其压缩格式,加载 `compressed-tensors` checkpoint。

### 5.3 FP8 E4M3FN 位模式与手动解码

E4M3FN = 1 符号位 + 4 指数位 + 3 尾数位,指数偏置 7,FN = 只有有限值(无 inf/nan,
±448 封顶)。解码规则(对照 [kernels/fp8.py](../hydraserve/kernels/fp8.py)):

```
exp = (bits >> 3) & 0xF, mantissa = bits & 0x7
exp == 0(次正规): value = mantissa × 2⁻⁹
否则:             value = (1 + mantissa/8) × 2^(exp−7)
符号位(bits & 0x80)决定正负
```

**为什么手动解码**:RTX 3090(SM86)没有原生 FP8 Tensor Core。kernel 把
`float8_e4m3fn` 视作 `uint8` 读出位模式、按上式还原成 BF16、乘 128×128 块的
inverse scale、BF16 dot 累加——在旧卡上不展开常驻 BF16 权重就能跑 FP8 checkpoint
(27B FP8 完整 64 层已实跑)。block-wise scaling 的思想见 DeepSeek-V3 技术报告
(2412.19437):细粒度 scale 用 1/128² 的参数开销换取与 BF16 相当的精度。

### 5.4 面试常问

- "INT4 反量化为什么即时做?" → 显存(§5.2);
- "group=128 怎么定?" → 精度/开销平衡;
- "FP8 为什么用 128×128 块缩放?" → 同上,且与 Tensor Core tile 对齐;
- "FP32 状态能不能也 block-scale?" → 不能,误差跨时间步累积(§2.6)。

---

## 6. Paged KV 与 recurrent state 管理

### 6.1 Paged KV([cache/paged_kv.py](../hydraserve/cache/paged_kv.py) + [cache/block_manager.py](../hydraserve/cache/block_manager.py))

思想与操作系统虚拟内存相同:KV 按固定大小块(默认 16 token)分配,**逻辑序列 → 物理块表**。
收益:消除逐请求预分配导致的碎片;多请求共享块成为可能。

- **引用计数**:每个物理块记录被多少请求/前缀引用,归零才回 freelist;
- **safety margin(headroom)**:`--kv-headroom-blocks` 从可分配容量中永久保留
  一部分页,防止工作集逼近最后一页时反复准入/失败([memory_planner.py](../hydraserve/cache/memory_planner.py) 按实际空闲显存规划页数);
- **事务性**:准入时原子预留全部 KV 页与 GDN slot,容量不够整体拒绝;decode 批次的
  KV 长度推进也是单事务——整批失败先回滚逻辑长度与状态,再二分重试隔离单请求;
- **prefix cache**(radix,只缓存 full-attention 层):物理页共享 + 写保护,命中页
  只复用存储;GDN 状态不缓存(依赖前层输出,不宣称 prefix-compute skip)。

### 6.2 Recurrent state pool([cache/state_pool.py](../hydraserve/cache/state_pool.py))

- GDN 状态形状固定(与长度无关)→ **固定槽位**预分配,layer-major 连续 GPU pool;
- decode 每轮"整批 gather → 跨层事务工作区 → commit",替代每层 `cat` 与逐请求
  回拷——简历的"零分配批量提交":运行期不再出现逐 batch 的临时分配
  (实测消除每步 52-816 MiB 临时分配,batch 1/4/8/16 搬运加速 1.83×/1.39×/1.36×/1.22×);
- 最终 logits 成功前不发布新状态(事务性,保证失败可回滚)。

### 6.3 双状态容量联合准入

一个请求同时需要两类资源:KV 页(随长度线性)+ state slot(固定)。admission 时
**两类一起预留**,否则流式输出中途才失败;permanent 超容量请求单独失败,入口过载
HTTP 429,统一容量快照供路由与监控复用。

---

## 7. 双状态传输协议

### 7.1 为什么 PD 分离,以及代价在哪

prefill 与 decode 对 GPU 的用法相反(§1.3),放同一张卡互相干扰。PD 分离把两者
放不同卡,但引入**传输代价**。三种模式([transfer/descriptor.py](../hydraserve/transfer/descriptor.py) 的 `TransferMode`):

| 模式 | 传输内容 | 适用带宽 | 备注 |
|------|---------|---------|------|
| FULL | BF16 KV + FP32 状态 | NVLink 级(112 GB/s,32K 9B 约 9 ms) | 最简单 |
| QUANTIZED | INT4 KV(345 MB)+ 状态 | P2P 级(12-16 GB/s,约 29 ms) | KV 压缩 3.2× |
| PARTIAL | 仅 FP32 状态(53.48 MB)+ decode 端 KV 重算 | SHM 级(4.58 GB/s 约 12 ms + 重算) | 当前实测模式 |

### 7.2 描述符的强不变量(学习协议设计的好范本)

`StateTransferDescriptor` 用类型系统 enforce 规则:

- PARTIAL 不得携带 KV 区域;QUANTIZED 必须含 INT4 KV;循环状态区域必须
  FP32 且不可量化;
- `state_token_count ∈ [1, prompt_length]` 且 `prompt_length − state_token_count ≤ 1`
  ——即只支持**全量或 N-1 截断**,没有中间态。

### 7.3 N-1 truncation 与首 token 预播种

GDN 状态是逐 token 递推的,第 T 步的状态依赖第 T 个 token。prefill 端把前
N−1 个 token 的状态发给 decode 端,decode 端用 prefill 已生成的**首 token**
(描述符里的 `first_token_id`)本地重算最后一步,得到完整状态——避免多传一次
往返,同时首 token 成为 replay 校验的权威输出。注意:recurrent state 的迁移是
"算到哪传到哪",和 KV 的"按 token 截断"不是一回事。

### 7.4 传输机制

- **SHM 后端**:typed ndarray 单信封写共享内存,**header 最后发布**——接收方
  永远看不到半写的 payload;GDN 状态经 pinned host staging 搬运;
- **CUDA P2P 后端**:先 `cudaDeviceCanAccessPeer` 能力检测,不可用则显式失败回退
  SHM(开发机与云端均为 CNS,所以实测路径是 SHM PARTIAL);
- **层级流水线**(协议+单测):层 i 的输出传完即可开始层 i+1,与计算重叠;
  这是 NVLink 场景的预留能力。

**动手**:`python -m mmap` 写一个"先写 payload 再写 header"的单信封示例,体会
发布协议为何要按这个顺序;再算 53.48 MB ÷ 4.58 GB/s ≈ 11.7 ms 与
32K prefill(~1.8 s)的数量级差。

---

## 8. Continuous Batching 与调度容错

### 8.1 为什么需要 continuous batching

decode 是带宽密集:batch 越大,同一份权重被越多 token 共享,利用率越高。
静态 batching 等整批结束才进下一批(木桶效应);**迭代级调度**(Orca,2302.10523)
每步都重新组批:完成的请求即时退出,新请求即时加入。

### 8.2 chunked prefill 与干扰

长 prompt 的 prefill 若独占一次执行,会长时间阻塞 decode(§1.3 的 2.5-6.4×);
chunked prefill 把 prompt 切成 `--prefill-chunk-size` 块,在 decode 迭代间
插空执行。代价:continuation chunk 要走 paged 历史读取(§3.3)。

### 8.3 公平调度([engine/fair_scheduler.py](../hydraserve/engine/fair_scheduler.py))

每个候选请求打一个**越低越优先**的分数(调度器取最小者):

```
score = service_tokens/(priority+1)          # WFQ:已服务 token 除以权重,最"亏欠"者优先
      − priority_bias·priority               # 高优先级减分 → 更优先
      − 0.25·waiting_rounds                  # 等待老化:等得越久分越低 → 防饿死
      − 8.0·deadline_urgency                 # deadline 逼近时突增减分
deadline_urgency = max(0, 1 − 剩余时间/1.0s)
```

第一项是经典 **weighted fair queueing**:服务量除以优先级权重,选比值最小者,
长请求不会饿死短请求、高优先级获得加权份额;后三项是 HydraServe 在此之上的
优先级/老化/deadline 修正。

- 临时容量不足的请求回候选队列,不阻塞后续可准入请求;
- `--max-active-requests`(持有资源的请求数)与 `--max-batch-size`(单步进
  kernel 数)分离,调度器在 batch 之外保留等待集。

### 8.4 preemption 与精确 replay

高优先级/早 deadline 请求可在 decode 迭代边界**抢占**低紧迫度请求,立即释放其
KV/GDN 容量;受害请求稍后用 `prompt + generated[:-1]` **精确重算**恢复——保留
已输出 token、采样 step 与停止序列状态,客户端无感知。`--max-preemptions-per-request`
(默认 2)限制反复抢占。GPU kernel 不可中断,所以 deadline 是**协作式**(kernel
边界)而非微秒级抢占——这是必须会解释的设计取舍。

### 8.5 容错:worker 监督与降级

decode 子进程退出/RPC 超时 → 从路由摘除 → 重建进程与 IPC → 模型名/容量握手
通过后重新加入;故障 worker 上全部绑定原子失效,在途请求保留已输出历史、
rebind 健康 worker 后精确 replay,不直接向客户端报错。prefill 故障时新请求
fail-closed 到 collocated,恢复后自动回 PD。`/health`、`/metrics` 暴露健康、
恢复中状态、重启计数与 fault suspension。

---

## 9. 1P+ND 与校准式自适应路由

### 9.1 1P+ND([engine/multi_worker.py](../hydraserve/engine/multi_worker.py))

一个 prefill worker + N 个各自持有 KV/GDN 容量的 decode worker:
registry 先过滤不健康/容量不足目标 → 按 decode load、prefix-cache 匹配长度、
链路带宽/跳数评分 → 绑定后不再变更 → 各 GPU RPC 并行发起、结果按原请求序归并。
一个 decode worker 失败不丢其他 worker 已生成的 token(按 worker 汇总部分结果)。

### 9.2 延迟曲线模型([router/adaptive_router.py](../hydraserve/router/adaptive_router.py) 的 `LatencyCurve`)

$$\text{cost}(L, \ell) = (f + aL + bL^2)\cdot(1 + s\,\ell)$$

L = prompt 长度,ℓ = decode load(0-1),s = 负载放大系数。
二次项捕捉 attention 的 O(L²),load 乘子是"decode 忙时 prefill 更慢"的一阶外部性。

### 9.3 拟合([router/calibration.py](../hydraserve/router/calibration.py))

- 输入:concurrency-1 预热基准结果(失败请求剔除);低负载样本(ℓ≤0.05)拟合基础
  曲线,负载样本单独拟合 `s`(中位数估计,截断到 10);
- 最小二乘 + **非负约束**:3 个变量的 NNLS 用 active-set 枚举(2³−1=7 个子集
  各自 lstsq,取可行最优)——防止噪声拟合出"随长度下降"的延迟曲线;
- 输出 profile + 每曲线的 RMSE 与样本范围。

### 9.4 在线决策(`CostAwareRouter.decide` 全流程)

1. 分别预测 collocated 与 PD 成本,PD 乘**风险系数** 1.10(RPC 超时/重算的不确定性);
2. `savings = collocated − risk_adjusted_pd`,必须超过 `max(5 ms, 5%·collocated)`;
3. 短于 `minimum_pd_prompt_tokens` 直接 collocated(保守);
4. **Schmitt-trigger 迟滞**:当前路线的切换需要额外跨过迟滞带
   (`max(5 ms, 2%·collocated)`),减少边界抖动;
5. **在线 EWMA 校准**:按 prompt 长度桶(`bit_length(L)−1`)维护实测/预测比
   correction(α=0.2,首观测直接采用),连续 ≥5 次且偏离超过 1.5×(或低于 1/1.5)
   判定 drift → **fail closed 到 collocated**(RPC 超时结果未知,盲重试有重复
   执行风险),`/health` 转 degraded;
6. 置信度 = 该桶最小观测数/5,capped 1。

### 9.5 为什么静态阈值是错的(实测证据)

2026-08-14 实测:9K prompt 在 SHM PARTIAL 下,静态 8K 阈值会把请求路由到 PD,
但 PD 实际 TTFT 比 collocated **慢 49%**(KV 重算 + IPC 开销 > 干扰隔离收益)。
装上成本路由后,同一请求 `route_reason=cost_model_collocated` 正确落回本地。
结论:阈值必须由实测传输/重算成本建模,不能只按长度猜。

**动手**:取 `benchmark_output/2026-08-14_*_c1_*.json`,用 numpy 复现 §9.3 的
NNLS 拟合,与 `python -m hydraserve fit-router-profile` 输出对照。

---

## 10. 基准方法学

| 指标 | 定义 |
|------|------|
| TTFT | 提交 → 首 token(排队 + prefill + 传输) |
| TPOT | 首 token 后每输出 token 平均时长(decode 效率) |
| 分位数 | P50/P95/P99,尾延迟看 P99 |
| request/s、output tok/s | 吞吐,除以整段墙钟(含尾部) |

规范要点:

- **warmup** 排除首次 Triton/CUDA 编译,且 warmup 后清空路由 EWMA/迟滞状态,
  避免把冷编译误学成 drift;
- 到达过程:burst(测峰值吞吐)/ fixed / Poisson(测排队行为);4 个独立 Poisson
  各 λ/4 之和 = Poisson λ,这是多进程 DP 基线的公平做法;
- 公平对比:同采样参数、同容量配置、同截断、同 warmup;
- 干扰测量:同卡跑 prefill 时测 decode 的 TPOT 退化倍数(2.5×/6.4× 的测法)。

---

## 11. 面试自测题

1. 为什么 decode 是带宽密集而 prefill 是算力密集?batch 如何影响二者的利用率?
2. 手写 delta rule 递推式,并解释 beta gate 与 decay 的语义。
3. 为什么 GDN 状态不能 INT4 量化而 KV 可以?(误差跨时间步累积 vs 独立存储)
4. online softmax 的修正因子为什么 ≤1?推导两段拼接。
5. Paged Attention 里物理页表怎么查?BLOCK_T 16 是什么含义?
6. AWQ GEMM 为什么能在 kernel 内即时反量化?显存账怎么算?
7. E4M3FN 位模式怎么手工解码?为什么 3090 上必须这样做?
8. PARTIAL/FULL/QUANTIZED 三种模式各适合什么带宽?传输时间怎么算?
9. N-1 truncation 为什么是 N-1 而不是 N?first token 由谁生成、用来干什么?
10. SHM 单信封为什么 header 最后写?
11. continuous batching 相比静态 batch 的收益原理(迭代级调度)?
12. preemption 后怎么恢复?为什么 replay `prompt + generated[:-1]` 而不是重跑?
13. 公平调度公式三项各起什么作用?老化为什么能防饿死?
14. 为什么 deadline 是协作式而不是抢占式?
15. 路由延迟曲线的四项系数各是什么含义?非负约束为什么必要?
16. EWMA 校正在干什么?drift 判定后为什么 fail closed?
17. 为什么 9K prompt 静态阈值会路由错、成本路由能选对?
18. worker 崩溃后请求经历了什么?什么条件下客户端无感知?
19. TTFT/TPOT 的定义;Poisson λ/4 × 4 进程为什么等价于一个 λ 均衡器?
20. 本项目的诚实边界:哪些是实卡验证、哪些只有单测?

(每题答案都在上文对应小节;答不出来的回到该节重读。)

---

## 12. 仓库阅读顺序与术语表

```text
config.py                    ← 模型长什么样(架构字段)
kernels/reference.py         ← 每个算子的数学 oracle(先读)
kernels/{rmsnorm,gdn,kv_cache,paged_attention,awq,fp8}.py
cache/{paged_kv,block_manager,state_pool,prefix_cache,memory_planner}.py
transfer/{descriptor,backend,pipeline}.py
engine/{serving_loop,fair_scheduler,chunked_prefill,pd_worker,multi_worker}.py
router/{calibration,adaptive_router}.py
api/server.py → benchmark/runner.py
```

| 术语 | 含义 |
|------|------|
| GDN | Gated DeltaNet,Qwen3.5/3.6 的线性注意力层 |
| GQA | Grouped Query Attention,多 query head 共享 KV |
| PD 分离 | prefill 与 decode 放不同 GPU |
| 1P+ND | 一个 prefill worker + N 个 decode worker |
| PARTIAL_TRANSFER | 只传循环状态、decode 端重算 KV |
| online softmax | 分块 softmax 的增量合并算法 |
| EWMA | 指数加权移动平均(α=0.2) |
| drift | 在线观测与 profile 的持续偏离 |
| fail closed | 异常时退回最保守路径(collocated) |
| replay | 用 prompt+已生成前缀重算恢复请求 |

## 延伸阅读(只有想深挖时才需要)

- GPU/CUDA:《Programming Massively Parallel Processors》前 8 章;
  CUDA C++ Programming Guide(Execution Model / Memory 两章);
- Triton:官方教程 01-10(triton-lang.org);
- 架构线:RetNet(2307.08621)、Mamba(2312.00752)、Mamba-2(2405.21060)、
  DeltaNet(2406.06484)、Gated Delta Networks(2412.06464);
- Attention 系统:FlashAttention(2205.14135/2307.08691/2407.08608)、
  vLLM(2309.06180)、Orca(2302.10523)、SGLang RadixAttention(2312.07104);
- 量化:GPTQ(2210.17323)、AWQ(2306.00978)、KIVI(2402.02750)、
  DeepSeek-V3 FP8(2412.19437)、FP8 格式(2209.05433);
- PD 分离:DistServe(2401.09670)、Mooncake(2407.00079)、Splitwise(2311.18677)、
  Sarathi(2308.16369)。
