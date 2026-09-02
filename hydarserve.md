# HydraServe 当前实现与架构说明

> 文档审阅基线：HydraServe `b4f778d`（2026-08-27 工作区）。
>
> V4 压测基线：`a090ec9`，结果元数据标记为 `git_dirty=true`。
>
> 项目状态：研究与工程原型，不应把某一次压测结果解释为“PD 普遍优于 DP”。

## 1. 项目定位

HydraServe 是一个面向混合注意力模型的 Prefill–Decode（PD）分离推理引擎原型。它不依赖 Transformers、vLLM 或 SGLang 执行模型，当前主要围绕 Qwen3.5/Qwen3.6 的混合结构展开：

- Full Attention 层保存随上下文长度增长的 KV Cache；
- Gated DeltaNet（GDN）层保存固定大小的 FP32 recurrent state 和 causal-convolution state；
- PD 分离时必须同时处理这两类状态，不能只迁移传统 Transformer KV Cache。

项目的核心价值是验证以下能力：

1. 从零实现混合注意力模型运行时与 Triton Kernel；
2. 定义 KV Cache 与 recurrent state 的统一传输协议；
3. 在消费级多 GPU 环境中验证静态 PD、条件 PD 和动态 Hybrid 调度；
4. 研究 prefill 干扰隔离、decode 容量和状态传输成本之间的权衡。

当前结果只支持“在仓库给定硬件、负载和配置下，H1 优于相应 D0/P0 基线”，不支持“PD 在所有场景都优于 DP”的一般性结论。

## 2. 当前模型配置与状态大小

### 2.1 Qwen3.5-4B preset

当前 `hydraserve/config.py` 中的 4B preset 为：

| 配置 | 当前值 |
|---|---:|
| hidden size | 2560 |
| hidden layers | 32 |
| Full Attention interval | 4 |
| Full Attention layers | 8 |
| Linear Attention layers | 24 |
| attention heads | 16 |
| KV heads | 4 |
| attention head dim | 256 |
| GDN key heads / dim | 16 / 128 |
| GDN value heads / dim | 32 / 128 |
| convolution kernel dim | 4 |
| recurrent dtype | FP32 |

因此，4B 模型不是“36 层、8 Full + 28 Linear”，而是 **32 层、8 Full + 24 Linear**。

### 2.2 KV Cache 大小

BF16 KV 每 token 的字节数为：

```text
8 full layers × 4 KV heads × 256 head dim × 2(K,V) × 2 bytes
= 32,768 bytes/token
```

对应的理论数据量为：

| Prompt 长度 | BF16 KV |
|---:|---:|
| 8K | 256 MiB |
| 16K | 512 MiB |
| 32K | 1 GiB |

KV Cache 的存储量化和 PD 链路量化是两个独立配置：

- `kv_quant=int8`：GPU KV Cache 使用 INT8；
- `pd_transfer_quant=None`：传输时仍发送完整 BF16 KV；
- `pd_transfer_quant=int8`：传输链路使用 INT8 payload。

V4 使用的是 **INT8 KV Cache + 完整 BF16 链路传输**，不能把它描述成 INT8 wire transfer。

### 2.3 Recurrent state 大小

4B preset 的 GDN 状态为：

```text
SSM shape  = (24, 32, 128, 128)
SSM bytes  = 50,331,648

conv width = 2 × (16 × 128) + (32 × 128) = 8192
conv shape = (24, 8192, 4)
conv bytes = 3,145,728

total      = 53,477,376 bytes ≈ 53.48 MB ≈ 51.0 MiB
```

状态池还需要 batch workspace、模型权重、KV Cache 和 CUDA Graph buffer，因此不能只根据单槽大小直接推导整卡是否 OOM。实际容量由 `GpuLinearStatePool` 和 memory planner 共同决定。

## 3. 系统架构

### 3.1 运行模式

HydraServe 当前包含三类主要运行模式：

| 模式 | 行为 |
|---|---|
| 同卡混部（Collocated）/请求级数据并行（DP） | 每个 GPU worker 都是完整模型副本；一个请求绑定一个副本，并在本地完成 Prefill 和 Decode |
| 静态预填充-解码分离（Static PD） | 固定 P worker 负责 Long Prefill，固定 D-bound worker 接收状态并继续 Decode |
| 动态混合角色（Dynamic Hybrid） | Hybrid worker 空闲时服务 Short/Decode，Long 请求到达时切换为 Prefill producer |

动态 Hybrid 的目标不是永久减少 Decode GPU，而是通过工作守恒调度（work-conserving scheduling，也可称非空转调度）在两种角色间切换：

```text
无 Long 请求：Hybrid 与 D-bound 完整模型副本一起服务 Short/Decode
Long 请求到达：Hybrid 停止接收新 Decode 工作并切换到 Prefill
Long Prefill 完成：请求交给目标 D-bound owner，Hybrid 回到逻辑 Decode 池
```

这里的物理资源与逻辑角色必须分开理解：

| 名称 | 实际含义 | 能否本地 Prefill | 能否 Decode | 是否产生 Long PD 状态 |
|---|---|---:|---:|---:|
| D0 的 DP worker | `MultiGPUCollocatedBackend` 中的完整模型副本，一个请求固定绑定一个副本 | 是 | 是 | 否 |
| H1 的 D-bound worker | `decode_devices` 对应的完整模型副本；持有请求最终 KV/state | Short 可以 | 是 | 通常不作为 Long 的 P producer |
| H1 的 Hybrid worker | `prefill_devices` 对应的完整模型副本；逻辑角色可以切换 | 是 | 空闲时可以 | Long 到达时可以 |
| 逻辑 Decode 池 | 当前可接受 Decode/Short 的 worker 集合 | 取决于成员 | 是 | 不是物理池或 NCCL group |

因此，代码中不存在“只能 Decode、没有 Prefill 能力的残缺模型卡”：

- D0 的多卡模式是请求级数据并行（request-level data parallelism）。每张卡持有完整权重并独立处理请求，不做 Tensor Parallel collective 或 all-reduce。
- H1 中的 GPU 在模型存储层面同样是完整副本，只是调度角色不对称。
- “D 卡”更准确的名称是 **D-bound 完整模型副本**或**解码归属 worker**：它是 Long PD 状态的接收方和后续 Decode owner，但仍可为 Short 请求执行本地 Prefill+Decode。
- “常驻 D”只能作为“角色固定为 D-bound、不切换成 Long Prefill producer”的简称，不能理解为该卡完全不执行 Prefill。本文后续不再使用容易误解的“常驻 D 卡”。
- “Decode 池”是调度器看到的逻辑解码可用 worker 集合，包括 D-bound worker，以及当前处于 `DECODE` 角色的 Hybrid worker；它不是独立的物理显卡池、模型分片或通信组。
- 不提供多 GPU 拓扑参数时，CLI 默认是单 GPU 同卡混部，不是多卡 DP；显式提供 `--dp-devices` 才进入 D0 完整副本 DP，使用 `--adaptive --decode-devices ... --prefill-devices ...` 才进入 H1 动态 Hybrid。

实验拓扑 `H1=1H+3D` 的准确含义是：1 个可切换角色的完整模型副本，加 3 个以状态承接和 Decode 为主、同时能为 Short 请求执行同卡 Prefill 的完整模型副本。

### 3.2 Hybrid worker 状态机

代码中的角色状态是：

```text
DECODE -> PREFILL_PENDING -> PREFILL_ACTIVE -> DECODE
```

- `DECODE`：可参与 collocated short 和 decode 调度；
- `PREFILL_PENDING`：已经绑定 Long 请求，停止接收新的普通 Decode 工作，等待安全切换；
- `PREFILL_ACTIVE`：执行 Long Prefill、分块发布状态并完成迁移收尾；
- 回到 `DECODE`：重新成为可用 decode worker。

Prefill 计算、Chunk KV 传输和最终 bundle 提交是 `PREFILL_ACTIVE` 内部的执行阶段，并不是三个独立的 worker 角色状态。

条件 PD 使用 `prompt_tokens >= conditional_pd_tokens`。如果希望长度恰好为 2048 的 short 留在 collocated 路径，阈值必须设置为大于 2048；V4 使用 4096。

### 3.3 一个请求可能走的四条路径

```text
HTTP request
    |
    v
分词（tokenize）+ 有界准入
    |
    v
worker容量预留 + 路由
    |
    +-- D0 ------------------------------------------------------+
    |   选择一个DP副本 -> 本地Prefill -> 本地Decode -> 返回token  |
    |                                                            |
    +-- H1 Short / D-bound -------------------------------------+
    |   绑定D-bound副本 -> 本地Prefill -> 本地Decode              |
    |                                                            |
    +-- H1 Short / idle Hybrid ---------------------------------+
    |   绑定DECODE角色的Hybrid -> 本地Prefill -> 本地Decode        |
    |                                                            |
    +-- H1 Long / PD -------------------------------------------+
        预留D-bound副本 -> 绑定Hybrid P -> P执行Prefill
        -> KV + recurrent state分块迁移 -> D恢复并replay
        -> D继续逐token Decode
```

前三条是同卡混部路径：Prompt 和后续 Decode 在同一完整模型副本上完成，不发生跨 GPU 状态迁移。第四条才是真正的预填充-解码分离 Long 路径。

### 3.4 单请求端到端完整链路

本节从一个 OpenAI-compatible 请求进入服务开始，一直追踪到最终 token 返回客户端，包括路由、模型内部 Kernel、状态传输、dtype 变化、逐 token 生成和资源释放。

#### 3.4.1 HTTP、Tokenize 与请求入队

请求首先进入 `/v1/completions` 或 `/v1/chat/completions`：

1. Chat 请求先通过 tokenizer chat template 渲染成 Prompt 文本；
2. tokenizer 把字符串编码成 Python `list[int]`；
3. 校验 `max_tokens`、context limit、sampling 参数、priority、timeout 和 stream；
4. `ContinuousGenerationLoop.submit` 创建 `ServingRequest`；
5. 用户未提供 seed 时，系统生成请求级 seed；
6. 有界准入队列同时限制请求数和 `prompt_tokens + max_new_tokens` 总量；
7. 返回 `GenerationHandle`，HTTP 线程随后阻塞读取或流式消费 `GenerationEvent`。

这一阶段的数据主要位于 CPU：

```text
prompt text:       UTF-8 string
token_ids:         Python int tuple
sampling metadata: Python dataclass
```

#### 3.4.2 准入控制、容量预留与路由

GenerationLoop 从输入队列取出请求后，先进行 worker 级准入控制（admission control），而不是立刻发射模型 Kernel。候选 worker 至少检查：

- 是否有足够的 KV block；
- 是否有空闲 recurrent-state slot；
- worker 是否健康；
- prefix cache 命中量；
- 当前 Decode load；
- Long 请求是否有可用 Hybrid Prefill worker。

系统通常为请求预留：

```text
prompt_tokens + max_new_tokens - 1
```

个逻辑 KV token。减一是因为最后采样出的 output token 如果已经触发停止条件，就不会再作为下一轮模型输入写入 KV。

H1 条件式 PD 的路由规则如下：

1. **Short 请求**：长度小于阈值，空闲 Hybrid 和 D-bound worker 按实时 Decode load 竞争，请求绑定其中一个完整模型副本，走本地 Prefill+Decode。
2. **Long 请求**：长度大于等于阈值，先选择并预留一个 D-bound worker，建立不可变的 Decode owner；再绑定一个 Hybrid worker，把其角色从 `DECODE` 切换到 `PREFILL_PENDING`。
3. **Hybrid 暂不可用**：已预留的 D-bound worker 可以回退到同卡 Prefill，避免 reservation 泄漏或请求丢失。

#### 3.4.3 Prompt 在模型内部的 Kernel 级计算

以下以 Qwen3.5-4B BF16 主路径为例。Prompt token 在目标 worker 上转换为：

```text
input_ids: torch.int64 [batch, sequence]
```

Embedding lookup 得到 BF16 hidden state：

```text
hidden: [batch, sequence, 2560], BF16
```

模型共有 32 层，包括 24 个 Linear Attention/GDN 层和 8 个 Full Attention/GQA 层。每层执行两段 Pre-Norm residual：

```text
residual = hidden
hidden = RMSNorm(hidden)
hidden = Attention_or_GDN(hidden)
hidden = residual + hidden

residual = hidden
hidden = RMSNorm(hidden)
hidden = MLP(hidden)
hidden = residual + hidden
```

RMSNorm Triton Kernel 将输入转为 FP32，计算平方和、方差、`rsqrt` 和权重乘法，最后存回 BF16。

##### GDN 层

一层 GDN 的 Kernel 路径为：

```text
BF16 hidden
  |
  +-- fused QKVZ GEMM --> mixed(Q,K,V) + output gate，BF16
  |
  +-- fused BA GEMM ----> beta_raw + step_raw，BF16
                           |
                           v
                    GDN gating Triton
                    sigmoid(beta) + decay
                    输出 FP32 beta/decay
```

`mixed` 先经过 causal depthwise-convolution：

```text
mixed [B,T,8192]
 + convolution weight
 + fixed history state
 -> convolved Q/K/V
 -> next convolution state
```

Blocked causal-conv Kernel 按 `(batch×token, channel_tile)` 并行，一个 Triton program 处理一段 channel。随后张量重排为：

```text
Q/K: [B,T,16,128]
V:   [B,T,32,128]
```

GDN 的 value head 数是 key head 的两倍；blocked Kernel 通过 `HEAD_RATIO` 映射共享对应 Q/K，避免真实执行 `repeat_interleave`。

GDN recurrent Kernel 的启动网格为：

```text
grid = (batch, value_head, value_dim_tile)
```

每个 program 载入一个 `[key_dim, BLOCK_V]` FP32 state tile，并沿 token 维顺序递推：

```text
S'ₜ = exp(gₜ) · Sₜ₋₁
pₜ  = kₜᵀS'ₜ
δₜ  = βₜ(vₜ-pₜ)
Sₜ  = S'ₜ+kₜδₜᵀ
oₜ  = qₜᵀSₜ
```

token 间存在真实依赖，不能并行；并行度来自 batch、head 和 value tile。state tile 在该 program 的 token 循环内驻留寄存器/片上存储，循环结束后统一写回 global memory。GDN core 随后经过 fused gated RMSNorm：内部以 FP32 累加归一化并计算 `SiLU(gate)`，输出 BF16，再执行 output projection。

##### Full Attention / GQA 层

Full Attention 层先执行 fused QKV projection：

```text
Q + output_gate: [B,T,16,512]
K:               [B,T,4,256]
V:               [B,T,4,256]
```

Q 的最后一维拆成真实 Query 和 output gate：

```text
Query:       [B,T,16,256]
Output gate: [B,T,16,256]
```

随后依次执行 Q/K RMSNorm、RoPE、Paged KV 写入、Attention、`attention × sigmoid(output_gate)` 和 output projection。Paged KV 写入先用 block table 将逻辑 token 位置映射到物理 block/offset。V4 `kv_quant=int8` 会针对每个 token、每个 KV head 执行：

```text
BF16 K/V -> FP32 absmax -> FP32 scale
          -> round/clamp -> INT8 K/V
```

因此 GPU Cache 保存 INT8 K/V payload 和 per-token-per-head FP32 scale。读取时，当前实现通过 `int8.float() × scale` 恢复 BF16 page tensor，再进入 Flash/paged attention。

不同阶段使用不同 Attention Kernel：

- 首个多 token chunk：FlashAttention varlen；
- 带历史的 continuation chunk：paged flash prefill；
- Flash 不可用时：自研 tiled paged online-softmax；
- 单 token Decode：默认 split-K paged attention，也可切换 flash/reference。

Split-K Decode 将长 context 分片。每个 split 计算局部 `maximum/denominator/accumulator`，reduce Kernel 再按 online-softmax 规则缩放合并，避免一个 program 串行扫描全部历史。

##### MLP、最终投影与首 token

每个 Attention/GDN 子层之后都执行：

```text
BF16 hidden
 -> fused gate_up GEMM
 -> Triton SiLU(gate) × up（内部FP32，输出BF16）
 -> down projection
 -> residual add
```

32 层完成后执行 final RMSNorm。LM Head 只投影最后一个 Prompt 位置，默认生成 BF16 logits `[1,vocab]`；环境变量也可强制 FP32 logits。Greedy 且无 penalty 时直接对原 dtype 做 batched argmax；temperature、top-k、top-p、min-p 或 penalty 路径先将该行 logits 转为 FP32，再过滤、softmax、multinomial，得到首个 output token `y0`。

#### 3.4.4 Long PD 的接收方先启动、分块迁移与 N−1 重放

假设 Prompt 为 `x0...xN-1`，第一个生成 token 为 `y0`。Long PD 主路径遵循接收方先启动（receiver-first）和分块传输（chunked transfer）：

```text
Client / ServingLoop
        |
        | 1. admission + route
        v
MultiWorkerBackend
        |
        | 2. 先向 D 派发 prepare，确保接收方已启动
        | 3. 再向 P 派发 prefill
        v
Prefill Worker -------------------------- Decode Worker
        |                                      |
        | 4. 每个 prefill chunk 完成后抽取 KV  |
        |---- manifest / KV chunk ------------>|
        |                                      | 5. 后台接收并安装
        | 6. 在 N-1 边界提取 recurrent state  |
        | 7. P 处理最后一个 prompt token       |
        |---- 最后一个 token 的 KV ------------>|
        |---- recurrent state + 描述符 -------->|
        |                                      | 8. 导入 N-1 state
        |                                      | 9. replay 最后一个 token
        |<----------- token 一致性 ------------|
        |
        | 10. P/D 两侧均完成后，才返回首个采样 token
        v
ServingLoop -> 后续 decode
```

具体执行顺序如下：

1. Coordinator 在目标 D-bound worker 预留 KV blocks 和 FP32 state slot。
2. Coordinator 先向 D 派发 `prepare`，确认 receiver 已进入接收路径。
3. Hybrid 从 `PREFILL_PENDING` 进入 `PREFILL_ACTIVE`，并为 N 个 Prompt token 分配本地 Paged KV。
4. P 先计算 `x0...xN-2`；每个 chunk 完成后记录 CUDA Event，并由独立 transfer stream 等待该 Event。
5. Transfer stream 从 Paged Cache 抽取已完成 chunk，复制到 Pinned CPU buffer，再写入分块 SHM Ring；D 后台接收并在独立 install stream 安装。
6. P 在 N−1 边界提取 GDN recurrent state，SSM 和 Conv state 统一转为 CPU FP32 连续数组。
7. P 继续计算最后一个 Prompt token `xN-1`，产生其 KV 和最终 logits，并采样得到 `y0`；`xN-1` 的 KV 也会被发送。
8. P 发送最终 bundle：描述符（descriptor）记录 `first_token_id=y0` 和 KV ranges，另带 FP32 recurrent state；已流式发送的 KV 不在 bundle 中重复传输。
9. D 校验所有 KV range 完整，安装 FP32 recurrent state，并将 `sequence_length` 设置为 N−1。
10. D 用 `xN-1` 再执行一次 `runtime.forward`，读取已安装历史 KV，将 GDN state 从 N−1 精确推进到 N；这个 forward 也会重写同一逻辑位置的 `xN-1` KV。
11. D 使用相同 sampling 参数重新得到首 token，并检查它是否等于 P 侧 `y0`。
12. P 侧 transfer future 和 D 侧 prepare 均完成后，Coordinator 才将 `y0` 交给 ServingLoop。

因此，**N−1 只表示 recurrent state 的 token 边界，不表示只传 N−1 个 KV token**。流式 FULL KV 路径传输包括最后一个 Prompt token 在内的全部 KV。客户端可见 TTFT 也包含 D 端接收、安装和 replay 的尾部时间，不能表述为“首 token 完全不等待传输”。

#### 3.4.5 V4 主路径的完整数据类型（dtype）变化

| 阶段 | dtype / 表示 | 说明 |
|---|---|---|
| HTTP Prompt | UTF-8 string | CPU |
| tokenizer 输出 | Python int tuple | CPU |
| 模型输入 | `torch.int64` | GPU/Embedding 所在设备 |
| 权重、hidden、residual | BF16 | 4B V4 主路径 |
| RMSNorm/SiLU 内部 | FP32 accumulate | 输出重新存为 BF16 |
| Q/K/V projection | BF16 | RoPE 后仍为模型 dtype |
| P/D KV Cache payload | INT8 | V4 `kv_quant=int8` |
| KV scale | FP32 | per-token-per-head |
| Attention 输入 | INT8×FP32 scale→BF16 | softmax 统计量通常为 FP32 |
| GDN beta/decay | FP32 | gating Kernel 输出 |
| GDN SSM state | FP32 | 递推和 state pool 均为 FP32 |
| Conv state | 协议和 state pool 为 FP32 | P 侧 fresh buffer 可随 hidden dtype 创建，codec 发送前统一 `.float()` |
| P 侧 FULL wire KV | BF16 raw bits 视作 NumPy `uint16` | 只是 reinterpret，不是整数数值转换 |
| P 侧 wire recurrent | NumPy FP32 | Pinned D2H 后进入 SHM |
| SHM Ring | bytes + metadata | 固定 slot 数据面 |
| D 收到 FULL KV | `uint16`→reinterpret BF16 | 恢复原 BF16 位模式 |
| D 端 Cache 安装 | BF16→FP32 scale+INT8 | 按目标 cache 配置重新量化 |
| D 端 state pool | FP32 | 固定 SSM/Conv slot |
| LM Head logits | 默认 BF16 | `HYDRASERVE_FP32_LOGITS=1` 时为 FP32 |
| 随机采样 | FP32 score/probability | Greedy 快路径可直接 argmax BF16 |
| 输出 | Python int→增量 UTF-8 string | 进入事件、tokenizer 和 SSE/JSON |

V4 FULL KV 的传输链路可归纳为：

```text
P INT8 Cache
 -> 读取时反量化为BF16
 -> BF16 raw bits作为uint16进入Pinned Buffer/SHM
 -> D reinterpret为BF16
 -> D按自身kv_quant重新量化为INT8+FP32 scale
```

所以当前实现包含一次 P 侧反量化和一次 D 侧重量化；直接做 INT8 Cache-to-Cache 传输属于后续优化方向。AWQ packed INT4 和 FP8 E4M3FN 则是权重格式，不应与 KV Cache 或 wire transfer 的量化混为一谈。

#### 3.4.6 后续解码（Decode）token 如何逐个生成

ServingLoop 将 READY 请求放入 active 集合。每轮 Decode：

1. 公平解码调度器（`FairDecodeScheduler`）根据已服务 token、优先级、等待老化和截止期限紧迫度选择请求；
2. MultiWorker backend 按不可变 owner 分组：D-bound 请求一组，仍绑定 Hybrid 的 collocated Short 请求一组；
3. 不同物理 worker 的 Decode batch 可以独立并行发射，不需要组成跨 GPU 同宽 batch；
4. 每个 worker 为组内请求增长一个 KV 逻辑 token，首次 Decode 的输入是刚生成的 `y0`；
5. `GpuLinearStatePool.batch` 通过 `index_select` 将各请求 FP32 SSM/Conv slot 收集到预分配 workspace；
6. `runtime.decode_batch` 执行 32 层单 token 前向；Full Attention 写入 `y0` 的 K/V，再读取 `prompt+y0` 历史；GDN 原地推进 SSM 和 Conv state；
7. 所有层和 LM Head 成功后，事务才提交 pooled recurrent state 并推进 `sequence_length`；
8. 从新 logits 采样 `y1`，worker 更新 generated history，ServingLoop 将其包装成 `GenerationEvent`；
9. HTTP 层用 `IncrementalTextDecoder` 增量解码，并通过 SSE 或最终 JSON 返回。

循环关系为：

```text
输入y0 -> 写入y0的KV/state -> logits -> 采样y1
输入y1 -> 写入y1的KV/state -> logits -> 采样y2
...
```

最新采样出的 token 在当轮尚未进入 KV；只有它在下一轮成为模型输入时，才会产生自己的 K/V 和 GDN 状态更新。

#### 3.4.7 停止、释放与客户端可见结果

每次生成 token 后，ServingLoop 检查 EOS、token 级 stop sequence、`max_new_tokens`、timeout 和 cancel。触发停止后：

1. 请求从 active 集合移除；
2. release 提交到独立 executor，避免阻塞 Decode 关键路径；
3. owner worker 释放 Paged KV block、prefix 引用和 FP32 state slot；
4. `GenerationHandle` 收到 finished event；
5. 非流式请求汇总可见 token 并解码完整文本；流式请求逐 token 发送 SSE，stop/EOS token 按 API 语义从最终文本裁剪。

#### 3.4.8 端到端代码路径索引

| 阶段 | 主要代码 |
|---|---|
| HTTP 解析与 tokenize | `hydraserve/api/server.py` |
| 请求对象、队列和事件 | `hydraserve/engine/serving_loop.py` |
| D0 完整副本 DP | `hydraserve/engine/collocated_multi.py` |
| H1 admission、路由、owner 和角色状态机 | `hydraserve/engine/multi_worker.py` |
| P/D 命令循环与本地 collocated 执行 | `hydraserve/engine/pd_service.py` |
| Long Prefill、chunk 发布、D 接收和 N−1 replay | `hydraserve/engine/pd_worker.py` |
| 模型逐层前向和 Decode transaction | `hydraserve/model/runtime.py` |
| GDN 与 causal-conv Kernel | `hydraserve/kernels/gdn.py` |
| split-K/tiled Paged Attention | `hydraserve/kernels/paged_attention.py` |
| RMSNorm、gated norm、SiLU | `hydraserve/kernels/rmsnorm.py`, `hydraserve/kernels/activation.py` |
| Paged KV 和 INT8 写入 | `hydraserve/cache/paged_kv.py`, `hydraserve/kernels/kv_cache.py` |
| FP32 state slot 和 batch workspace | `hydraserve/cache/state_pool.py` |
| 描述符和状态 payload | `hydraserve/transfer/descriptor.py`, `hydraserve/transfer/pipeline.py` |
| BF16/uint16/INT8/FP32 转换 | `hydraserve/transfer/runtime_codec.py`, `hydraserve/cache/kv_quantizer.py` |
| SHM Ring | `hydraserve/transfer/shm_ring.py` |
| Sampling 和 logprob | `hydraserve/engine/sampling.py` |

## 4. Kernel 与模型运行时

### 4.1 已实现的主要 Kernel

| 组件 | 当前实现 |
|---|---|
| RMSNorm | Triton RMSNorm 与 gated RMSNorm |
| Causal Conv | blocked Triton kernel，保留 legacy 切换 |
| GDN recurrence | Triton recurrent update，支持 blocked head mapping |
| Activation | fused SiLU-and-Mul |
| Full Attention prefill | FlashAttention varlen；不可用时使用 paged online-softmax |
| Full Attention decode | reference、split-K 或 flash paged decode |
| KV 写入 | paged scatter，含 Graph-safe INT8 batch writer |
| AWQ GEMM | grouped asymmetric INT4 unpack + GEMM |
| FP8 GEMM | E4M3FN 位解码与 block scale |
| Projection | gate-up、QKV/QKVZ、BA 等投影融合 |

### 4.2 AWQ 数据格式

当前 AWQ 路径按 4-bit nibble 解包：一个 32-bit packed word 容纳 8 个 INT4 值。解码值为 `0..15`，随后减去对应 zero point，属于 grouped asymmetric quantization。

因此，以下说法都是错误的：

- “8 个 INT4 打包进一个 uint8”；
- “权重直接使用 signed `[-7, 7]`，同时又带 asymmetric zero point”。

### 4.3 CUDA Graph 与 torch.compile

CUDA Graph decode 已经是实际运行路径，并兼容批量 INT8 KV scale 写入。历史上的 Python `.item()` host sync 问题已经修复，不应继续列为当前致命 Bug。

`torch.compile` 也已经实现，但默认关闭：

```text
HYDRASERVE_TORCH_COMPILE=0  # 默认
```

它属于可选优化，不能写成“尚未实现”，也不能写成“V4 默认启用”。历史优化日志显示，CUDA Graph 的实际收益应以测量结果为准，不能用“TPOT 必然从 100ms 降到 30–40ms”之类的预测代替实测。

### 4.4 FlashAttention 路径

当前 prefill 不只在第一个 chunk 使用 FlashAttention：

- 首个 chunk 使用 varlen FlashAttention；
- 后续带历史页的 chunk 可走 paged flash prefill；
- 无 FlashAttention 时退化为自研 tiled paged prefill。

Decode 路径由 `HYDRASERVE_PAGED_ATTENTION` 在 flash/reference/split-K 之间选择。

## 5. KV、循环状态与缓存

### 5.1 PagedKVCache

Paged KV Cache 使用物理 block 管理请求 KV：

- admission 时预留请求所需容量；
- decode 以逻辑 token 位置映射到物理 block；
- INT8/INT4 storage codec 与传输 codec 分离；
- prefix cache 命中时可以共享受保护的物理页。

INT8 KV 写入已经使用批量 CUDA kernel 写 value 和 scale，可以进入 CUDA Graph capture。

### 5.2 GpuLinearStatePool

每个活跃请求占用固定的 FP32 recurrent-state slot。Decode batch 使用 `index_select` 把请求状态收集到预分配 workspace，再在事务结束后写回。

当前空闲槽结构是 Python list：分配使用 `pop(0)`，释放后执行排序。因此它不是严格 O(1) queue；在当前容量下通常不是主要性能瓶颈，但文档不应错误标注复杂度。

### 5.3 Prefix cache

当前可用层级为：

- L1：GPU paged prefix cache；
- L2：可选 HostPrefixCache；
- Remote/L3：没有生产级远端缓存实现。

Host L2 只有在配置 `host_prefix_cache_gb > 0` 时启用；V4 配置为 0，因此 V4 结果没有使用 Host L2。

当前只缓存 Full Attention KV。GDN recurrent state 不进入 prefix cache，命中 KV 后仍需精确重算相应循环状态。

## 6. 状态传输协议

### 6.1 Descriptor

`StateTransferDescriptor` 描述请求级传输信息，包括：

- request/model/prompt metadata；
- transfer mode；
- recurrent state 的 `state_token_count`；
- streamed KV chunk ranges；
- Host prefix 命中信息；
- 一个或多个 `RegionDescriptor`。

`StateType` 枚举当前有 6 个值：

```text
FULL_ATTN_KV
SLIDING_WINDOW_KV
DSA_KV
MLA_KV
LINEAR_SSM
LINEAR_CONV
```

但当前端到端运行时真正读写的是 `FULL_ATTN_KV`、`LINEAR_SSM` 和 `LINEAR_CONV`。SWA、DSA、MLA 目前主要是协议扩展点，不能称为已完成的模型支持。

### 6.2 Transfer mode

| 模式 | 语义 |
|---|---|
| FULL | 迁移完整 KV 与 recurrent state |
| INT8/QUANTIZED | 链路传输量化后的 KV，同时迁移原始 FP32 recurrent state |
| PARTIAL | 只迁移 recurrent state，D 端重算 Full Attention KV |

Linear recurrent state 必须保持未量化 FP32。

### 6.3 Chunked transfer 与 overlap

主 PD 路径默认启用 `HYDRASERVE_CHUNKED_TRANSFER=1`：

1. D receiver 先启动；
2. P 每完成一个 prefill chunk 就发布已完成的 KV 范围；
3. transfer executor 在后台抽取和发送；
4. D 使用 install stream 接收并写入目标 KV Cache；
5. 最终 descriptor 关联已经发送的 chunk range 和 recurrent state。

GPU BF16 Cache 路径可使用 fused Triton gather/scatter staging。若源 cache 本身是 INT8，当前 `extract_kv_range` 会走逐层 read/stack fallback，所以不能把 V4 的收益归因于 BF16 fused-gather 快路径。

这种实现提供了计算与传输重叠，但“完全无干扰”并不准确：两条路径仍然会竞争 GPU 带宽、CUDA allocator、CPU executor 和 SHM ring slot，最终也需要事件/任务同步。

### 6.4 SHM ring

POSIX SHM ring 使用多个固定大小 slot 和文件锁完成多 producer claim：

```text
FREE -> WRITING -> READY -> FREE
```

producer 写完 payload 后再翻转 READY 状态，因此是“READY-last”，并不是重新原子发布完整 header。descriptor digest 用于目标键校验，不是 payload 内容哈希。

已观察到的 H2/R2 hang 应保留为已知问题，但目前不能仅凭代码断定根因就是“3×64MB ring 放不下 512MB KV”：自适应 chunk 目标远小于整段 KV。更可靠的描述是，多 producer、同一目标 D、有限 ring slot 下仍存在背压或协议并发问题，根因需要独立复现和 tracing。

## 7. 调度、抢占与容错

### 7.1 FairDecodeScheduler

当前公平分数综合：

- 已服务 token 数；
- 请求 priority；
- 等待老化；
- 接近 deadline 时的 urgency。

deadline 主要来自请求 `timeout_ms`，不是自动由 `TTFT SLO + TPOT SLO × max_tokens` 推导。

### 7.2 连续批处理

ServingLoop 按 step 组织 decode batch，并执行：

1. 选择 runnable 请求；
2. 预留/检查 KV 与 state；
3. 执行 batch decode transaction；
4. 提交 token、KV 长度和 recurrent state；
5. 对失败 batch 回滚并隔离错误请求。

### 7.3 抢占与恢复

当前 preemption 不会把完整 KV 和 recurrent state spill 到 Host。它会释放请求占用的 GPU 资源；恢复时使用：

```text
prompt + generated_tokens[:-1]
```

重新执行精确 replay，恢复可继续 decode 的模型状态。它保证逻辑 token 序列一致，但会付出重算成本。

### 7.4 Worker 故障

Multi-worker 路径包含 worker supervision、进程重建、IPC 恢复和 fail-closed 路由。故障期间是否能无损继续请求，取决于请求所处阶段以及状态是否还能 replay，不能笼统表述为“所有请求自动无损迁移”。

## 8. V4 实验结果

### 8.1 实验口径

仓库 `results/v4/SUMMARY.md` 记录的主要配置为：

- 4×RTX 3090 24GB；
- Qwen3.5-4B BF16 weights + INT8 KV Cache；
- SHM 实测约 4.58 GB/s；
- concurrency 16；
- block size 256；
- chunk size 16384；
- conditional PD threshold 4096；
- Host prefix cache 关闭；
- 汇总文件将负载描述为 120s Poisson 窗口、0.8× offered load；但结果 JSON/CLI 元数据记录的是 `arrival_pattern=burst` 并使用冻结 trace。这里存在实验元数据口径不一致，重新复现时必须先统一。

结果基线是 `a090ec9`，且 JSON 元数据记录 `git_dirty=true`。因此数据可用于说明趋势，但不是当前 `b4f778d` 干净工作树的严格可复现结果；本文表格保留仓库汇总值，不替元数据冲突作额外推断。

### 8.2 H1 与 D0

下表中的 TPOT 和 TTFT 都取自结果 JSON 的 `by_class.short`；TTFT 为服务侧首 token 延迟，不包含 `client_queue_ms`，包含排队时间的端到端口径应查看 `e2e_ttft_ms`。

| 负载 | D0 SLO | H1 SLO | D0 吞吐 | H1 吞吐 | D0 TPOT p50 | H1 TPOT p50 | D0 TTFT p50/p99 | H1 TTFT p50/p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 s42 | 16/48 | **28/48** | 101 | **108** | 133ms | **75ms** | 678/2413ms | **389**/2808ms |
| R1 s43 | 16/48 | **31/48** | 98 | **109** | 156ms | **74ms** | 590/2041ms | **386**/3213ms |
| R1 s44 | 16/48 | **29/48** | 96 | **115** | 153ms | **70ms** | 644/2546ms | **392/987ms** |
| R2 s42 | 16/48 | **33/48** | **79** | 73 | 155ms | **97ms** | 712/4279ms | **383**/7329ms |
| R2 s43 | 16/48 | **32/48** | **71** | 66 | 161ms | **76ms** | 621/2478ms | **394**/9245ms |
| R2 s44 | 16/48 | **32/48** | **78** | 75 | 116ms | **78ms** | 691/5606ms | **439/5477ms** |
| R3 s42 | 16/44 | **30/44** | 94 | **112** | 153ms | **73ms** | 646/1995ms | **385**/3253ms |

在这些结果中：

- H1 short SLO goodput 相对 D0 提高 75%～106%；
- H1 TPOT p50 相对 D0 下降 38%～52%；
- H1 short TTFT p50 在全部数据点都更低，下降约 34%～46%；
- short TTFT p99 没有一致改善：R1 s44 和 R2 s44 更低，其余数据点更高，说明 Hybrid 路径仍存在明显的尾延迟抖动；
- H1 总吞吐保持 D0 的 92%～119%，即变化范围约为 −8%～+19%；
- H1 在 R1/R3 吞吐反超，在 R2 略低于 D0。

更准确的结论是：

> 在当前四卡消费级 GPU 实验中，1H+3D 通过隔离长 prefill 干扰，提高了 short 请求的 SLO goodput，TPOT p50 下降 38%～52%，TTFT p50 下降 34%～46%；但 TTFT p99 改善不稳定。吞吐代价取决于负载，R1/R3 总吞吐提高，R2 总吞吐小幅下降。

### 8.3 四拓扑单点结果

R1 seed42 的顺序为：

| 拓扑 | short SLO | TPOT p50 | short TTFT p50/p99 | 吞吐 |
|---|---:|---:|---:|---:|
| H1（1H+3D） | **28/48** | **75ms** | **389**/2808ms | **108.0 tok/s** |
| H2（2H+2D） | 23/48 | 96ms | 423/2912ms | 94.8 tok/s |
| D0（4×DP） | 16/48 | 133ms | 678/2413ms | 100.5 tok/s |
| P0（静态2P+2D） | 11/48 | 153ms | 3064/6089ms | 81.1 tok/s |

这说明在该负载下，一个承担 Long Prefill 的 Hybrid 完整模型副本已经够用；H2 将 D-bound worker 从 3 个减为 2 个，反而降低了可用 Decode 容量。这个结论不能直接外推到不同 GPU、模型、并发和到达分布。

### 8.4 vLLM 数据说明

旧文档中的“vLLM 174.5 vs HydraServe 64 tok/s、差 2.7×”不应继续使用。仓库后来增加了 `results/vllm_v4/` 和 `results/v4_flash/`：

- R1 vLLM 三 seed 平均吞吐约 151.7 tok/s；
- 对应旧 HydraServe D0 三 seed 平均约 98 tok/s；
- Flash decode 的 HydraServe D0 seed42 为约 121.9 tok/s。

这些结果表明 HydraServe 的 decode 性能仍有明显优化空间，但不同 runner 的 arrival pattern、trace 重放、量化、warmup 和并发口径必须完全对齐后，才能给出正式的框架间倍数结论。

## 9. 功能成熟度

### 9.1 主路径已经实现并被当前实验使用

- Hybrid worker 状态机；
- 条件式 PD（conditional PD）与工作守恒调度（work-conserving scheduling）；
- 接收方先启动的分块 SHM 传输（receiver-first chunked transfer）；
- Full KV + FP32 recurrent state 迁移；
- INT8 GPU KV Cache；
- Flash/paged prefill 与 paged decode；
- CUDA Graph decode；
- fused model projections；
- continuous batching、admission 与 replay。

### 9.2 已实现但默认关闭或 V4 未使用

- `torch.compile`；
- HostPrefixCache；
- INT8 wire transfer；
- ZMQWaveRouter；
- cost-aware router 的完整标定闭环；
- BF16 cache 路径的 fused staging 快路径。

### 9.3 协议或接口脚手架

- TP rank/world-size metadata 已进入 descriptor，但生产路径没有完成真实 TP state slicing；
- `StateType` 有 SWA/DSA/MLA 枚举，但运行时尚无完整端到端实现；
- DP Graph token-count 同步存在，但 padding 时要求 backend 实现 `decode_padded`，当前内置 backend 尚未提供该方法；
- bootstrap/control plane 已集成到部分路径，但不等于生产级多节点控制面。

### 9.4 尚未完成

- 运行时动态增加或移除 worker；
- 根据实时负载自动闭环调整 P:D 数量；
- 生产级 RDMA/NIXL/Mooncake 跨机传输；
- 完整多机容错和远端 L3 prefix cache。

## 10. 已知限制与风险

1. 当前是研究原型，默认路径和可选实验路径的成熟度不同。
2. V4 结果来自 dirty worktree，必须重新固化 commit、镜像、模型 hash 和命令行才能严格复现。
3. H2/R2 多 producer SHM hang 尚未完成根因定位。
4. Host L2、INT8 wire transfer、TP metadata 和 DP Graph sync 缺少同等强度的端到端实卡验证。
5. CUDA Graph 能降低 launch overhead，但不能替代 GEMM、attention 和状态搬运本身的性能优化。
6. 消费级单机 SHM 结果不能直接外推到百卡 RDMA 部署。
7. 不同框架对比必须统一模型权重、KV dtype、prompt/output trace、到达时间、并发、采样参数和 warmup。

## 11. 后续工作优先级

### P0：正确性与复现

1. 固化一套 clean-commit V4/V5 benchmark bundle；
2. 为 H2/R2 hang 增加 ring-slot、producer、consumer 和 chunk 生命周期 tracing；
3. 增加 PD 首 token 返回时序测试，明确 transfer/install 对 TTFT 的占比；
4. 为每种可选路径记录是否实际进入 fast path，而不是只记录配置值。

### P1：Decode 性能

1. 在统一 trace 下完成 split-K、flash decode、CUDA Graph 和 compile 的逐项消融；
2. 继续减少小 batch 的 Python dispatch 和 kernel launch；
3. 优化 INT8 cache 下的 KV range gather，避免逐层 read/stack fallback；
4. 分离 attention、GDN、GEMM、sampling 和状态事务的耗时统计。

### P2：传输与拓扑

1. 完成 INT8 wire transfer 的端到端实测；
2. 为 SHM ring 引入明确的 credit/backpressure 和 timeout diagnostics；
3. 完成真正的 TP state slicing 与多 rank 验证；
4. 接入生产级跨机 transport，并保留现有 descriptor/codec 边界。

### P3：调度与缓存

1. 补全 backend `decode_padded` 后再进行 DP Graph sync 多 rank 验证；
2. 对 HostPrefixCache 做容量、命中率和 PCIe 成本消融；
3. 建立在线 P:D 建议器，先输出观测建议，再考虑自动扩缩；
4. 对 replay 抢占成本建立预算，避免高频抢占放大重算。

## 12. 关键代码与数据索引

| 领域 | 路径 |
|---|---|
| 模型配置 | `hydraserve/config.py` |
| 模型运行时 | `hydraserve/model/runtime.py` |
| GDN Kernel | `hydraserve/kernels/gdn.py` |
| Paged Attention | `hydraserve/kernels/paged_attention.py` |
| KV 写入 | `hydraserve/kernels/kv_cache.py` |
| AWQ / FP8 | `hydraserve/kernels/awq.py`, `hydraserve/kernels/fp8.py` |
| Paged KV Cache | `hydraserve/cache/paged_kv.py` |
| Recurrent state pool | `hydraserve/cache/state_pool.py` |
| Prefix cache | `hydraserve/cache/prefix_cache.py`, `hydraserve/cache/host_prefix_cache.py` |
| Hybrid 调度 | `hydraserve/engine/multi_worker.py` |
| P/D worker | `hydraserve/engine/pd_worker.py` |
| Serving loop | `hydraserve/engine/serving_loop.py` |
| 连续批处理 | `hydraserve/engine/continuous_batching.py` |
| 公平调度 | `hydraserve/engine/fair_scheduler.py` |
| Descriptor | `hydraserve/transfer/descriptor.py` |
| Runtime codec | `hydraserve/transfer/runtime_codec.py` |
| Transfer pipeline | `hydraserve/transfer/pipeline.py` |
| SHM ring | `hydraserve/transfer/shm_ring.py` |
| Fused staging | `hydraserve/kernels/staging.py` |
| V4 汇总 | `results/v4/SUMMARY.md` |
| vLLM 对照 | `results/vllm_v4/` |
| Flash decode 快测 | `results/v4_flash/` |
| 历史优化记录 | `docs/OPTIMIZATION_LOG_2026-08-22.md` |

## 13. 阅读口径

阅读或更新本文档时，所有功能应明确标注为以下四种状态之一：

1. **主路径已使用**：代码存在，默认/实验配置真正进入，并有端到端结果；
2. **可选实现**：代码存在，但默认关闭或当前 benchmark 没有使用；
3. **接口脚手架**：descriptor、registry 或 unit test 存在，但缺少生产路径闭环；
4. **尚未实现**：只有设计目标或 TODO。

性能结论必须同时记录：commit、dirty 状态、GPU、模型/权重 hash、KV storage dtype、wire transfer dtype、trace hash、arrival pattern、并发、cache/state 容量、warmup 和完整命令行。缺少这些信息时，只能把结果视为探索性数据。
