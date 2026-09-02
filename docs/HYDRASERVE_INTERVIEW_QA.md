# HydraServe 项目面试准备：难点、成果、改善举措与问答

> 代码审阅基线：`b4f778d`。
>
> V4 结果基线：`a090ec9`，结果元数据为 `git_dirty=true`。
>
> 本文用于简历讲解和技术面试准备；项目事实以当前代码、`hydarserve.md` 和原始结果 JSON 为准。

## 1. 推荐的简历项目描述

### 1.1 项目概述

面向 Qwen3.5/Qwen3.6 GDN+GQA 混合注意力架构，针对 PD 分离中 Full Attention KV 与 GDN recurrent state 双状态一致性复杂、长 Prefill 严重干扰 Decode 的问题，从零实现覆盖自研 Triton Kernel、异构状态管理与传输、连续批处理及动态 Hybrid 角色调度的推理引擎；完成 4B 多 GPU 端到端压测，并在 9B BF16、27B AWQ/FP8 路径验证模型执行能力。实测长 Prefill 可使 Decode TPOT 放大 2.5～6.4×；在 4×RTX 3090 的三类负载中，动态 Hybrid 相比 4×DP 将 Short SLO goodput 提升 75%～106%。

### 1.2 技术工作

- **混合注意力执行与算子优化**：实现 GDN 增量规则（delta rule）递推、因果卷积（causal convolution）、Paged Attention K维分片（split-K）/分块复用预填充（tiled prefill）、分页KV散写（Paged KV scatter）等 Triton Kernel；通过跳过中间分块（chunk）的无效 LM Head、只计算末位置 logits，使 32K Prefill 从约 268.3s 降至 11.3s，提速 23.8×。
- **双状态管理与一致性协议**：实现 Paged KV Cache 与固定槽 FP32 recurrent-state pool；通过 `state_token_count`、N−1 state 边界、最后一个 prompt token replay 和 P/D 首 token 校验，保证 KV 与循环状态在跨 GPU 迁移后的精确一致性。
- **分块传输与计算重叠**：实现接收端先启动（receiver-first）、页锁定暂存缓冲区（Pinned staging buffer）、分块共享内存环形队列（SHM Ring）、后台传输执行器（transfer executor）和独立 CUDA 安装流（install stream），使已完成分块的 KV 抽取、传输和安装与后续 Prefill 重叠；4B 8K 完整 BF16 状态迁移约 66ms。
- **动态 Hybrid 角色调度**：设计 `DECODE→PREFILL_PENDING→PREFILL_ACTIVE→DECODE` 角色状态机、条件式PD路由（conditional PD）和工作守恒调度（work-conserving scheduling）；Hybrid 卡空闲时参与 Short/Decode，Long 到达时切换到 Prefill，并在 Long 分块边界有界穿插 Short 操作，改善静态 PD 的 P worker 空置和 Decode 容量损失。
- **系统化压测**：在 4×RTX 3090、客服 RAG、文档摘要、代码分析负载下，相比 4×DP，H1 的 Short SLO goodput 提升 75%～106%，TPOT p50 下降 38%～52%，TTFT p50 下降 34%～46%，总吞吐保持 92%～119%；TTFT p99 仍存在抖动，是后续优化重点。

### 1.3 简历中应避免的表述

- 不要说“完整 53MB 状态全部驻留 SRAM”；应说“每个 Triton program 的 recurrent tile 在 token 循环期间驻留寄存器/片上存储”。
- 不要把 INT8 KV Cache 和 INT8 链路传输混为一谈；V4 是 INT8 Cache、BF16链路传输（wire transfer）。
- 不要把逻辑角色切换描述成运行时增加或删除物理 GPU。
- 不要声称 32K 的 23.8× 全部来自某一个 Kernel；它主要来自 logits 裁剪后能够使用大 chunk 和 FlashAttention。
- `21.2s→529ms` 在当前仓库中缺少可定位的原始结果；补齐 commit、脚本和日志前不建议放进简历。
- “静态 PD SLO 仅 23%”是 R1 seed42 单点结果，不是所有负载的统一结论。

## 2. 一分钟项目介绍

> HydraServe 是我针对 Qwen3.5/Qwen3.6 GDN+GQA 混合注意力架构，从零实现的推理引擎。与普通 Transformer 不同，这类模型同时存在随上下文增长的 Full Attention KV Cache，以及固定大小的 GDN FP32 recurrent state，所以 PD 分离不能只搬 KV，还必须解决双状态边界和一致性问题。
>
> 我的工作分为三层：底层实现 GDN recurrence、Paged Attention、KV scatter 等 Triton Kernel；中间层实现 Paged KV Cache、recurrent state pool 和分块状态传输；调度层实现 continuous batching、conditional PD 和动态 Hybrid worker。Hybrid 卡空闲时参与 Short Decode，Long 请求到达后切换为 Prefill，从而隔离长 Prefill 对 Decode 的干扰，同时避免静态 P 卡长期空置。
>
> 在四张 RTX 3090、Qwen3.5-4B 和三类负载下，H1 相比 4×DP 将 Short SLO goodput 提升 75%～106%，TPOT p50 下降 38%～52%，总吞吐保持在 92%～119%。

## 3. 单请求端到端完整链路

这一节从一个 OpenAI-compatible 请求进入服务开始，一直追踪到最终 token 返回客户端。必须先区分“物理 GPU 上有什么”“worker 当前扮演什么角色”“一个请求采用哪条执行路径”。

### 3.1 先纠正 Decode 池、D 卡和 DP 的术语

HydraServe 当前没有“只能做 Decode、没有 Prefill 能力的残缺模型卡”。相关 GPU worker 都加载完整模型权重，只是调度角色不同。

| 名称 | 实际含义 | 能否本地 Prefill | 能否 Decode | 是否产生 Long PD 状态 |
|---|---|---:|---:|---:|
| D0 的 DP worker | `MultiGPUCollocatedBackend` 中的完整模型副本，一个请求固定绑定一个副本 | 是 | 是 | 否 |
| H1 的 D-bound worker | `decode_devices` 对应的完整模型副本；持有请求最终 KV/state | Short 可以 | 是 | 通常不作为 Long 的 P producer |
| H1 的 Hybrid worker | `prefill_devices` 对应的完整模型副本；逻辑角色可切换 | 是 | 空闲时可以 | Long 到达时可以 |
| Decode pool | 当前可接受 Decode/Short 的逻辑 worker 集合 | 取决于成员 | 是 | 不是一个物理或 NCCL pool |

因此：

- **D0 的四张卡确实是请求级数据并行副本**：每张卡都有完整权重，各自独立处理不同请求，不做 Tensor Parallel collective。
- **H1 中的四张卡在模型存储层面也都是完整副本**，但调度层不再完全对称。
- “D 卡”更准确的说法是 **D-bound full-model worker**：它是 Long PD 请求的状态接收方和后续 Decode owner，但仍能为 Short 请求执行本地 Prefill+Decode。
- “常驻 D”只表示该 worker 不切换成 Long 请求的远端 Prefill producer，不表示它完全不能计算 Prefill。
- “Decode pool”是调度视角：包括所有 D-bound worker，以及当前处于 `DECODE` 角色、可服务 Short/Decode 的 Hybrid worker。
- 它不是 PyTorch `DistributedDataParallel` 训练进程组，也没有 all-reduce；更准确地说是 **请求级 DP / full-replica serving**。

CLI 的“默认”也需要分清：

- 不提供多GPU拓扑参数时，是单GPU collocated，不是4×DP；
- 显式提供 `--dp-devices` 时，使用 `MultiGPUCollocatedBackend`，这才是D0的完整副本DP；
- 使用 `--adaptive --decode-devices ... --prefill-devices ...` 时，进入`MultiWorkerGenerationBackend`，物理副本仍然完整，但请求被区分为collocated与PD路径，并引入Hybrid角色状态机。

实验拓扑 `H1=1H+3D` 可以继续使用，但第一次出现时应解释为：

> 1 个可切换角色的完整模型副本，加 3 个以状态承接和 Decode 为主、同时支持 Short collocated Prefill 的完整模型副本。

### 3.2 一个请求可能走的四条路径

```text
HTTP request
    |
    v
tokenize + bounded admission
    |
    v
worker capacity reservation + route
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

前三条是同卡混部（collocated）路径：Prompt 和后续 Decode 在同一完整模型副本上完成，不需要跨 GPU 状态迁移。第四条才是真正的预填充-解码分离（PD-disaggregated）Long路径。

### 3.3 HTTP、Tokenize 与请求入队

请求首先进入 `/v1/completions` 或 `/v1/chat/completions`：

1. Chat 请求先通过 tokenizer chat template 渲染成 prompt 文本；
2. tokenizer 把字符串编码成 Python `list[int]`；
3. 校验 `max_tokens`、context limit、sampling 参数、priority、timeout 和 stream；
4. `ContinuousGenerationLoop.submit` 创建 `ServingRequest`；
5. 如果用户没有提供 seed，系统生成一个请求级 seed；
6. admission queue 按请求数和 `prompt_tokens + max_new_tokens` 两个维度做有界控制；
7. 返回 `GenerationHandle`，HTTP线程随后阻塞读取或流式消费 `GenerationEvent`。

此时数据仍主要在 CPU：

```text
prompt text: UTF-8 string
token_ids: Python int tuple
sampling metadata: Python dataclass
```

### 3.4 准入控制（Admission）、容量预留与路由

GenerationLoop从输入队列（incoming queue）取出请求后，先做worker级准入控制（admission），而不是立即发射模型Kernel。

每个候选 worker 至少检查：

- 是否有足够 KV block；
- 是否有空闲 recurrent-state slot；
- 是否健康；
- Full Attention KV页的prefix命中量（当前只作为存储共享和路由亲和性，不能视为完整混合状态命中）；
- 当前 Decode load；
- Long 请求是否有可用 Hybrid Prefill worker。

通常为请求预留：

```text
prompt_tokens + max_new_tokens - 1
```

个逻辑 KV token。减一的原因是：最后一个采样出的 output token 如果已经触发停止条件，就不需要再作为下一步模型输入写入 KV。

在 H1 conditional 模式中：

1. **Short 请求**：长度小于阈值。空闲 Hybrid 和 D-bound worker 根据实时 Decode load 共同竞争；请求可能绑定其中任意一个完整模型副本，然后走 collocated 路径。
2. **Long 请求**：长度大于等于阈值。先选择并预留一个 D-bound worker，建立不可变的 Decode owner；再绑定一个 Hybrid worker，并把其角色从 `DECODE` 切到 `PREFILL_PENDING`。
3. **Prefill worker 不可用**：已预留的 D-bound worker可回退到 collocated Prefill，而不是泄漏 reservation 或直接丢请求。

### 3.5 Prompt 在模型内部如何逐层计算

以 Qwen3.5-4B BF16 路径为例。Prompt token 在目标 worker 上转换为：

```text
input_ids: torch.int64 [batch, sequence]
```

随后执行 embedding lookup，得到 BF16 hidden state：

```text
hidden: [batch, sequence, 2560], BF16
```

模型共有 32 层，按固定顺序交替出现 24 个 Linear Attention 层和 8 个 Full Attention 层。每层都有两段 residual：

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

RMSNorm Kernel 把输入转换到 FP32 计算平方和、方差、`rsqrt` 和权重乘法，再存回与输入相同的 BF16 dtype。

#### 3.5.1 Linear Attention / GDN 层

一层 GDN 的 Kernel 级路径为：

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

blocked causal-conv Kernel 按 `(batch×token, channel_tile)` 并行，一个 program 处理一段 channel。随后 reshape 为：

```text
Q/K: [B,T,16,128]
V:   [B,T,32,128]
```

GDN value heads 是 key heads 的两倍；blocked Kernel通过 `HEAD_RATIO` 映射共享对应 Q/K，而不需要实际 `repeat_interleave`。

GDN recurrent Kernel 按：

```text
grid = (batch, value_head, value_dim_tile)
```

并行。每个 program 载入一个 `[key_dim, BLOCK_V]` FP32 state tile，然后在 token 维顺序执行：

```text
S'ₜ = exp(gₜ) · Sₜ₋₁
pₜ  = kₜᵀS'ₜ
δₜ  = βₜ(vₜ-pₜ)
Sₜ  = S'ₜ+kₜδₜᵀ
oₜ  = qₜᵀSₜ
```

token 之间存在真实依赖，不能并行；并行度来自 batch、head 和 value tile。state tile 在本 program 的 token 循环期间停留在寄存器/片上存储，最后统一写回 global memory。

GDN core 随后进入 fused gated RMSNorm：FP32累加归一化，同时计算 `SiLU(gate)`，输出回BF16；最后执行 output projection 回到 hidden size。

#### 3.5.2 Full Attention / GQA 层

Full Attention 层首先执行 fused QKV projection：

```text
Q + output_gate: [B,T,16,512]
K:               [B,T,4,256]
V:               [B,T,4,256]
```

Q 的最后维度拆成真实 Query 与 output gate：

```text
Query:       [B,T,16,256]
Output gate: [B,T,16,256]
```

然后依次执行：

1. Q/K RMSNorm，内部 FP32 accumulate、输出 BF16；
2. 对部分 rotary dim 应用 RoPE；
3. 把新 K/V 写入该请求的 Paged KV Cache；
4. 运行 Attention；
5. `attention × sigmoid(output_gate)`；
6. output projection 回到 hidden size。

Paged KV 写入时，逻辑 token 位置先通过 block table 映射到物理 block/offset。V4 的 `kv_quant=int8` 路径会针对每个 token、每个 KV head：

```text
BF16 K/V -> FP32 absmax -> FP32 scale
          -> round/clamp -> INT8 K/V
```

最终 GPU Cache 保存：

```text
K/V payload: INT8
K/V scale:   FP32, per-token-per-head
```

Attention 读取 INT8 Cache 时，当前实现先通过 `int8.float() × scale` 恢复为 BF16 page tensor，再进入 Flash/paged attention。这是正确但仍有优化空间的路径。

不同阶段选择不同 Attention Kernel：

- 首个多 token chunk：FlashAttention varlen；
- 带历史的 continuation chunk：paged flash prefill；
- Flash不可用时：自研 tiled paged online-softmax；
- 单 token Decode：默认 split-K paged attention，也可切换 flash/reference。

split-K Decode把长 context 分片，每个 split 计算局部 `maximum/denominator/accumulator`，reduce Kernel再用online-softmax缩放合并，避免单个program串行扫描全部历史。

#### 3.5.3 MLP、最终投影和第一个 token

每层 Attention/GDN 后都执行 MLP：

```text
BF16 hidden
 -> fused gate_up GEMM
 -> Triton SiLU(gate) × up
    （内部FP32计算，输出BF16）
 -> down projection
 -> residual add
```

32层完成后：

1. final RMSNorm；
2. LM Head只投影最后一个Prompt位置；
3. 默认得到BF16 logits `[1,vocab]`，也可通过环境变量强制FP32 logits；
4. greedy且无penalty时直接对原生dtype做batched argmax；
5. temperature/top-k/top-p/min-p或penalty路径先把单行logits转FP32，再做过滤、softmax和multinomial；
6. 得到第一个output token `y0`。

### 3.6 Long PD 请求的 N−1、迁移与 D 端恢复

假设 Prompt 为 `x0...xN-1`，第一个生成 token 为 `y0`。真实 Long PD 流程如下：

1. Coordinator先在目标 D-bound worker预留KV blocks和FP32 state slot。
2. Coordinator向D派发`prepare`，确认receiver已进入接收路径。
3. Hybrid角色从`PREFILL_PENDING`进入`PREFILL_ACTIVE`。
4. P为完整N个Prompt token分配本地Paged KV。
5. P先计算`x0...xN-2`；每个chunk完成后，回调发布对应KV逻辑范围。
6. transfer stream等待计算stream的CUDA Event，确保该chunk的KV已经写完。
7. P从Paged Cache抽取chunk，复制到Pinned CPU buffer，再写入SHM Ring。
8. D后台receiver读取chunk，并在独立install stream中安装到D端Paged Cache。
9. P在N−1边界提取GDN state：SSM和Conv统一转换为CPU FP32连续数组。
10. P继续计算最后一个Prompt token `xN-1`，生成其KV和最终logits。
11. `xN-1`的KV也被发送；因此N−1不是“只传N−1个KV token”。
12. P从最终logits采样得到`y0`，把`first_token_id=y0`写入descriptor。
13. P发送最终bundle：descriptor + FP32 recurrent state；streamed KV只在descriptor中记录range，不重复发送。
14. D确认所有KV range完整，安装FP32 recurrent state，并把`sequence_length`设置为N−1。
15. D用`xN-1`执行一次`runtime.forward`：读取已安装历史KV，推进GDN state到N，并重新得到一份首token logits；该forward也会在相同逻辑位置重新写入/覆盖`xN-1`的KV。
16. D使用相同sampling参数验证首token是否与P侧`y0`一致。
17. D把`y0`登记为该请求第一个generated token，请求状态转为READY。
18. Coordinator等待P结果和D prepare结果均完成，再把`y0`交给ServingLoop。

因此，P/D之间有两种不同的“最后token处理”：

- P计算`xN-1`是为了得到完整KV和第一个采样token；
- D replay `xN-1`是为了把收到的N−1 recurrent state推进到N，并做一致性验证。

### 3.7 V4 路径中的完整 dtype 变化

| 阶段 | dtype / 表示 | 说明 |
|---|---|---|
| HTTP prompt | UTF-8 string | CPU |
| tokenizer输出 | Python int tuple | CPU |
| 模型输入 | `torch.int64` | GPU或embedding所在device |
| BF16权重 | BF16 | 4B V4主路径 |
| hidden/residual | BF16 | GEMM和层间主dtype |
| RMSNorm/Silu内部 | FP32 accumulate | 输出重新存为BF16 |
| Q/K/V projection | BF16 | RoPE后仍为模型dtype |
| P/D KV Cache payload | INT8 | V4 `kv_quant=int8` |
| KV scale | FP32 | per-token-per-head |
| Attention输入 | INT8×FP32 scale→BF16 | softmax统计量通常FP32 |
| GDN beta/decay | FP32 | gating Kernel输出 |
| GDN SSM state | FP32 | 递推和state pool均为FP32 |
| Conv state | 协议和state pool为FP32 | fresh P RuntimeState初始buffer可随hidden dtype创建，codec发送前统一`.float()` |
| P侧FULL wire KV | BF16 raw bits视作`uint16` NumPy | 无数值转换，仅reinterpret便于SHM传输 |
| P侧wire recurrent | NumPy FP32 | Pinned D2H后进入SHM |
| SHM Ring | bytes + metadata | 固定slot数据面 |
| D收到FULL KV | `uint16`→reinterpret BF16 | 不是整数数值转换 |
| D端INT8 Cache安装 | BF16→FP32 scale+INT8 | V4会再次量化到目标cache |
| D端state pool | FP32 | SSM/Conv固定槽 |
| LM Head logits | 默认BF16 | `HYDRASERVE_FP32_LOGITS=1`时为FP32 |
| 随机采样（stochastic sampling） | FP32分数/概率 | 贪心快路径（greedy fast path）可直接argmax BF16 |
| 输出token | Python int | 进入GenerationEvent和tokenizer |
| 输出文本 | 增量UTF-8 string | SSE或最终JSON |

如果使用其他权重格式：

- AWQ权重保持packed INT4+scale+zero-point，`awq_linear`在GEMM中即时解包，输出回activation dtype；
- FP8权重保持E4M3FN位模式+block scale，Kernel解码后参与矩阵乘，输出回activation dtype；
- 这两种是权重dtype变化，不等同于KV Cache或wire transfer量化。

V4 FULL传输的关键细节是：

```text
P INT8 Cache
 -> 读取时反量化为BF16
 -> BF16 raw bits作为uint16进入Pinned Buffer/SHM
 -> D reinterpret为BF16
 -> D按自身kv_quant重新量化为INT8+FP32 scale
```

因此当前链路有一次P侧反量化和一次D侧重量化；实现直接INT8 Cache到INT8 Cache的零中间BF16传输，是后续优化方向。

### 3.8 后续每一个 Decode token 如何生成

ServingLoop把READY请求放入active集合。每一轮：

1. `FairDecodeScheduler`根据已服务token、优先级（priority）、等待老化（aging）和截止期限紧迫度（deadline urgency）选出请求；
2. MultiWorker backend按不可变owner把请求分组：D-bound worker一组、仍绑定Hybrid的collocated Short一组；
3. 不同物理worker的Decode batch可以并行发射，不需要等待其他GPU组成同宽batch；
4. 每个worker先为组内每个请求增长一个KV逻辑token；
5. 输入是每个请求上一步刚生成的token。例如第一次Decode输入`y0`；
6. `GpuLinearStatePool.batch`使用`index_select`把各请求FP32 SSM/Conv slot收集到预分配workspace；
7. `runtime.decode_batch`执行32层单token前向；
8. Full Attention层把`y0`对应的新K/V写入Paged Cache，再读取`prompt+y0`历史执行split-K/flash attention；
9. GDN层原地推进SSM和Conv state；
10. 所有层和LM Head成功后，事务才提交pooled recurrent state并推进`sequence_length`；
11. 从新logits采样得到`y1`；
12. worker把`y1`写入请求generated history并返回Coordinator；
13. ServingLoop把`y1`包装成`GenerationEvent`；
14. HTTP层用IncrementalTextDecoder把token增量解码为文本并通过SSE发送。

随后重复：

```text
输入y0 -> 写入y0的KV/state -> logits -> 采样y1
输入y1 -> 写入y1的KV/state -> logits -> 采样y2
...
```

注意：采样得到的最新token在当轮还没有进入KV；只有它作为下一轮输入时，才产生自己的K/V和GDN状态更新。

### 3.9 停止、释放与客户端可见结果

每次产生token后，ServingLoop检查：

- EOS；
- token级stop sequence；
- `max_new_tokens`；
- timeout或cancel。

触发停止后：

1. 请求从active集合移除；
2. release放到独立executor，避免阻塞Decode关键路径；
3. owner worker释放Paged KV block、prefix引用和FP32 state slot；
4. GenerationHandle收到finished event；
5. 非流式请求汇总所有可见token并解码为完整文本；
6. 流式请求逐token发送SSE，stop/EOS对应token会按API语义从最终可见文本中裁剪。

### 3.10 面试中如何概括这条链路

> 请求进入后先tokenize并进入有界队列，admission会在一个完整模型副本上预留KV block和GDN state slot。Short请求绑定D-bound或空闲Hybrid副本走本地Prefill+Decode；Long请求预留D owner后绑定Hybrid P。模型内部每层先RMSNorm，再走GQA Full Attention或GDN recurrence，最后经过MLP和LM Head。V4的hidden是BF16，GDN state是FP32，KV在GPU中是INT8+FP32 scale；Long PD传输时P把KV反量化成BF16 raw bits，通过Pinned Buffer和分块SHM发送，D安装后重新量化到INT8 Cache，同时导入N−1 FP32 state并replay最后一个Prompt token。首token一致后，请求进入owner worker的Continuous Decode batch，每一步把上一个生成token写入KV和GDN state、计算logits、采样下一个token，直到EOS、stop或长度上限，再异步释放KV和state。

### 3.11 端到端代码路径索引

| 阶段 | 主要代码 |
|---|---|
| HTTP解析与tokenize | `hydraserve/api/server.py` |
| 请求对象、队列与事件 | `hydraserve/engine/serving_loop.py` |
| D0完整副本DP | `hydraserve/engine/collocated_multi.py` |
| H1 admission、路由、owner绑定与角色状态机 | `hydraserve/engine/multi_worker.py` |
| P/D进程命令循环和本地collocated执行 | `hydraserve/engine/pd_service.py` |
| Long Prefill、chunk发布、D接收和N−1 replay | `hydraserve/engine/pd_worker.py` |
| 模型逐层前向和Decode transaction | `hydraserve/model/runtime.py` |
| GDN和causal-conv Kernel | `hydraserve/kernels/gdn.py` |
| split-K/tiled Paged Attention | `hydraserve/kernels/paged_attention.py` |
| RMSNorm、gated norm、SiLU | `hydraserve/kernels/rmsnorm.py`, `hydraserve/kernels/activation.py` |
| Paged KV存储和INT8写入 | `hydraserve/cache/paged_kv.py`, `hydraserve/kernels/kv_cache.py` |
| FP32 state slot和batch workspace | `hydraserve/cache/state_pool.py` |
| descriptor和状态payload | `hydraserve/transfer/descriptor.py`, `hydraserve/transfer/pipeline.py` |
| BF16/uint16/INT8/FP32转换 | `hydraserve/transfer/runtime_codec.py`, `hydraserve/cache/kv_quantizer.py` |
| SHM Ring | `hydraserve/transfer/shm_ring.py` |
| sampling和logprob | `hydraserve/engine/sampling.py` |

### 3.12 核心术语中英文对照

正文和面试回答统一优先使用中文名称，第一次出现时保留英文原词，方便对应论文、代码和命令行。

| 英文术语 | 推荐中文名称 | 在本项目中的含义 |
|---|---|---|
| work-conserving scheduling | 工作守恒调度 / 非空转调度 | 只要存在可执行请求，就尽量不让可用GPU空闲；Hybrid无Long时服务Short/Decode |
| collocated | 同卡混部 / 同卡执行 | 同一完整模型副本完成Prompt预填充和后续解码，不迁移状态 |
| disaggregated PD | 预填充-解码分离 | P执行预填充，状态传给D后继续解码 |
| conditional PD | 条件式PD路由 | 根据Prompt长度或成本模型决定走同卡路径还是PD路径 |
| admission control | 准入控制 | 在执行前检查并预留KV块、状态槽和worker容量 |
| bounded admission queue | 有界准入队列 | 同时限制排队请求数和排队token数 |
| request-level data parallelism | 请求级数据并行 | 每个GPU持有完整模型，不同请求绑定不同副本独立执行 |
| full-model replica | 完整模型副本 | 持有完整模型权重、KV Cache和GDN状态能力的worker |
| D-bound worker | D侧绑定完整模型副本 / 解码归属worker | Long状态的接收方和后续解码owner，也可为Short做同卡预填充 |
| Hybrid worker | 混合角色worker | 可在解码可用角色和Long预填充角色之间切换的完整模型副本 |
| Decode pool | 逻辑解码池 / 解码可用worker集合 | 当前能够接受Short或Decode工作的worker集合，不是NCCL进程组 |
| request owner | 请求归属worker / 状态owner | 保存该请求KV和GDN状态并负责后续Decode的worker |
| state machine | 状态机 / 角色状态机 | `DECODE→PREFILL_PENDING→PREFILL_ACTIVE→DECODE` |
| receiver-first | 接收端先启动 | 先启动D接收方，再允许P发送分块，避免环形队列被填满后死锁 |
| chunk / chunked prefill | 分块 / 分块预填充 | 把长Prompt切成多个可调度、可迁移的执行单元 |
| chunked transfer | 分块传输 | 每个Prompt分块完成后立即发送对应KV范围 |
| computation-communication overlap | 计算-通信重叠 | P计算后续分块时并行发送、接收和安装前一分块 |
| staging buffer | 暂存缓冲区 | 分页GPU KV与连续传输payload之间的中间缓冲区 |
| pinned buffer / pinned memory | 页锁定缓冲区 / 页锁定内存 | 支持异步GPU与CPU复制的不可换页主机内存 |
| transfer executor | 传输执行器 | 后台抽取、复制和发送KV的线程执行器 |
| install stream | 安装流 | D端用于异步安装KV和状态的独立CUDA Stream |
| backpressure | 背压 | consumer处理速度不足导致producer因无空闲slot而暂停 |
| manifest | 传输清单 | 在正式payload前发布的请求、分块范围和传输模式元数据 |
| wire transfer / wire dtype | 链路传输 / 链路数据类型 | P到D数据面真正发送的格式，与GPU Cache格式相互独立 |
| fast path | 快路径 | 满足特定dtype、shape和设备条件时使用的高性能实现 |
| fallback path | 回退路径 | 快路径条件不满足时使用的兼容实现 |
| prefix cache | 前缀缓存 | 复用相同Prompt前缀对应的Full Attention KV物理页 |
| state pool / state slot | 状态池 / 状态槽 | 保存每个活跃请求FP32 SSM与Conv状态的固定槽结构 |
| split-K | K维分片 / 上下文分片 | 将长KV序列分成多个split并行计算，再归并softmax统计量 |
| tiled prefill | 分块复用预填充 | 多个query共享同一KV tile，减少历史KV重复读取 |
| online softmax | 在线Softmax / 流式Softmax | 通过运行最大值、分母和累加量分块计算稳定softmax |
| continuous batching | 连续批处理 | 每个Decode step动态加入、移除和组合活跃请求 |
| short budget | 短请求预算 | 每个Long分块边界最多允许插入的Short操作数量 |
| aging | 等待老化 | 请求等待越久，调度优先级逐步提高，避免饥饿 |
| deadline urgency | 截止期限紧迫度 | 请求接近timeout deadline时增加调度优先级 |
| preemption / replay | 抢占 / 重放恢复 | 释放请求GPU状态，恢复时通过历史token重新计算 |
| graph-safe | CUDA图安全 | 不包含host sync或动态行为，可被CUDA Graph稳定捕获 |
| kernel grid | Kernel启动网格 / 并行程序网格 | 一次Kernel启动的CUDA thread block或Triton program数量及排列方式 |
| Triton program | Triton程序实例 | 一个Grid坐标对应的并行执行实例，内部通常包含多个warp，不等于单个线程 |
| tile | 数据分块 / 计算块 | 一个program一次加载和处理的局部Tensor区域 |
| warp | 线程束 | NVIDIA GPU上按同一指令协同执行的一组32线程 |
| occupancy | SM驻留率 / 并发驻留度 | 一个SM可同时驻留的warp或program比例，受寄存器和共享内存等资源限制 |
| register pressure | 寄存器压力 | 单个program使用过多寄存器导致驻留并行度下降或发生spill |
| shape bucket | 形状分桶 | 将相近Batch或Block Table宽度Padding到少数固定shape以复用CUDA Graph |
| transactional commit | 事务式提交 | 计算全部成功后才把workspace中的新状态写回请求owner，失败时保持旧状态 |
| dirty worktree / dirty commit | 非干净工作树 / 含未提交修改的实验版本 | 结果不是由可唯一定位的clean commit产生 |
| goodput | 有效吞吐 / 达标吞吐 | 在单位时间或请求集合中满足SLO的有效工作量 |

代码标识符不翻译，例如`PREFILL_PENDING`、`HYDRASERVE_CHUNKED_TRANSFER`、`MultiWorkerGenerationBackend`应保持原样；口述时再补充对应中文含义。

## 4. 系统与 Kernel 级设计全景

这一节不再按“实现了哪些功能”罗列代码，而是按统一的工程分析方法说明：面对什么约束、选择了什么数据布局和并行方式、为什么没有采用更直接的方案、代价是什么、失败时如何回退。项目的完整性来自这些层次共同闭环，而不是某一个 Kernel 或某一个调度策略。

### 4.1 框架分层与资源所有权

HydraServe 已经形成端到端推理引擎闭环，只是没有沿用 vLLM 的类名。主路径可以划分为：

```text
OpenAI-compatible HTTP与Tokenizer
                |
                v
ContinuousGenerationLoop
有界队列、准入、公平Decode、停止与异步释放
                |
                v
GenerationBackend
单卡 / 请求级DP / MultiWorker Hybrid
                |
                v
绑定单GPU的Worker进程
RPC、容量、角色、故障恢复
                |
                v
QwenTextRuntime（相当于Model Runner）
逐层前向、Batch Decode、CUDA Graph、Sampling
                |
                v
PagedKVCache + GpuLinearStatePool
                |
                v
Triton / FlashAttention Kernel
```

| 层次 | 主要职责 | 关键设计 |
|---|---|---|
| API层 | Tokenize、参数校验、SSE和增量解码 | HTTP线程只提交请求，不直接持有GPU状态 |
| 全局请求循环 | 有界排队、准入、连续批处理、公平性和生命周期 | 调度器依赖统一`GenerationBackend`协议，不耦合具体GPU拓扑 |
| 多Worker后端 | 选择D owner和Hybrid、容量预留、路由和角色切换 | Long先确定不可变D owner，再临时绑定P producer |
| Worker进程 | 加载完整模型、持有Cache/state、执行RPC | 一张GPU对应一个完整模型执行实例，进程内串行边界保护物理GPU |
| Model Runner | 模型逐层前向、Prefill和Decode transaction | `QwenTextRuntime`统一调用GDN/GQA/MLP/LM Head |
| 状态层 | Paged KV和固定槽recurrent state | 变长状态用Block，固定大状态用Slot，两套生命周期独立管理 |
| 传输层 | 描述符、Pinned Buffer、SHM Ring和安装流 | 控制面描述边界，数据面传payload；Cache dtype与wire dtype解耦 |
| Kernel层 | Norm、Conv、GDN、Attention、KV写入和量化GEMM | 按算子依赖选择并行维度，提供快路径和正确性回退 |

资源所有权必须保持单一：Coordinator拥有请求控制状态；D-bound worker拥有Long请求最终KV/state和后续Decode；Hybrid P只在Prefill期间临时产生状态；GPU Block和state slot最终由owner释放。这样可以避免同一请求在多个进程中被重复Decode或重复释放。

### 4.2 GPU执行模型：Thread、Warp、Program、Grid和Tile

CUDA Kernel的启动形式可以抽象为：

```text
kernel<<<grid, block>>>(...)
```

Triton写成：

```text
kernel[grid](...)
```

一个Triton program对应一个Grid坐标，内部通常由多个warp协作，因此“一个program处理一个token”不等于“一个线程串行处理整个token”。一个program负责的数据区域称为tile。

例如二维GEMM，一个program处理`[BLOCK_M, BLOCK_N]`输出tile：

```text
grid_m = ceil(M / BLOCK_M)
grid_n = ceil(N / BLOCK_N)
grid   = (grid_m, grid_n)
```

Kernel设计需要同时回答：

1. Grid沿哪些维度展开；
2. 一个program负责哪些Tensor元素；
3. 哪些维度能并行，哪些维度有真实依赖；
4. tile驻留寄存器、共享内存还是HBM；
5. tile大小如何影响复用、寄存器压力和SM驻留率；
6. 边界位置如何通过mask处理；
7. Prefill与Decode的shape变化是否值得使用不同Kernel。

GPU内存层次的基本取舍是：HBM容量大但延迟和带宽成本高；寄存器/片上存储快但容量有限。优化目标不是把完整请求状态放进片上，而是让当前program正在重复使用的小tile尽量少回写HBM。

### 4.3 RMSNorm与Gated RMSNorm：为什么一行一个Program

输入`[B,T,H]`在Kernel入口展平为：

```text
rows    = B × T
columns = H
grid    = (rows,)
```

每个program选择一行，也就是一个token的hidden vector：

```text
row = program_id(0)
offsets = 0...BLOCK-1
BLOCK = next_power_of_2(H)
```

4B模型`H=2560`时使用`BLOCK=4096`，`offset>=2560`的位置通过mask置零。program内多个warp并行加载这一行，转FP32后完成：

```text
rms = rsqrt(sum(x²) / H + eps)
y   = x × rms × weight
```

选择一行一个program是因为每个token有独立归一化因子，完整hidden维归约可以在同一program内完成。若多个program拆一行，就要把局部平方和写到全局内存，再启动归约和归一化Kernel；若一个program处理多行，则同时存活数据增加，可能提高寄存器压力、降低SM驻留率。

Prefill有大量row，天然提供充足Grid并行度；小Batch Decode只有少数row，GPU利用率较低，因此进一步优化方向是Residual+Norm融合、Norm+量化融合或Persistent Decode，而不是改变RMSNorm数学公式。

Gated RMSNorm进一步在同一Kernel内完成`SiLU(gate)`和逐元素乘法，减少一次中间Tensor写回和下一Kernel读取。内部归约和非线性使用FP32，最终输出回BF16。

### 4.4 Projection、MLP与Logits裁剪：同一权重，不同M维

Prefill和Decode的大多数Projection使用相同权重，但GEMM的`M`维完全不同：

```text
Prefill: [总Query Token数, hidden] × weight
Decode:  [Decode Batch Size, hidden] × weight
```

Prefill的大M GEMM有较高Tensor Core利用率，通常偏计算密集；Decode的小M GEMM权重复用不足，更容易受HBM读取和Kernel Launch影响。这也是Prefill大Kernel执行时间远高于CPU Launch，而Decode更适合CUDA Graph的原因之一。

MLP路径为：

```text
hidden
 -> fused gate_up GEMM
 -> Triton SiLU(gate) × up
 -> down projection
 -> residual
```

融合`gate`和`up`投影减少一次权重调度，SiLU-and-Mul融合避免把激活结果写回HBM再读取。没有把整个MLP融合为一个Kernel，是因为大权重GEMM仍更适合专用矩阵乘实现，强行融合会增加编译复杂度和寄存器压力。

Prefill只需要最后一个Prompt位置的首token logits。中间chunk完全跳过LM Head，末chunk只切出最后一行做`[1,H]×[H,V]`，避免生成`[chunk,vocab]`巨大Tensor。这项优化同时降低显存峰值、无效GEMM和小chunk数量。

### 4.5 Causal Conv与GDN：并行维度和递推维度分离

Causal depthwise-conv Kernel按：

```text
grid = (batch × token, channel_tile)
```

展开。一个program负责一个token位置的一段channel，并通过causal mask读取当前和历史窗口。Conv history是请求级状态：Prefill产生下一边界history，Decode每步消费一个token并推进固定窗口。

GDN recurrent Kernel按：

```text
grid = (batch, value_head, value_dim_tile)
```

展开。每个program加载`[key_dim, BLOCK_V]` FP32 state tile，在token循环内执行decay、prediction、delta和rank-1 update：

```text
S'ₜ = exp(gₜ) · Sₜ₋₁
pₜ  = kₜᵀS'ₜ
δₜ  = βₜ(vₜ-pₜ)
Sₜ  = S'ₜ+kₜδₜᵀ
oₜ  = qₜᵀSₜ
```

token维存在`Sₜ`依赖`Sₜ₋₁`的真实依赖，当前实现不在token维并行；并行度来自batch、head和value tile。选择切value维，是为了让每个program独立更新输出列tile，并在长token循环中复用同一state tile。GDN的value head数是key head的两倍，Kernel用`HEAD_RATIO`映射共享Q/K，避免显式`repeat_interleave`和额外显存流量。

所谓“state驻留片上”只指当前tile在本program的token循环中尽量保留在寄存器/片上存储；约53.48MB的完整请求状态仍位于显存。`BLOCK_V`过大会增加寄存器压力和spill，过小会增加program数量与重复加载，必须用基准测试调优。

Prefill时每个program内部循环多个token，执行时间长；Decode时token loop为1，但Batch通常较小。两条路径共享递推数学和状态布局，未来仍可针对单token Decode做更激进的融合。

### 4.6 PagedKVCache与Paged Attention：存储和计算不是同一组件

PagedKVCache把显存划分为固定物理Block，通过Block Table建立逻辑页到物理页映射：

```text
logical_block = token_position // block_size
offset        = token_position % block_size
physical      = block_table[request, logical_block]
KV位置        = cache[physical, offset]
```

固定页减少连续大块分配造成的外部碎片，但最后一个未填满Block仍有内部碎片。`KVBlockManager`负责分配、引用计数、Prefix共享、预留和释放；`PagedKVCache`负责数据、scale和Attention metadata。

Paged Attention是读取这些离散页的计算Kernel。它直接消费固定Cache基地址、Block Table和`sequence_lengths`，避免先把离散Block gather成连续KV Tensor。二者共享映射机制，但职责分别是“数据放在哪里”和“如何在原位计算Attention”。

KV写入也按逻辑位置映射到物理页。Prefill写入多个token，Decode批量写入每请求一个token。V4 INT8路径在同一Graph-safe Triton Kernel中写INT8 K/V和per-token-per-head FP32 scale，避免Python逐项`.item()`造成GPU到CPU同步。

### 4.7 Prefill与Decode为什么使用不同Attention Kernel

| 阶段 | Query长度 | KV长度 | 主要Kernel | 主要瓶颈 |
|---|---:|---:|---|---|
| 首个Prefill chunk | 多token | 当前chunk | FlashAttention varlen | 大规模矩阵计算 |
| Continuation Prefill | 多token | 历史+当前chunk | Paged Flash或tiled online-softmax | 历史KV重复读取 |
| Decode | 每请求1 token | 完整动态context | Split-K/Flash Paged Decode | 长KV扫描和小Kernel Launch |

首个长度为`T`的causal Prefill约有`T(T+1)/2`个Query-Key pair；Decode虽然每步只输入一个token，但仍需让一个Query扫描整个历史，因此复杂度约为`O(context)`。这里“变长Decode”指不同请求的context长度不同并且每步加一，不是Query长度大于一。

Tiled Prefill让一个program同时处理多个相邻query row，使它们共享已加载的KV tile，减少HBM重复读取。Decode没有多个query row可复用，长context则用Split-K并行扫描：每个split输出局部最大值`m`、分母`l`和累加量`acc`，Reduce阶段按online-softmax规则重新缩放合并。短context下Split-K的中间写入和Reduce可能得不偿失，因此保留Flash/reference回退和可切换路径。

### 4.8 CUDA Graph、动态Decode与形状分桶

CUDA Graph捕获的是固定GPU操作序列、依赖、Kernel参数和内存地址。Replay前可以改变静态Tensor中的值，但不能任意改变Tensor shape、地址、Kernel Grid、动态分配或Host控制流。

HydraServe为每张Decode Graph准备固定地址Buffer：

```text
input_ids      [B,1]
positions      [B,1]
block_table    [B,W]
lengths        [B]
state slot_ids [B]
logits         [B,1,vocab]
```

每轮先把新的token、position、Block Table、真实长度和state slot ID复制进去，再`graph.replay()`。一张Graph覆盖Embedding到Logits以及pooled state commit，不是每层或每个Kernel各一张。

Decode每请求Query长度固定为1，但context不同且增长：

```text
input_ids shape固定为[B,1]
sequence_lengths的shape固定为[B]
值可以从[42,8194,...]变成[43,8195,...]
```

Block Table宽度`W`属于shape，不能任意变化，因此按2的幂分桶：

```text
需要5、6、7、8个Block -> W=8
需要9...16个Block     -> W=16
```

不足的位置填`-1`，真实`sequence_lengths`保证Kernel不访问无效项。分桶不改变物理Block Size、不移动KV，只统一元数据Tensor宽度。当前Graph key为`(B,W)`；同一key重复观察默认16次后才捕获，偶发shape继续Eager，避免为一次性组合支付捕获和私有内存池成本。

准入时按`prompt+max_new-1`预留未来Block，因此单请求的Block Table宽度通常在生命周期内稳定。Batch成员变化仍可能改变最大`W`，Batch Size变化也需要另一张Graph。理论组合很多，但只捕获热点组合；进一步可把Batch也分为`1/2/4/8/...`桶，以较少Graph换取Padding计算。

Prefill更难Graph化，因为本轮Query Token总数直接改变GEMM M维、Norm row数、FlashAttention metadata、GDN token loop、KV写入量、Workspace和首段/续段/末段分支。即使固定chunk，最后chunk和ragged batch仍变化；而Prefill Kernel本身执行时间长，CPU Launch占比低，所以优先收益通常小于Decode。

Capture warmup和capture会真实执行Decode transaction，可能改写KV和GDN state。实现先快照受影响的state slot、workspace和KV page，捕获后恢复；如果捕获区域出现Host sync或动态行为，则标记该shape失败并回退Eager。这是正确性设计，不只是性能细节。

### 4.9 双状态池与事务式Decode

KV随token增长，使用Block allocator；GDN SSM/Conv对每个请求大小固定，使用FP32 slot pool：

```text
request_id -> state_slot
request_id -> block_ids[]
```

Decode Batch形成时，`GpuLinearStatePool.batch`把离散slot通过`index_select`收集到预分配workspace。Kernel只操作连续batch workspace；全部层、output projection和LM Head成功后，才把新state按`slot_ids`写回Pool并推进`sequence_length`。

这个事务边界避免半轮失败：如果第17层或LM Head报错，不能让前16层的新GDN state覆盖owner中的旧状态，否则重试会从不一致边界继续。ServingLoop还可二分失败Batch，保留成功请求并隔离单个失败请求。

State workspace预分配是为了避免每步动态CUDA分配，并为CUDA Graph提供稳定地址；代价是需要在启动时同时规划常驻slot和最大Batch workspace，而不能只按单请求53.48MB估算容量。

### 4.10 显存规划、容量预留与Prefix Cache

Memory Planner不是简单用“空闲显存/每token KV”计算Block数，而是先为以下内容留下预算：

- 模型权重和量化元数据；
- 每请求FP32 recurrent slot；
- Decode batch state workspace；
- CUDA library、Graph pool和临时激活；
- allocation guard与KV headroom。

然后才把剩余显存转换为BF16或INT8 KV Block。准入同时检查KV Block和state slot，避免只分到KV却没有recurrent state，或者Prefill完成后D无法接收。

当前实现采用保守的完整容量担保：准入时按`prompt_tokens + max_new_tokens - 1`预留未来KV Block。好处是请求一旦准入就不会在Decode中途因KV不足失败，D owner、PD迁移目标和CUDA Graph的Block Table宽度也更稳定；代价是请求提前EOS或用户把`max_new_tokens`设得过大时，尚未写入的预留页会长期占用显存。这部分浪费通常远大于最后一个Block的内部碎片，是后续提高并发的重点。

Prefix Cache使用Radix Tree共享完整Block前缀，并设置`CacheNamespace(model, tokenizer revision, model revision, adapter)`隔离边界：即使token ID数值相同，也不能跨模型版本或LoRA复用。Cost-aware policy同时考虑前缀长度、复用频率、重算成本和占用Block；频率门卫可拒绝一次性扫描污染Cache，引用计数保证活跃请求使用的共享Block不能被驱逐。

必须明确当前能力边界：GPU Prefix Cache只保存Full Attention KV页，不保存同一token边界的GDN FP32 recurrent state和Conv history。相同Prompt仍能匹配并共享Full Attention物理页，层间GDN/Full Attention交叉不会让这些KV失效；但Runtime仍需从token 0推进GDN/Conv状态，并重新执行各层Projection和Attention，因此当前收益主要是KV存储去重、减少共享页写回和提供路由亲和性，而不是像纯Transformer APC那样跳过整段Prefix Prefill。

HostPrefixCache是有界LRU/Radix可选路径。V4主压测没有启用，是为了把变量限制在Hybrid调度、PD传输和INT8 GPU Cache，避免Host命中率、PCIe安装和容量策略改变对照口径；“代码实现”与“本轮实验启用”必须分开表述。

### 4.11 传输协议、并发流水线与背压

传输层分为控制面和数据面：Descriptor/Manifest描述request、model、layer、token range、dtype和状态边界；Pinned Buffer与SHM Ring承载真正payload。这样可以在不改变调度器的情况下切换FULL、INT8或其他backend。

Long路径先启动D receiver，再启动P producer。每个Prefill chunk完成后记录CUDA Event，Transfer Stream等待Event后执行KV gather和D2H；后台线程写SHM；D consumer读取后在Install Stream执行H2D和scatter；P计算下一chunk。依赖关系为：

```text
P Compute(chunk i)
    -> CUDA Event
    -> P Transfer(chunk i) -----> SHM -----> D Install(chunk i)
P Compute(chunk i+1)与后两段并行
```

SHM Ring slot使用`FREE→WRITING→READY→FREE`状态机。Producer必须先claim FREE slot，Consumer处理速度不足时Producer阻塞形成背压；Receiver-first避免多个Producer先填满Ring而Consumer尚未启动的循环等待。固定slot避免每个chunk创建共享内存对象，但slot大小、数量和inflight深度需要在吞吐、内存和尾延迟之间折中。

V4 FULL wire以逻辑BF16 KV为兼容边界，BF16 raw bits用`uint16`容器进入SHM；P的INT8 Cache读取时先反量化，D再按本地Cache格式量化。这解耦两端布局和量化配置，但增加约两倍KV链路字节及一次反量化/重量化。两端格式完全兼容时，后续应直接传INT8 payload和FP32 scale；不兼容时回退BF16。

### 4.12 两级准入、公平调度与动态角色

系统存在两级约束：全局ServingLoop限制排队请求数、排队token数、活跃请求数和每轮token预算；Backend/Worker级准入检查具体GPU的KV Block、state slot、健康状态、Prefix命中、Decode load和角色。

异步Prefill按物理执行池限制未完成RPC数量，而不是让无界Host线程池提前为所有Long请求预留D资源。这样避免“只有一个P能前进，却有大量排队Long占满所有D KV”的资源囤积。

Decode公平分数综合：

```text
已服务token / (priority+1)
- priority bias
- aging weight × 等待轮数
- deadline bias × 截止期限紧迫度
```

已服务较少、优先级较高、等待更久或接近deadline的请求更早入选。Chunk边界的Short插入另有有界预算，预算耗尽必须继续Long，从而同时约束Short最坏等待和Long饥饿。

Hybrid角色状态机为：

```text
DECODE -> PREFILL_PENDING -> PREFILL_ACTIVE -> DECODE
```

`PREFILL_PENDING`先停止接收新普通Decode，再在安全边界切换；`PREFILL_ACTIVE`内部包含计算、chunk发布和收尾。Short可在D-bound或空闲Hybrid本地完成；Long先预留不可变D owner，再临时绑定Hybrid P；P不可用时回退D本地Prefill，保证reservation不会泄漏。

### 4.13 确定性Sampling与双端一致性

Batch中的每个请求独立应用history、penalty、temperature、top-k/top-p/min-p。Plain Greedy走批量argmax快路径；复杂采样把单行logits转FP32。随机种子按`request seed + generation step`混合，因此相同请求无论和哪些请求组Batch，都使用相同随机流，避免Continuous Batching改变输出。

Long PD中P计算完整Prompt并采样`y0`；D导入N−1 GDN state后replay最后Prompt token，再使用相同history、sampling参数、seed和step=0采样，记录首token是否一致。它能够发现状态边界、KV范围、dtype安装或sampling metadata错误，但token相同不代表logits/state逐元素一致，因此属于在线轻量语义哨兵，而不是完整数值证明。

### 4.14 抢占、故障恢复和可观测性

容量不足时可选择低紧迫度且重放成本较低的请求作为抢占victim。当前不把GPU KV完整保存到Host，而是释放Block/state；恢复时用`prompt + generated[:-1]`重放，最后已采样token作为下一轮输入。优点是恢复语义简单，缺点是长请求重算昂贵，因此限制每请求抢占次数。

Worker Registry维护健康状态、容量快照、拓扑代价、Prefix affinity和不可变request binding。Worker进程失败后先标记不健康并解除其请求绑定，再按指数退避重启并验证模型一致；设备本地状态丢失的请求只能通过重放恢复，不能假设新进程继承旧KV。

实验和线上诊断分别记录route reason、admission wait、prefill queue、transfer、Decode batch、replay mismatch、worker restart和release failure等信息。Benchmark manifest保存Git commit、dirty状态、CLI、模型文件、trace hash、CUDA/Triton版本和GPU拓扑，防止把到达模式或代码版本差异误认为调度收益。

### 4.15 设计覆盖矩阵

| 方面 | 核心问题 | 当前设计 | 代价或后续方向 |
|---|---|---|---|
| API与生命周期 | 流式、取消、停止和资源释放如何闭环 | GenerationHandle/Event、增量解码、异步release | API生态和生产鉴权仍不如成熟框架 |
| 连续批处理 | 请求动态加入退出 | 每轮选择active请求并按owner并行发射 | 小Batch仍有Python/Launch开销 |
| 公平性 | Short低延迟与Long不饿死 | service token、priority、aging、deadline和short budget | 权重仍需按真实SLO校准 |
| Worker路由 | 容量、负载、Prefix和拓扑冲突 | 先过滤容量/健康，再综合score并不可变绑定 | 多节点控制面尚未完成 |
| 动态角色 | 静态P空闲、Long又需隔离 | Hybrid三态状态机和工作守恒调度 | 角色数量尚未在线自动伸缩 |
| KV管理 | 变长状态、显存碎片和中途OOM | 固定Block、Block Table、引用计数和完整输出容量预留 | 尾页碎片较小，但按`max_new_tokens`整段预留可能显著降低并发 |
| GDN状态 | 固定大FP32状态和递推边界 | Slot Pool、Batch workspace和事务提交 | 每请求约53.48MB限制并发 |
| Prefix复用 | 一次性扫描污染、跨版本误复用和混合状态不完整 | Namespace、频率门卫、成本感知驱逐；当前只共享Full Attention KV页 | 未缓存GDN/Conv checkpoint，不能跳过完整Prefix计算；V4也未启用 |
| Kernel执行 | 算子依赖和数据复用不同 | Norm行级、Conv channel tile、GDN value tile、Attention split/tile | 需要持续做shape级autotune |
| CUDA Graph | 动态Batch/context与静态捕获冲突 | 静态Buffer、`(B,W)`热点Graph、失败回退 | Batch分桶和Graph内存仍可优化 |
| 量化 | 权重、KV、wire语义易混淆 | AWQ/FP8权重、INT8 GPU KV、独立wire mode | 原生INT8 Cache直传未完成 |
| PD传输 | 双状态、背压和计算重叠 | Receiver-first、Event、Pinned、Ring、Install Stream | SHM限单机，多P背压需credit |
| 正确性 | N−1、最后token和首token易off-by-one | Descriptor边界、Replay和首token哨兵 | 还需logits/state离线强校验 |
| 故障处理 | Batch或Worker失败会污染其他请求 | 事务state、Batch二分、健康隔离、重启和重放 | 长请求恢复成本高 |
| 可复现性 | 到达模式和dirty commit造成口径漂移 | Manifest、hash、环境和分阶段指标 | 现有V4仍需clean commit重跑 |

面试时不需要一次讲完本章。推荐先说“系统层、状态层、Kernel层、传输层、调度层都有独立设计”，再根据追问进入对应的shape、Grid、dtype、资源所有权和失败语义。

## 5. 项目核心难点

| 难点 | 根因 | 解决方案 | 结果或现状 |
|---|---|---|---|
| 双状态一致性 | KV 是变长分页状态，GDN state 是固定 FP32 状态，token 边界不同 | descriptor 显式记录状态类型和 token 边界；N−1 state + 最后 token replay + 首 token 校验 | 实现 P/D 两侧状态闭环，避免静默生成漂移 |
| GDN 递推 Kernel | token 维有强依赖，不能像 Attention 一样完全并行 | 按 batch/head/value tile 并行，tile 在 token 循环期间留在片上，最终写回 | 支持 prefill 与单 token decode 两条 Triton 路径 |
| 长上下文 Prefill | 无效全词表 logits 占显存；continuation 反复扫描历史 KV | 中间 chunk 跳过 LM Head；末 chunk 只算末位置；Flash/paged tiled prefill | 32K Prefill 约 268.3s→11.3s |
| 分块迁移与背压 | 多 producer 可能先填满有限 SHM slot，D receiver 尚未启动 | receiver-first、固定 slot 状态机、后台 transfer、独立 install stream | 8K 完整状态约 66ms；多 P→单 D 仍需进一步 tracing |
| Prefill 干扰 Decode | 大 GEMM/Attention 占用 SM、HBM 和执行队列 | Long Prefill 隔离到 Hybrid；空闲期 Hybrid 回到 Decode；chunk 边界穿插 Short | H1 提高 SLO goodput，同时保留大部分 DP 吞吐 |
| 调度公平性 | Short需要低延迟，Long不能被无限插队 | 有界短请求预算（short budget）、优先级、等待老化和截止期限紧迫度 | 控制Short最坏等待，同时保证Long推进 |
| 性能结论复现 | 压测执行器（benchmark runner）、到达模式（arrival pattern）、非干净实验版本（dirty commit）容易造成口径漂移 | 保存模型/trace hash、CLI、GPU、环境和分位数；计划用干净提交重跑 | 当前趋势可信，严格复现仍需补强 |

## 6. 关键改善举措

### 6.1 消除无效 logits

旧 Prefill 路径会为每个 chunk 的所有位置执行 LM Head，生成 `[chunk, vocab]` logits，但系统只消费最后一个 prompt 位置的 logits。

改善后：

1. 中间 chunk 完全跳过 LM Head；
2. 最后 chunk 只投影最后一个位置；
3. 显著降低显存峰值和无效 GEMM；
4. 32K 能使用更大的 chunk；
5. Full Attention 更多地进入 FlashAttention 快路径。

结果：云端历史基准由约 268.3s 降至 11.3s，约 23.8×。

### 6.2 GDN recurrent tile 复用

GDN 状态形状为 `[batch, value_head, key_dim, value_dim]`。Kernel 按 value 维切 tile：

1. 从 global memory 加载 recurrent tile；
2. 在 token 循环内重复使用；
3. 每步计算 decay、prediction、delta 和 rank-1 update；
4. 完成整个序列后统一写回。

改善目标是减少每个 token 对同一状态 tile 的重复 global-memory 读写。完整状态仍位于显存，不能描述为全部驻留 SRAM。

### 6.3 Paged Attention split-K 与 tiled prefill

- Decode split-K：把长 context 划分为多个 split，分别计算局部 softmax `m/l/acc`，再通过 online-softmax reduce 合并。
- Tiled prefill：一个 program 同时处理多个 query row，让同一 KV tile 被多个 query 复用，避免逐 query 重复扫描历史。
- Flash 路径：首 chunk 使用 varlen FlashAttention，continuation 可使用 paged flash prefill；不可用时回退到自研 tiled kernel。

### 6.4 双状态传输策略

4B、8K Prompt 的 BF16 KV 为：

```text
8 layers × 4 KV heads × 256 dim × 2(K,V) × 2 bytes × 8192
= 256 MiB
```

Recurrent state 约为：

```text
SSM  = 50,331,648 bytes
Conv =  3,145,728 bytes
总计 = 53,477,376 bytes ≈ 53.48 MB
```

总迁移量约 309MiB。在 4.58GB/s 的 SHM 带宽下，对应约 66～70ms，与实测一致。

改善举措包括：

- D receiver 先于 P producer 启动；
- P 每完成一个 chunk 就发布对应 KV 范围；
- Pinned Buffer 支持异步 D2H/H2D；
- transfer executor 与 Prefill 计算重叠；
- D 使用独立 CUDA stream 安装 KV；
- descriptor 关联 streamed ranges 与 recurrent state。

### 6.5 修复 CUDA Graph 与 INT8 KV 冲突

旧实现用 Python `.item()` 逐项读取和写入 INT8 KV scale，触发 GPU→CPU 同步，使 CUDA Graph capture 失败。

改善后将 value 和 scale 的写入放进批量 CUDA/Triton Kernel，实现 Graph-safe INT8 KV scatter。需要强调：这是 GPU Cache 写入优化，不代表 V4 使用 INT8 链路传输。

### 6.6 动态 Hybrid 调度

代码中的 Hybrid 角色状态机为：

```text
DECODE -> PREFILL_PENDING -> PREFILL_ACTIVE -> DECODE
```

Prefill计算、chunk传输和最终bundle发送都是`PREFILL_ACTIVE`期间的子阶段，并不是额外的HybridRole枚举值。

- 空闲时：Hybrid 参与 Short/Decode，接近 4×DP 的 Decode 容量；
- Long 到达：Hybrid 停止接收新 Decode 工作并切换到 Prefill；
- Prefill chunk 边界：有界处理 Short 操作；
- 状态迁移完成：Long 请求交给目标 D；
- 清理完成：Hybrid 返回 Decode 池。

该机制改善的是逻辑角色利用率，不是物理 GPU 热插拔。

## 7. 最终成果

### 7.1 模型与执行路径

| 模型 | 验证范围 |
|---|---|
| Qwen3.5-4B BF16 | 完整模型执行、多 GPU PD/Hybrid、V4 压测 |
| Qwen3.5-9B BF16 | 完整模型执行路径验证 |
| Qwen3.6-27B AWQ/FP8 | 64层执行与量化权重路径验证 |

“全规模验证”不表示三个模型都完成了完全相同的四卡 V4 压测。面试时应主动区分功能验证和系统压测。

### 7.2 H1 与 4×DP 的实测结果

| 负载 | D0 SLO | H1 SLO | D0 TPOT p50 | H1 TPOT p50 | D0 TTFT p50/p99 | H1 TTFT p50/p99 | 吞吐比例 H1/D0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 s42 | 16/48 | 28/48 | 133ms | 75ms | 678/2413ms | 389/2808ms | 107% |
| R1 s43 | 16/48 | 31/48 | 156ms | 74ms | 590/2041ms | 386/3213ms | 111% |
| R1 s44 | 16/48 | 29/48 | 153ms | 70ms | 644/2546ms | 392/987ms | 119% |
| R2 s42 | 16/48 | 33/48 | 155ms | 97ms | 712/4279ms | 383/7329ms | 92% |
| R2 s43 | 16/48 | 32/48 | 161ms | 76ms | 621/2478ms | 394/9245ms | 93% |
| R2 s44 | 16/48 | 32/48 | 116ms | 78ms | 691/5606ms | 439/5477ms | 96% |
| R3 s42 | 16/44 | 30/44 | 153ms | 73ms | 646/1995ms | 385/3253ms | 119% |

结论：

- Short SLO goodput 相对提升 75%～106%；
- TPOT p50 下降 38%～52%；
- TTFT p50 下降 34%～46%；
- 总吞吐保持 D0 的 92%～119%；
- TTFT p99 并未稳定改善，是当前系统的主要尾延迟问题。

### 7.3 四拓扑单点结果

R1 seed42：

| 拓扑 | Short SLO | TPOT p50 | TTFT p50/p99 | 吞吐 |
|---|---:|---:|---:|---:|
| H1：1H+3D | 28/48 | 75ms | 389/2808ms | 108.0 tok/s |
| H2：2H+2D | 23/48 | 96ms | 423/2912ms | 94.8 tok/s |
| D0：4×DP | 16/48 | 133ms | 678/2413ms | 100.5 tok/s |
| P0：静态2P+2D | 11/48 | 153ms | 3064/6089ms | 81.1 tok/s |

这个单点说明动态角色切换是收益的重要来源：静态 P 卡空闲时无法贡献 Decode，而 H1 可以重新加入 Decode 池。该结论不能直接外推到不同模型、硬件或负载。

## 8. 高频面试问题与参考回答

### 8.1 为什么混合注意力 PD 比普通 Transformer 更复杂？

普通 Transformer 主要迁移 KV Cache。GDN+GQA 还需要迁移 GDN recurrent matrix 和 causal-convolution history。KV 随 context 线性增长，GDN state 固定大小且要求 FP32，两者的布局、精度、生命周期和迁移策略都不同，因此需要异构状态协议。

### 8.2 4B 的模型结构和状态大小是多少？

当前 4B preset 是 32 层，其中 8 层 Full Attention、24 层 Linear Attention。BF16 KV 每 token 是 32KiB；GDN recurrent state 每请求约 53.48MB，其中 SSM 约 50.33MB、Conv 约 3.15MB。

### 8.3 为什么 GDN state 必须保持 FP32？

GDN state 会跨 token 持续递推。decay、预测误差和 rank-1 update 会把低精度误差累积到后续所有 token。当前实现允许权重和 KV 量化，但协议强制 LINEAR_SSM 和 LINEAR_CONV 使用未量化 FP32。

### 8.4 GDN delta rule 的计算过程是什么？

```text
S'ₜ = exp(gₜ) · Sₜ₋₁
pₜ  = kₜᵀS'ₜ
δₜ  = βₜ(vₜ-pₜ)
Sₜ  = S'ₜ+kₜδₜᵀ
oₜ  = qₜᵀSₜ
```

q和k先归一化，q再乘 `key_dim^-0.5`。Kernel 的难点是 token 维存在真实递推依赖，只能在 batch、head 和 value tile 上并行。

### 8.5 “状态驻留 SRAM”到底是什么意思？

不是完整状态驻留片上。每个 Triton program 加载一个 `[key_dim, BLOCK_V]` state tile，在本 program 的 token 循环中留在寄存器或片上存储，避免每个 token 重复读写 global memory，最后统一写回。

### 8.6 Paged Attention split-K 怎么保证 softmax 正确？

每个 context split 分别计算局部最大值 `m`、分母 `l` 和 value 累加量 `acc`，reduce 时通过 online-softmax 重新缩放：

```text
m = max(m_old, m_split)
l = l_old·exp(m_old-m) + l_split·exp(m_split-m)
acc = acc_old·exp(m_old-m) + acc_split·exp(m_split-m)
```

最终输出 `acc/l`。这样可以并行扫描长 context，同时保持数值稳定。

### 8.7 Tiled prefill 为什么更快？

逐 query 方案会让相邻 query 重复读取相同历史 KV。Tiled prefill 让一个 program 同时处理多个 query row，同一个 KV tile 被多个 query 复用，减少 HBM 读取；同时通过 `key_position <= query_position` 保证 causal mask。

### 8.8 32K 的 23.8× 是如何得到的？

旧路径每个 chunk 都计算所有位置的全词表 logits，造成巨大无效 GEMM 和显存占用，使 32K 只能拆成多个小 chunk。优化后中间 chunk 跳过 LM Head、末 chunk 只计算最后一个位置，32K 可以采用大 chunk 并更多进入 FlashAttention。历史数据约为 268.3s→11.3s。

### 8.9 Paged KV Cache 和 recurrent state pool 为什么要分开？

KV 是随 token 增长的变长状态，适合 block allocator；recurrent state 对每个请求大小固定，适合 slot pool。Decode batch 前通过索引把多个 slot 收集到预分配 workspace，计算完成后事务式写回。

### 8.10 N−1 truncation 是不是只传 N−1 个 token 的 KV？

不是。N−1 只表示 recurrent state 的边界。P 在 N−1 处提取循环状态，随后处理最后一个 prompt token，并发送该 token 的 KV。D 导入 N−1 state 后 replay 最后一个 token，将 recurrent state 推进到 N，并校验 P/D 首 token。

### 8.11 为什么不使用 PARTIAL 模式只传 recurrent state？

PARTIAL 会让 D 重新计算完整 Full Attention KV。对长 Prompt 来说，重算 Prefill 往往需要秒级，而8K完整状态传输约66ms，因此节省几十毫秒带宽却增加数秒计算通常不划算。是否使用 PARTIAL 仍取决于网络带宽和目标硬件。

### 8.12 INT8 KV Cache 和 INT8 transfer 有什么区别？

前者描述 GPU Cache 的存储格式，后者描述 P→D wire payload。V4 配置是 `kv_quant=int8`、`pd_transfer_quant=None`，因此 Cache 是 INT8，但传输时发送完整 BF16 KV。

### 8.13 传输与计算如何重叠？

协调器先启动 D receiver，再执行 P Prefill。P 每完成一个 chunk 就把已完成 KV 交给后台 executor 抽取和发送，D 在独立 CUDA stream 上安装；与此同时 P 继续计算下一 chunk。最终仍需等待 transfer future 和 D prepare，因此不能声称首 token 完全不等待传输。

### 8.14 SHM Ring 如何避免多 producer 冲突？

slot 使用 `FREE→WRITING→READY→FREE` 状态机。producer 通过文件锁原子 claim FREE slot，写完数据后最后翻转 READY；consumer 读取后恢复 FREE。receiver-first 避免 P 先填满 ring、D consumer 尚未启动造成的死锁。

### 8.15 fused gather/scatter 做了什么？

它根据 block table，把分页 KV 的逻辑 token 范围一次性整理为连续 staging tensor，或者把连续 payload 写回目标分页 Cache，减少逐层和逐块 Python dispatch。当前快路径主要覆盖未量化 GPU Cache；INT8 Cache 仍可能回退到逐层 read/stack。

### 8.16 为什么动态 H1 比静态 PD 好？

静态 PD 始终占用固定 P worker，即使没有 Long 请求也不能贡献 Decode。H1 的 Hybrid worker 空闲时与 D-bound 完整模型副本一起服务 Short/Decode，Long 到达才切换到 Prefill，既保留隔离能力，又降低 P worker 空置时间。

### 8.17 为什么 H2 不如 H1？

当前负载中一张 Hybrid 已经基本满足 Prefill 需求。H2 增加了一个可切换 Hybrid 副本，但减少了一个 D-bound 完整模型副本，稳定可用的 Decode 容量降低25%；额外 Prefill 供给没有充分利用，因此 TPOT 和吞吐反而下降。

### 8.18 Chunk 边界为什么可以穿插 Short？

每个 chunk 结束时，模型状态和 KV 长度处于一致边界，可以安全暂停 Long。系统在这个边界处理有限数量的 Short 操作，再继续下一个 Long chunk，把 Short 最坏等待从完整 Long Prefill 缩短到一个 chunk 加有限预算。

### 8.19 如何防止 Short 一直插队导致 Long 饿死？

使用有界短请求预算（short budget）：每个Long分块边界最多处理固定数量的Short操作，预算耗尽后必须推进Long。公平调度器还结合优先级、等待老化（aging）和截止期限紧迫度（deadline urgency）。

### 8.20 抢占时是否保存 GPU KV 到 Host？

没有。当前抢占释放 KV block 和状态资源；恢复时通过 `prompt + generated[:-1]` 重新 Prefill/replay，恢复可继续 Decode 的精确状态。这样实现简单且一致性清晰，但长请求重算成本较高。

### 8.21 SLO goodput 和普通吞吐有什么区别？

普通吞吐统计单位时间生成的 token 数，不关心请求是否满足延迟目标。SLO goodput 只统计成功且 TTFT/TPOT 等指标满足阈值的请求。长 Prefill 干扰下，系统可能总吞吐不低，但大量 Short 超时，因此 goodput 很差。

### 8.22 75%～106% 是怎么计算的？

例如 R1 seed42，D0 有16个Short达标，H1有28个：

```text
(28-16)/16 = 75%
```

R2 seed42是16→33：

```text
(33-16)/16 ≈ 106%
```

这是相对提升，不是增加75～106个百分点。

### 8.23 为什么吞吐只有 D0 的92%时仍认为H1更好？

优化目标是SLO goodput而不只是token吞吐。R2中H1总吞吐小幅下降，但Short SLO由16/48提高到32～33/48，TPOT也明显改善。系统用约4%～8%的吞吐换取接近翻倍的达标请求数。在R1/R3中，总吞吐还实现了反超。

### 8.24 TTFT 是否也改善？

Short TTFT p50 在全部V4数据点下降34%～46%，但p99没有稳定改善，部分R2 seed的尾延迟明显变差。因此准确结论是“中位TTFT改善，尾延迟仍需治理”。

### 8.25 为什么不直接在vLLM/SGLang上开发？

项目目标是验证混合注意力双状态协议和动态角色调度，需要直接控制GDN state、Paged KV布局、descriptor和worker状态机。从零实现便于进行这些实验，但生产化时仍应评估将成熟机制接入vLLM/SGLang，而不是重复维护完整生态。

### 8.26 Decode池里的卡是不是默认都是DP？“常驻D”是否准确？

所有多GPU worker都持有完整模型权重，因此从模型副本角度是请求级DP，而不是TP；但执行角色并不完全相同。D0的worker是对称collocated副本。H1的D-bound worker会给Short执行本地Prefill+Decode，也会接收Long的PD状态后继续Decode；Hybrid worker空闲时同样可以承接Short和Decode，Long到达时才切成Prefill producer。因此“常驻D”只能作为拓扑简称，更准确的术语是“D-bound完整模型副本”，Decode池则是当前可被调度执行Decode的逻辑worker集合。

### 8.27 什么是工作守恒调度（work-conserving scheduling）？

工作守恒表示：只要系统中存在能够在某个空闲worker上安全执行的请求，就不让该worker因为固定角色划分而空转。在HydraServe中，没有Long请求时，Hybrid worker会加入逻辑解码池，承接Short的同卡预填充和后续Decode；有Long请求时再切换为Prefill角色。它描述的是资源不空转原则，不等于“所有请求都立即执行”，请求仍需满足KV容量、状态槽、角色状态、优先级和公平调度约束。

### 8.28 为什么长Prefill会让Decode TPOT放大2.5～6.4×？

当Long Prefill和Decode落在同一GPU上时，两者虽然可以放在不同CUDA Stream，但并不会获得物理资源隔离。Prefill的大M GEMM和大范围Attention会长时间占用Tensor Core、SM、寄存器、L2和HBM带宽；已经发射的大Kernel也不能在任意位置被Decode抢占。Decode本身是小Batch、小Kernel链，单步计算短，却要求每一步及时完成，因此一旦在执行队列中等待Prefill Kernel，TPOT就会直接增加。Prompt越长、Prefill Kernel越大，干扰窗口通常越长。2.5～6.4×是固定Decode负载下的实测结果，不应表述成所有硬件和负载上的普遍倍数。

### 8.29 为什么实现了Prefix Cache，V4压测却没有启用？

V4要验证的是动态Hybrid角色本身能否隔离Long Prefill并改善Short SLO。如果开启Prefix Cache，不同拓扑或seed可能产生不同Full Attention页命中率，KV占用和传输量都会成为额外变量，无法判断收益来自角色调度还是缓存。因此主对照实验关闭Prefix Cache以保持工作量一致。还要主动说明：当前缓存只覆盖Full Attention KV，不包含GDN/Conv checkpoint，即使命中也不能跳过完整Prefix计算；后续必须先实现混合状态Bundle，再独立消融有效跳过token数、状态安装成本、TTFT和显存占用。

### 8.30 调度是否考虑Short和Long的到达顺序？

考虑了，而且不是固定“Short永远优先”或“Long永远优先”。典型情况如下：

| 到达场景 | 调度行为 | 设计目的 |
|---|---|---|
| 只有Short | 所有当前可解码worker，包括空闲Hybrid，按容量和负载承接本地Prefill+Decode | 工作守恒，避免Hybrid空转 |
| Long先到，Short后到 | Long绑定Hybrid和D-bound owner；Hybrid在chunk边界最多执行有界数量的Short，D侧已有Decode继续推进 | 限制Short等待，同时不让Long饿死 |
| Short先到，Long后到 | 已有Short不被粗暴迁移；Hybrid停止接收新的长期Decode归属，在安全边界切到`PREFILL_PENDING/ACTIVE` | 避免破坏已建立状态，又让Long及时获得Prefill资源 |
| 多个Long连续到达 | 等待队列先按priority、admission age、deadline和到达序排序，再逐个检查Hybrid执行槽、D owner的KV/state容量；资源不足时延后准入 | 控制OOM、资源囤积和排队饥饿 |
| 高优先级请求后到 | priority提供偏置，但仍受容量与安全边界约束；aging和short budget限制长期压制 | 兼顾业务优先级与公平性 |

实现上有三个连续决策：ServingLoop先根据priority、admission age和deadline决定谁先尝试准入；Backend再做worker级容量预留和不可变owner绑定；请求进入Active集合后，ServingLoop的公平Decode调度器按已服务token/权重、priority、aging和deadline选择下一批，Backend再按owner拆分并发给各worker。这样分别解决“谁先进入系统”“请求放到哪张卡”和“活跃请求下一轮先服务谁”。

### 8.31 当前Prefix Cache对Qwen混合模型是否真的有用？vLLM如何处理？

相同Prompt能够匹配HydraServe缓存的Full Attention KV，因为这些KV已经包含前序GDN层对hidden的影响，层间交叉不会让KV本身失效。但它不是完整的计算命中：从Prefix末尾继续执行还需要所有GDN层在同一token边界的FP32 recurrent state和Conv history；当前缓存没有这些状态，所以Runtime仍从token 0重算Prompt。准确表述是“当前实现了Full Attention KV页共享和存储去重，但尚未实现混合模型端到端Prefix计算复用”。

当前vLLM对Qwen3.5/Qwen3-Next类模型使用Hybrid KV Cache Manager：Full Attention KV和GDN/Mamba类内部状态分别组成Cache Group，各自查找命中后取共同可恢复边界。Full Attention组提供历史KV，状态组提供对齐边界上的recurrent/Conv checkpoint，然后只重算边界之后的suffix并至少重算最后一个Prompt token以得到logits。Qwen当前主要使用实验性的`mamba_cache_mode=align`稀疏保存状态检查点，`all`模式并未普遍支持，因此命中粒度和状态成本仍是成熟框架也在解决的问题。

HydraServe后续应把缓存项扩展为事务式`HybridPrefixEntry`：`namespace + state_token_count + Full Attention block_ids + FP32 GDN state + Conv history`。匹配时先找最长KV前缀，再找不超过它的最近状态checkpoint，选择共同边界，安装两类状态后只计算suffix。由于4B每个checkpoint约53.48MB，不能每16个token保存一次，应只为高频、长且重算昂贵的Prefix在chunk边界稀疏保存，并比较GPU/Host安装成本与重算成本后决定准入和淘汰。

## 9. 压力追问与回答

### 9.1 这些数据能严格复现吗？

当前数据能支持同一实验组内的趋势，但还不属于严格复现：V4基于`a090ec9`且`git_dirty=true`，汇总写Poisson而JSON/CLI记录burst。正式发布前需要使用clean commit统一arrival pattern，并固化模型hash、trace hash、warmup和完整命令行。

### 9.2 Prefill干扰2.5～6.4×如何测量？

固定模型、Decode batch、上下文和输出长度，先测无并发Prefill的稳态TPOT，再同时执行不同长度的Long Prefill，比较相同Decode负载的TPOT；排除首次Triton编译和Graph capture，并使用多个seed重复。简历保留该数字的前提是能够展示对应原始日志。

### 9.3 当前最大瓶颈是什么？

执行性能主要有三类瓶颈：小batch Decode的attention/kernel launch开销；INT8 Cache传输时逐层read/stack fallback；Hybrid路径的TTFT p99抖动。资源管理还有两个明显缺口：按`max_new_tokens`整段预留降低并发；Prefix Cache没有GDN/Conv checkpoint，不能跳过混合模型Prefix计算。其次是多P→单D的SHM背压和benchmark复现口径。

### 9.4 HydraServe与vLLM还有多少差距？

仓库现有R1数据中，vLLM 4×DP三seed平均约151.7 tok/s，旧HydraServe D0约98 tok/s，Flash D0 seed42约121.9 tok/s。但runner的arrival pattern不同，不能直接宣布严格倍数。差距主要来自成熟attention kernel、融合、CUDA Graph覆盖和长期硬件调优。

### 9.5 项目最典型的Bug是什么？

可以回答receiver-first死锁：早期P先发送chunk，而D prepare还在线程池排队；多个producer填满SHM slot后等待consumer，D又尚未启动接收，形成死锁。修复为先派发D prepare，并用Event确认receiver启动后再执行P producer。

也可以回答INT8 KV Graph问题：旧路径用Python `.item()`写scale，引入GPU→CPU同步并破坏Graph capture；改为批量CUDA/Triton写value和scale后恢复Graph-safe路径。

### 9.6 项目最难的部分是什么？

推荐回答双状态一致性，而不只是“写Kernel”：P/D必须在N−1 state、最后一个prompt token、KV长度和首token之间保持严格一致，任何一处off-by-one都可能造成不报错的生成漂移。最终通过descriptor边界、最后token replay和P/D首token校验闭环。

### 9.7 如果扩展到100张GPU，首先改什么？

1. 用RDMA/NIXL/Mooncake类transport替换单机SHM；
2. 数据面与控制面分离；
3. 引入service discovery、故障检测和拓扑感知路由；
4. 完成真实TP state slicing；
5. 使用credit-based flow control；
6. 建立远端prefix cache；
7. 根据P/D队列、网络和SLO动态建议拓扑比例。

## 10. 后续改善计划

改进顺序遵循三个原则：先保证状态边界和实验口径正确，再解决显存与尾延迟，最后扩展多节点能力；每个优化都必须有独立开关和对照实验，不能只看总吞吐掩盖TTFT、TPOT或正确性退化。

| 优先级 | 方向 | 当前主要缺口 | 目标 |
|---|---|---|---|
| P0 | 正确性与可复现性 | dirty commit、arrival口径和边界校验不完整 | 每个数字可由固定manifest和原始日志复现 |
| P1 | 混合状态Prefix Cache | 只有Full Attention KV，没有GDN/Conv checkpoint | 从共同状态边界恢复，真正跳过Prefix计算 |
| P2 | KV容量与生命周期 | 按`max_new_tokens`整段预留降低并发 | 分段担保、按需增长，同时控制中途OOM风险 |
| P3 | Decode与Kernel | 小Batch Launch、HBM流量和shape敏感 | 降低TPOT并保持Graph-safe和数值一致 |
| P4 | 状态传输 | INT8 Cache经BF16中转、SHM背压和单机限制 | 减少链路字节并提高真实计算通信重叠率 |
| P5 | 调度与扩展 | TTFT p99、固定拓扑比例和单机控制面 | SLO驱动的在线角色与多节点扩展 |

### P0：正确性和可复现性

- 使用clean commit重新运行完整H1/D0/P0实验；
- 统一Poisson/burst和冻结trace的语义；
- 为模型、trace、环境、CLI和结果生成manifest；
- 为N−1、最后token replay和首token返回时序增加端到端测试；
- 增加P/D logits误差、GDN state误差、KV逐层误差和最终token一致性的离线强校验，不能只依赖首token相同；
- 为CUDA Graph capture前后状态恢复、INT8 scale写入、Prefix引用计数和Batch失败回滚增加不变量测试；
- 对H2/R2 hang增加producer/consumer/slot生命周期tracing。

验收指标：相同seed和sampling参数下Eager/Graph、同卡/PD、BF16/INT8 Cache路径输出一致；clean commit多seed重跑后结论和置信区间稳定。

### P1：混合状态Prefix Cache

当前Full Attention KV页命中只能减少存储和写回，不能跳过混合模型Prefix。改进设计为：

```text
HybridPrefixEntry
  namespace
  token_ids / prefix hash
  state_token_count
  Full Attention block_ids
  FP32 GDN recurrent checkpoint
  FP32 Conv history
  frequency / recompute_cost / bytes / last_access
```

- 分别查找最长Full Attention KV前缀和最近可用GDN/Conv checkpoint，取两者共同可恢复边界；
- 在同一事务中Pin KV页、安装FP32状态并建立Block Table，任一阶段失败则回滚全部引用；
- 从checkpoint之后只计算suffix；完整Prompt命中时沿用N−1思想，至少replay最后一个Prompt token产生最终logits；
- 不按每个KV Block保存约53.48MB状态，只在chunk边界对高频、长且重算昂贵的Prefix稀疏保存；
- 根据`节省的Prefill时间 - GPU/Host状态安装时间`决定是否命中，不能只按token长度；
- 建立GPU热checkpoint和Host冷checkpoint两级缓存，Host项在异步H2D期间Pin，避免恢复与淘汰竞态；
- KV页和GDN checkpoint作为同一个Bundle驱逐，避免只剩一半状态的伪命中。

验收指标：分别记录KV命中token、状态checkpoint边界、真正跳过token、状态安装时间、节省的Prefill时间、TTFT变化、每命中节省的计算量以及每GB缓存产生的goodput收益。

### P2：KV容量、预留和淘汰

- 把`prompt + max_new_tokens - 1`整段预留改为分段担保，例如先保证未来256/512个Decode token，接近窗口末端再扩展；
- `grow_many`继续保持Batch原子性：整批扩展成功后才能发射Decode，失败时不允许部分请求先增长；
- 为下一轮Decode、投机token、PD install和故障恢复保留独立watermark，避免所有路径竞争最后一批Block；
- 将扩展窗口对齐到现有CUDA Graph的Block Table宽度桶，在显存利用率和Graph复用之间折中；
- 活跃Full Attention KV保持不可淘汰；容量不足时先驱逐无活跃引用的Prefix Bundle，再考虑低紧迫度请求抢占和精确replay；
- 统计`logical_tokens / reserved_tokens / physical_capacity`，区分未写入预留空间和最后一页内部碎片；
- 暂不优先把无关请求塞进同一尾页，因为它需要子页所有权、base offset、Copy-on-Write和更复杂的Attention ABI；先对Block Size做敏感性实验，只有尾页浪费成为主瓶颈时再评估token级tail allocator。

验收指标：在相同24GB显存下比较最大并发、KV利用率、准入等待、抢占/重放次数、SLO goodput和CUDA Graph命中率；不能只报告“可接纳请求数”。

### P3：Decode性能与Kernel优化

- 对split-K、flash decode、CUDA Graph和`torch.compile`做逐项消融；
- 降低小batch Python dispatch和kernel launch；
- 按Prefill/Decode、Batch、Context和dtype分别autotune GDN `BLOCK_V`、Paged Attention `BLOCK_T/num_splits`和Warp数，同时监控寄存器数、spill、共享内存和SM驻留率；
- 为Decode Batch增加`1/2/4/8/...`Padding桶并实现backend `decode_padded`，与现有Block Table宽度桶组成有限热点Graph集合；
- 将Gate/Up GEMM的`SiLU(gate)×up`尝试下沉到GEMM epilogue，消除`gate_up`的HBM写回和再读取；保留独立Kernel回退，不盲目把两个大型GEMM做成高寄存器压力Mega Kernel；
- 优化GDN单token Decode路径、Projection融合和INT8 KV读取，确保Tile增大带来的复用收益没有被Occupancy下降或register spill抵消；
- 根据Context阈值选择普通Paged Decode、Split-K或Flash路径，短Context避免支付partial buffer和Reduce开销；
- 分离Attention、GDN、GEMM、sampling和state transaction耗时。

验收指标：除TPOT p50/p99外，同时报告Kernel时间、Launch占比、HBM吞吐、寄存器/Shared Memory、Occupancy、Graph replay率和Eager回退原因。

### P4：状态传输与计算重叠

- 为INT8 Cache实现直接fused gather，避免逐层read/stack；
- 完成`INT8 value + FP32 per-token-per-head scale`直传，D端布局兼容时避免P反量化和D重量化，不兼容时回退BF16 wire；
- 为SHM ring增加显式credit、backpressure和timeout diagnostics；
- 用CUDA Event只建立Compute→Transfer和Install→Ready的必要依赖，清除热路径中的设备级`synchronize()`；
- 联合调优chunk字节、Ring slot数和`max_inflight_chunks`，避免chunk过小导致控制开销，或过大导致首块等待和Decode干扰；
- 验证传输/install对正在运行Decode的真实干扰，而不只测平均传输耗时；
- 多节点使用RDMA/NIXL/Mooncake类数据面替换SHM，Descriptor、receiver-first和credit协议保持上层语义不变。

验收指标：报告wire bytes、有效带宽、D2H/SHM/H2D/install分段时间、P计算与传输重叠比例、producer阻塞时间以及传输期间Decode TPOT增量。

### P5：调度、尾延迟和扩展性

- 治理TTFT p99，区分P队列、D prepare、transfer和client queue贡献；
- 用队列长度、预测Prefill成本、Decode负载、网络credit和SLO miss risk建立在线P:D/Hybrid角色建议器，再考虑自动角色数量调整；
- Prefix-aware路由不能只看Full Attention KV命中token，还要看目标worker是否拥有可恢复的GDN checkpoint以及状态安装成本；
- 对priority bias、aging、deadline urgency和short budget做真实负载校准，报告Long最大等待和Short尾延迟，验证无饥饿；
- 抢占victim分数加入已生成长度、重放成本、Prefix可恢复边界和deadline，避免只按优先级释放最昂贵的Long；
- 验证多rank DP Graph同步和worker故障后的请求重绑定；
- 完成TP state slicing和多rank测试；
- 分离数据面与控制面，引入service discovery、健康探测、拓扑感知路由和跨节点Prefix目录。

验收指标：在客服RAG、长摘要和代码分析三类trace上同时比较SLO goodput、TTFT/TPOT p50/p99、吞吐、Long饥饿率、角色空置率、抢占成本和故障恢复时间。

## 11. 面试数据备忘

| 项目 | 数值或口径 |
|---|---|
| 4B层数 | 32层：8 Full + 24 Linear |
| BF16 KV/token | 32KiB |
| 8K BF16 KV | 256MiB |
| 4B recurrent state | 53,477,376 bytes，约53.48MB |
| 8K总状态迁移 | 约309MiB |
| SHM带宽 | 约4.58GB/s |
| 8K迁移时间 | 约66ms |
| 32K历史Prefill优化 | 268.3s→11.3s，23.8× |
| H1 SLO提升 | 75%～106% |
| H1 TPOT p50 | 下降38%～52% |
| H1 TTFT p50 | 下降34%～46% |
| H1吞吐保持 | D0的92%～119% |
| R1 s42静态PD | Short SLO 11/48，约23% |
| 当前主要问题 | TTFT p99、INT8 gather fallback、SHM多producer背压、复现口径 |

## 12. 回答原则

每个性能数字至少准备回答以下问题：

1. baseline是什么；
2. 实验只修改了什么变量；
3. 模型、硬件、并发和数据集是什么；
4. 指标是整体还是Short分类，是p50还是p99；
5. 是否包含client queue；
6. Cache dtype和wire dtype分别是什么；
7. commit是否clean；
8. 原始结果文件在哪里。

比起只背“24×”或“106%”，能够解释边界条件、失败案例和未完成工作，更能证明项目是本人真实完成并深入理解的。
