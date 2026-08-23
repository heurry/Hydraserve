<p align="center">
  <a href="README.md"><img alt="status" src="https://img.shields.io/badge/status-work%20in%20progress-yellow"></a>
  <a href="hydraserve/api/server.py"><img alt="serving" src="https://img.shields.io/badge/serving-HydraServe-00A3E0"></a>
  <a href="hydraserve/api/server.py"><img alt="api" src="https://img.shields.io/badge/API-OpenAI--compatible-brightgreen"></a>
  <a href="docs/BENCHMARK_2026-08-14.md"><img alt="benchmark" src="https://img.shields.io/badge/benchmark-ready-blue"></a>
  <a href="hydraserve/engine/multi_worker.py"><img alt="pd" src="https://img.shields.io/badge/PD-disaggregated-6F42C1"></a>
  <a href="hydraserve/kernels/"><img alt="kernels" src="https://img.shields.io/badge/kernels-Triton-orange"></a>
  <a href="hydraserve/config.py"><img alt="model" src="https://img.shields.io/badge/model-Qwen3.5%20%7C%20Qwen3.6-red"></a>
</p>

<p align="center">
  <img src="assets/hydraserve-logo.svg" alt="HydraServe" width="520">
</p>

面向混合注意力架构 LLM 的 Prefill–Decode 分离推理引擎原型,从零实现,不依赖
Transformers / vLLM 等外部引擎执行模型。参考架构为 Qwen3.5 / Qwen3.6 的
混合注意力(Gated Delta Network + GQA)。

## 项目定位

HydraServe 是一个引擎原型。它的目标不是证明"PD 分离比 DP 好",而是验证
"能否从零写出一个支持混合注意力 PD 分离的推理引擎"。核心价值在引擎实现与
混合注意力特有的双状态传输协议(KV Cache + 循环状态),不在某一次基准结论。

PD 分离真正发挥价值的场景是大规模部署(prefill 节点占比 <5% 的 100+ GPU
量级);本项目在消费级 GPU 上完成引擎与协议验证,设计目标可扩展到多节点。

与已有工作的区别:

- [vLLM + NIXL](https://github.com/vllm-project/vllm/pull/41869) /
  [vLLM + Mooncake](https://github.com/vllm-project/vllm/pull/46807) 已合并
  GDN PD 分离支持,但依赖 NIXL/Mooncake + RDMA;
- HydraServe 是独立实现,不依赖外部 connector,面向消费级 GPU 验证。

## 核心特性

### 运行时与 Kernel

- 直接加载 sharded safetensors 的独立 Qwen text runtime;
- 自写 Triton kernel:RMSNorm、gated RMSNorm、causal conv、GDN 递推规则、
  Paged Attention、Paged KV scatter;
- 自写 Triton grouped asymmetric INT4 GEMM(packed weight/zero-point,
  group=128)与 128×128 block-scaled E4M3FN GEMM;在无原生 FP8 Tensor Core 的
  GPU 上直接位解码,不展开常驻 BF16 权重;
- FlashAttention varlen GQA prefill(可选);禁用时走自写 Triton Paged
  online-softmax;
- chunked prefill,支持物理页历史读取与 causal offset。

### KV 与循环状态管理

- Paged KV block allocator 与固定槽 FP32 recurrent-state pool;
- 请求在 prefill 前原子预留最大输出所需 KV 页与 GDN state slot,避免流式输出
  中途因容量不足失败;
- 真正的两值/byte grouped symmetric INT4 KV codec;
- memory planner 按实际空闲显存规划可分配页数,支持 `--kv-headroom-blocks`
  预留与显式缩容报告;
- 块粒度 full-attention prefix radix cache:model/tokenizer/revision/adapter
  命名空间、引用保护、频率门禁、成本/大小/新鲜度淘汰、物理页共享与写保护;
  GDN 循环状态不缓存,始终精确重算。

### PD 分离与双状态传输

混合注意力模型在 prefill 与 decode 之间需要迁移两类状态:

| 状态 | 大小(32K context) | 特性 | 迁移方式 |
|------|-------------------|------|----------|
| Full-attention KV | 9B:约 1 GB BF16 / 345 MB INT4 | 线性增长,可量化 | FULL / QUANTIZED / PARTIAL |
| GDN 循环状态 | 4B/9B:53.48 MB;27B:158.86 MB | FP32 固定大小,不可量化 | 整体传输 |

- 强校验的双状态传输描述符,`FULL` / `QUANTIZED` / `PARTIAL` 三种传输语义;
  PARTIAL 只传循环状态,由 decode 端重算 full-attention KV;
- 传输后端抽象:进程内、POSIX shared memory、CUDA P2P(带硬件能力检测与
  自动回退);层级流水线协议(协议与单测,未实卡验证);
- 1P+ND 拓扑:一个 prefill worker + N 个各自持有 KV/GDN 容量的 decode worker,
  并行 RPC、按原请求顺序归并结果；支持按 prompt 阈值执行 conditional PD，
  short 在 D 本地 prefill+decode、long 才迁移状态。

### 调度与容错

- 异构上下文长度的 Continuous Batching decode;
- N-1 truncation 与首 token 一致性校验;跨 GPU 浮点 argmax 漂移不错误终止请求;
- 事务式 KV 长度推进与 GDN 状态检查点:整批失败先回滚,再二分重试隔离单请求;
- 优先级加权公平调度、等待老化防饿死、deadline urgency;
- prefill RPC 以关联 ID 多路复用；长 prefill 在 chunk 边界有界让出 GPU，处理
  P 卡上的 short admission/prefill/decode/release，避免串行 RPC 锁把 short TPOT
  阻塞到整段长 prefill 之后；可用 `--prefill-short-policy never` 做固定角色消融;
- 抢占与精确 replay(`prompt + generated[:-1]`),保留已输出 token 与采样状态;
  `--max-preemptions-per-request` 限制反复抢占;
- decode/prefill worker 监督:故障自动重建进程与 IPC,模型名/容量握手后重新
  加入,故障期间 fail-closed 降级到 collocated;
- 有界 admission queue(请求数与 token 双上限),超载返回 HTTP 429,单请求永久
  超容量单独失败,`timeout_ms` 硬 deadline 超时返回 408 / SSE `timeout_error`。

### 自适应路由

- 逐请求 cost-aware 路由:二次延迟曲线 + decode load 联合估计 collocated 与
  PD 成本,Schmitt-trigger 迟滞防抖动,在线 EWMA 校准,漂移超限 fail closed;
- `fit-router-profile` 从 concurrency-1 预热基准自动拟合路由 profile
  (非负约束,输出样本范围与 RMSE),无需手工调参。

### 服务与可观测性

- OpenAI-compatible API:`/v1/completions`、`/v1/chat/completions`、
  `/v1/models`、SSE 流式;
- 采样支持 `temperature`、`top_p`、`top_k`、`min_p`、repetition/presence/
  frequency penalty、逐请求 `seed`、最多 20 个 `logprobs`、最多 4 个文本 stop
  序列、`stream_options.include_usage`;未实现的字段(多 choice、tools、
  logit bias)会被明确拒绝;
- `/health` 与 `/metrics`(Prometheus 文本格式)暴露队列深度、KV/state 容量、
  路由校准、prefix cache、抢占与恢复计数、worker 健康等指标。

### 基准

- ShareGPT、HumanEval、LongBench、WikiText-103、GSM8K 流式低内存数据适配器;
- TTFT / TPOT / P50 / P95 / P99 延迟与吞吐统计;
- burst、固定速率、Poisson 到达 trace;`--warmup` 排除首次 kernel 编译。

## 架构概览

```text
OpenAI-compatible HTTP ──▶ 有界 Admission Queue ──▶ Cost-aware Router
                                                      │
                     ┌────────────────┬───────────────┼────────────────┐
                     │                │               │                │
              collocated      静态 PD 分离      自适应 PD           1P+ND
              (单卡)          (1P + 1D)        (逐请求路由)      (1P + N×D)
                                     │
                                     ▼
              Prefill Worker ──双状态传输(KV + GDN 循环状态)──▶ Decode Worker(s)
```

## 模型兼容性

兼容性由架构字段决定,不由参数量或模型名字决定。一个模型需要提供:

- 每层的 `layer_types`,或可推导的 `full_attention_interval`;
- `num_hidden_layers`、`hidden_size`;
- GQA 的 attention/KV head 数与 `head_dim`;
- GDN 的 key/value head 数、head dim 与 convolution kernel;
- `mamba_ssm_dtype=float32`(循环状态不可量化)。

加载器会自动识别 Qwen 多模态配置中的嵌套 `text_config`。4B/9B/27B 只是预置,
不是规模白名单;同架构的其他参数规模可直接通过其 `config.json` 接入,缓存与
循环状态形状会自动跟随配置变化。

已实跑验证的权重格式:

| 格式 | 已实跑模型 | 说明 |
|------|-----------|------|
| BF16 | Qwen3.5-4B / 9B | 完整 32 层 prefill + decode |
| AWQ / INT4(compressed-tensors) | Qwen3.6-27B | 完整 64 层;INT4 权重在 GEMM 中即时解包,不生成完整反量化矩阵 |
| block-scaled FP8(E4M3FN) | Qwen3.6-27B | 无原生 FP8 Tensor Core 时按 `uint8` 位模式手动还原有限 E4M3FN 值 |

超显存模型由 loader 按实际空闲显存选择最少的 host-streaming 投影,embedding
保留在 CPU。

## 安装

要求:Python ≥ 3.10;GPU 路径需要 NVIDIA GPU + CUDA。CPU 协议层只依赖 NumPy。

```bash
pip install -e '.[gpu]'        # PyTorch + Triton(GPU 执行)
pip install -e '.[prefill]'    # FlashAttention(可选,预填充加速)
pip install -e '.[serve]'      # tokenizers(HTTP 服务文本编解码)
pip install -e '.[dev]'        # pytest 等开发依赖
```

## 快速开始

检查模型目录中可识别的配置:

```bash
python -m hydraserve inspect-models /path/to/models
```

单卡 collocated 服务:

```bash
python -m hydraserve serve /path/to/Qwen3.5-4B \
  --device cuda:0 --port 8000 \
  --max-batch-size 64 --max-active-requests 256 \
  --max-queue-size 1024 --max-queue-tokens 1048576 \
  --kv-headroom-blocks 128

curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

静态 PD 分离(prefill 与 decode 各占一卡,两个模型进程常驻):

```bash
python -m hydraserve serve /path/to/Qwen3.5-4B \
  --pd --device cuda:0 --decode-device cuda:1 --port 8000
```

自适应模式:同一常驻双 GPU 服务对每个请求在 collocated 与 PD 间动态选择,
短请求直接在 decode worker 完成 prefill,长请求由 prefill worker 生成 GDN
状态后转交 decode worker:

```bash
python -m hydraserve serve /path/to/Qwen3.5-4B \
  --adaptive --device cuda:0 --decode-device cuda:1 --port 8000
```

1P+ND(一个 prefill worker + 多个 decode worker):

```bash
python -m hydraserve serve /path/to/Qwen3.5-4B \
  --adaptive --device cuda:0 \
  --decode-devices cuda:1 cuda:2 cuda:3 --port 8000
```

启用 full-attention prefix KV 页缓存(容量以物理 block 数计):

```bash
python -m hydraserve serve /path/to/Qwen3.5-4B \
  --adaptive --device cuda:0 --decode-device cuda:1 \
  --prefix-cache-blocks 1024 --prefix-cache-min-frequency 2
```

可选地加载路由 profile(`--router-profile configs/router/rtx3090-4b-shm-partial.json`),
或在相同模型/硬件/配置下用预热后的 concurrency-1 基准自动拟合(输入需覆盖至少
三个不同 prompt 长度):

```bash
python -m hydraserve fit-router-profile \
  --collocated benchmark_output/collocated-short.json benchmark_output/collocated-long.json \
  --pd-disaggregated benchmark_output/pd-short.json benchmark_output/pd-long.json \
  --output configs/router/my-profile.json
```

运行基准:

```bash
python -m hydraserve benchmark \
  /path/to/Qwen3.5-4B /path/to/datasets \
  --dataset gsm8k --limit 100 --concurrency 8 \
  --output benchmark_output/gsm8k.json
```

命令后追加 `--pd --decode-device cuda:1` 跑固定 PD,追加 `--adaptive` 跑逐请求
混合路由。

## 测试

```bash
python -m pytest
```

GPU kernel 与真实模型 smoke 测试需要在 CUDA 可见环境运行;FlashAttention
相关测试在安装了 `flash-attn` 的环境单独执行。

## 当前状态与路线图

本项目处于原型阶段。以下能力已通过真实 GPU 验证(2× RTX 3090 24 GB,无
NVLink,拓扑不支持 CUDA peer access):

- Qwen3.5-4B/9B BF16、Qwen3.6-27B AWQ/INT4 与 block-scaled FP8 的完整
  prefill + decode;
- 双进程、双 GPU 的 SHM PARTIAL 传输端到端链路;
- collocated、静态 PD、自适应路由三种服务模式。

以下能力已实现并覆盖单测,但尚未在真实硬件上验证:

- CUDA P2P 传输(开发拓扑不支持 peer access,自动回退 SHM);
- 1P+ND 的 N>1 多 decode worker;
- 层级流水线(NVLink/P2P 场景)。

压力与长稳验证、生产级多节点形态仍在路线图中。

## 文档

- [main.md](main.md) — 设计目标、硬件数据与完整里程碑;
- [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) — 逐切片实现记录;
- [docs/BENCHMARK_2026-08-14.md](docs/BENCHMARK_2026-08-14.md) — 路由决策实测分析;
- [scripts/cloud-4gpu/](scripts/cloud-4gpu/) — 4×RTX 3090 云端压测(1P+3D vs 4×单卡)脚本。
