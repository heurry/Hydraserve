# HydraServe

HydraServe 是一个面向 Qwen3.5/Qwen3.6 混合注意力模型（GDN + GQA）的
Prefill–Decode 分离推理引擎。当前主线按 [`main.md`](main.md) 从零实现，
不复用仓库此前的引擎代码；旧仓库完整内容保存在
`archive/pre-implementation-2026-08-13` 分支。

## 当前进度

推理内核与首个真实模型纵切片已经完成：

- 从 Hugging Face `config.json` 动态解析混合层布局和所有状态维度；
- 4B/9B/27B 只是预置，不是模型规模白名单；
- Paged KV block allocator 与固定槽 FP32 recurrent-state pool；
- 请求在 prefill 前原子预留最大输出所需 KV 页和 GDN state slot，避免流式输出中途
  才因容量不足失败；decode batch 的 KV 长度推进也是单事务；
- 真正的两值/byte grouped symmetric INT4 KV codec；
- 强校验的双状态传输描述符；
- `FULL`、`QUANTIZED`、`PARTIAL` 三种不同传输语义；
- 进程内测试后端与 POSIX shared-memory PARTIAL 后端；
- chunked prefill、first-token seeding、请求状态机、自适应路由；
- decode 端 PARTIAL KV 重算及状态安装的端到端 CPU 测试。
- 直接加载 sharded safetensors 的独立 Qwen text runtime；
- FlashAttention varlen GQA prefill；
- 自写 Triton RMSNorm、gated RMSNorm、causal conv、GDN recurrent rule；
- 自写 Triton Paged Attention 和 Paged KV scatter；
- chunked prefill 的物理页历史读取与 causal offset：首 chunk 可用 FlashAttention，
  continuation chunk（或禁用 Flash 时）走自写 Triton Paged online-softmax；
- 支持异构上下文长度的 Continuous Batching decode executor；
- Qwen3.5-4B BF16 真实 32 层 prefill/decode GPU smoke。
- Qwen3.5-9B BF16（独立 lm_head）真实 32 层 chunked prefill smoke；
- Qwen3.6-27B compressed-tensors AWQ/INT4 真实 64 层 prefill + decode smoke；
- 自写 Triton grouped asymmetric INT4 GEMM（packed weight/zero-point，group=128）；
- 独立 prefill/decode worker、N-1 truncation 与首 token 一致性校验；
- 真实双进程、双 GPU 的 SHM PARTIAL_TRANSFER 端到端链路；
- FULL/INT4 QUANTIZED KV 安装路径与真实物理页读取；
- CUDA P2P 后端及硬件能力检测（本机 NODE 拓扑无 peer access，自动回退 SHM）；
- 完整块粒度的 full-attention prefix radix cache（不错误缓存 GDN 状态）：支持
  model/tokenizer/revision/adapter 命名空间、引用保护、频率 doorkeeper、成本/大小/新鲜度
  淘汰评分、容量上限和有界频率元数据；已接入 PagedKVCache 的物理页引用计数、共享、
  写保护和回收，活跃请求容量不足时会先淘汰无引用低价值缓存页；
- ShareGPT、HumanEval、LongBench、WikiText-103、GSM8K 低内存数据适配器。

当前还包括驻留式 Continuous Batching 生成循环、直接读取 `tokenizer.json` 的文本
tokenizer、OpenAI-compatible completions/chat/SSE API，以及 TTFT/TPOT/P50/P95/P99
benchmark runner。HTTP 和 benchmark CLI 可在单 GPU collocated 与双进程双 GPU
PARTIAL PD 间切换，也支持同一常驻双 GPU 服务对每个请求动态选择 collocated 或 PD。
短请求直接在 decode worker 完成 prefill，长请求由 prefill worker 生成 GDN 状态后转交
decode worker；两条路径共享相同的 KV/GDN 准入与 continuous decode 生命周期。P2P 后端已实现，但当前两卡拓扑不支持 CUDA peer access，因此不能在本机伪装为
真实 P2P 实测；层级流水线也只完成协议和单测，不宣称已在 NVLink/P2P 上验证。

服务入口具有有界 admission queue（请求数与 token 双上限）。临时 KV/state 容量不足
会保持排队，单请求永久超过 worker 容量会单独失败，入口过载返回 HTTP 429。统一的
KV/state 容量快照供后续逐请求路由、worker 负载均衡和监控复用。

这里的“已实现”仍不等于整个系统已经达到生产完成态。抢占后的精确状态/KV 回放与
decode 故障域隔离已经实现；1P+ND（N>1）多卡实测、worker 自动恢复、采样语义、
压力与长稳验证仍在生产化路线中。

## 模型兼容性

兼容性由架构字段决定，不由参数量或模型名字决定。一个模型需要提供：

- 每层的 `layer_types`，或可推导的 `full_attention_interval`；
- `num_hidden_layers`、`hidden_size`；
- GQA 的 attention/KV head 数和 `head_dim`；
- GDN 的 key/value head 数、head dim 和 convolution kernel；
- `mamba_ssm_dtype=float32`。

加载器会自动识别 Qwen 多模态配置中的嵌套 `text_config`。检查本机模型目录：

```bash
python -m hydraserve inspect-models /mnt/nvme-data/models/LLM_model
```

当前机器上的 4B、9B、27B BF16、27B FP8 和 27B AWQ/INT4 配置均已通过检查。
同架构的其他参数规模可以直接通过其 `config.json` 接入；若内部维度不同，缓存和
循环状态形状会自动跟随配置变化。

“配置可识别”与“该权重格式已能执行”分开记录：BF16 runtime 已实跑 4B/9B；
compressed-tensors AWQ/INT4 已实跑 27B。27B BF16 约 52 GB，不能放入单张 24 GB
3090；本机 27B FP8 语言权重本身约 25.08 GiB，也超过可用显存，且 block-scaled
FP8 GEMM 尚未实现，loader 会明确拒绝而不是静默转 BF16 或调用外部后端。

27B AWQ 的 checkpoint 保留 GDN 投影为 BF16、量化 MLP/full-attention linear。
HydraServe 将只做 token lookup 的 embedding 留在 CPU，把独立 lm_head 和执行权重
放在 GPU，以约 22.02 GiB PyTorch allocation 完成 64 层 forward；INT4 权重在
GEMM 中即时解包/去零点/缩放，不生成完整反量化矩阵。

注意：真实 Qwen GDN recurrent state 按 value heads 保存，conv state 保存完整
Q/K/V depthwise-conv 通道。因此 FP32 双状态是 4B/9B 约 53.48 MB/请求，27B
约 158.86 MB/请求；早期设计文档中的 25/50 MB 估算偏低。

## 开发与测试

当前 CPU 协议层只依赖 NumPy：

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

本机环境包含一个依赖不完整的 ROS pytest 插件，因此示例显式关闭第三方插件自动加载。
GPU kernel 测试必须在 CUDA 可见环境运行；FlashAttention 测试使用安装了
`flash-attn` 的环境单独执行。

检查本机基准数据（大文件均流式读取，LongBench 不解压）：

```bash
python -m hydraserve inspect-datasets /mnt/nvme-data/datasets/benchmark --limit 1
```

当前目录中 `wikitext-103-raw.tar.gz` 和 `wikitext-103-test.csv` 是 0 字节，
加载器固定使用有效的 `wikitext-103-test.jsonl`。

启动文本 API（模型执行只使用 HydraServe runtime；`tokenizers` 仅用于文本编解码）：

```bash
python -m hydraserve serve /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  --device cuda:0 --port 8000 \
  --max-batch-size 64 --max-queue-size 1024 --max-queue-tokens 1048576

curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

双 GPU PARTIAL PD 服务使用同一 API：

```bash
python -m hydraserve serve /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  --pd --device cuda:0 --decode-device cuda:1 --port 8000
```

此模式中两个模型进程长期驻留：GPU0 做 prefill 并通过 SHM 传输 FP32 GDN 状态，
GPU1 重算 full-attention KV 后进入 Continuous Batching decode。

逐请求自适应模式：

```bash
python -m hydraserve serve /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  --adaptive --device cuda:0 --decode-device cuda:1 --port 8000
```

启用 full-attention Prefix KV 页缓存（容量以物理 block 数计，频率门禁默认 2）：

```bash
python -m hydraserve serve /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  --adaptive --device cuda:0 --decode-device cuda:1 \
  --prefix-cache-blocks 1024 --prefix-cache-min-frequency 2
```

命中页只复用 full-attention KV 的物理存储并写保护；GDN recurrent/conv state 不缓存，
仍逐请求精确重算。由于后续 GDN 层依赖前层输出，当前实现不宣称跳过整个命中 prefix
的模型计算；这是显存共享与 worker affinity 基础，不虚报为完整 prefix-compute skip。

路由在 admission 成功时绑定，依据 prompt 长度和统一 KV/GDN 容量快照决策，执行中不
改变归属。RPC 超时属于结果未知：当前请求失败并隔离 prefill 路径，后续请求安全降级到
collocated，不对同一请求进行可能重复执行的盲重试。`/health` 暴露容量，`/metrics`
输出 Prometheus 文本格式的队列、KV、state slot、路由和 worker 健康指标。

1P+ND 使用一个 prefill worker 和多个各自持有 KV/GDN 容量的 decode worker：

```bash
python -m hydraserve serve /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  --adaptive --device cuda:0 \
  --decode-devices cuda:1 cuda:2 cuda:3 --port 8000
```

worker registry 先过滤不健康或容量不足的目标，再联合 decode load、Prefix Cache
真实探测的匹配长度和链路带宽/跳数评分；预留成功后 worker binding 不再改变。一个 continuous
decode batch 跨多个 worker 时，各 GPU RPC 并行发起，结果按原请求顺序归并。本机仅有
两张 GPU，已真实验证新集群后端的 1P+1D 纵切片；1P+ND 的选择、绑定、分组和并发协议
已单测，N>1 真实硬件验证仍是明确门禁。

`--prefill-chunk-size` 控制 prompt 分块。Paged KV 会预留容量，但 attention 的逻辑
长度只推进到当前已写入 token，不会读取未来未初始化页。最后一个单 token chunk 与
多 token chunk 共用同一套 Paged 历史语义。

当前采样器只支持 greedy `temperature=0`，API 只处理文本。运行本机 benchmark：

```bash
python -m hydraserve benchmark \
  /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  /mnt/nvme-data/datasets/benchmark \
  --dataset gsm8k --limit 100 --concurrency 8 \
  --output benchmark_output/gsm8k.json
```

在相同命令后增加 `--pd --decode-device cuda:1` 可跑固定 PD，使用 `--adaptive` 可跑
逐请求混合路由。两请求
冷启动烟测已打通 collocated 与 PD；这种短 prompt 小样本中 PD 更慢，不能作为
crossover 或吞吐结论。

runner 支持 `--warmup` 排除首次 kernel 编译，并支持 `burst`、固定速率和 seeded
Poisson arrival trace。常驻 PD coordinator 会异步等待 GPU0 prefill，让 GPU1 继续
推进已有 decode；GPU1 安装新请求的重算阶段仍需与 decode 串行。
运行时 decode 采用事务式状态检查点：整批失败会先回滚逻辑 KV 长度和 GDN 状态，再
二分重试隔离单请求故障。1P+ND 后端按 worker 汇总部分结果，因此一个 decode worker
失败不会丢弃其他 worker 已成功生成的 token。

## 代码结构

```text
hydraserve/
├── config.py                 # 动态模型配置与预置
├── model/                    # safetensors loader 与独立 Qwen runtime
├── kernels/                  # reference、Triton 与 FlashAttention 边界
├── cache/                    # Paged KV、FP32 state pool、INT4 codec
├── transfer/                 # 描述符、后端、双状态 pipeline
├── engine/                   # chunked prefill、continuous batching、状态机
├── router/                   # 自适应 PD 路由
├── api/                      # OpenAI-compatible JSON/SSE HTTP 层
└── benchmark/                # 流式数据适配器、延迟与吞吐统计
```

设计目标、硬件数据和完整里程碑见 [`main.md`](main.md)。
