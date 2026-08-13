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
- 自写 Triton 128×128 block-scaled E4M3FN GEMM；在无原生 FP8 Tensor Core 的
  RTX 3090 上手动解码位模式，并以最小 host-streaming 集合运行完整 27B FP8；
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

这里的“已实现”仍不等于整个系统已经达到生产完成态。抢占后的精确状态/KV 回放、
decode 故障域隔离、带老化的加权公平调度、decode worker 自动恢复和完整单 choice
采样语义已经实现；1P+ND（N>1）多卡实测、压力与长稳验证仍在生产化路线中。

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
compressed-tensors AWQ/INT4 和 block-scaled FP8 均已实跑 27B。27B BF16 约 52 GB，
不能放入单张 24 GB 3090；本机 27B FP8 语言权重约 25.08 GiB，同样超过可用显存。
loader 会把 embedding 和 lm_head 留在 CPU，并按实际空闲显存选择最少的一组大 FP8
投影做按需 host streaming，保留 1 GiB 给 recurrent state、KV 和激活；其余权重 GPU
常驻。RTX 3090 不支持原生 E4M3 Tensor Core，HydraServe Triton kernel 直接读取
`uint8` 位模式、手动还原有限 E4M3FN 值，再应用 128×128 inverse scale 和 BF16 dot，
不展开常驻 BF16 权重，也不调用外部推理后端。完整 64 层 prefill + decode 已实跑。

27B AWQ 的 checkpoint 保留 GDN 投影为 BF16、量化 MLP/full-attention linear。
HydraServe 将只做 token lookup 的 embedding 留在 CPU，把独立 lm_head 和执行权重
放在 GPU，以约 22.02 GiB PyTorch allocation 完成 64 层 forward；INT4 权重在
GEMM 中即时解包/去零点/缩放，不生成完整反量化矩阵。runtime 会把 `input_device`
显式暴露给 serving、PD 和恢复路径：AWQ token id 直接在 CPU 构造并查表，只将选中的
embedding row 传入 GPU，不再先创建 GPU token tensor、每步同步回 CPU 后再查表。

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
  --max-batch-size 64 --max-active-requests 256 \
  --max-queue-size 1024 --max-queue-tokens 1048576 \
  --kv-headroom-blocks 128

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

路由在 admission 成功时绑定，执行中不改变归属。默认 SHM/PARTIAL 路由使用二次延迟
曲线联合估计 collocated 与 PD 成本，加入收益下限和 PD 不确定性惩罚，并按 prompt
长度桶用真实 prefill 延迟做 EWMA 修正；不再把 8K 等固定阈值当成普遍 crossover。
RPC 超时属于结果未知：当前请求失败并隔离 prefill 路径，后续请求安全降级到
collocated，不对同一请求进行可能重复执行的盲重试。`/health` 暴露容量和路由校准状态，
`/metrics` 输出 Prometheus 文本格式的队列、KV、state slot、路由成本观测和 worker
健康指标。

可用 `--router-profile configs/router/rtx3090-4b-shm-partial.json` 显式加载 profile。
JSON 分别提供 `collocated` / `pd_disaggregated` 的 fixed、linear、quadratic、decode-load
系数，以及最小绝对/相对收益和风险倍率。不同模型、传输后端或硬件应使用各自测量得到
的 profile；固定 PD 实验仍使用 `--pd`，不会被 adaptive 路由覆盖。

profile 不需要手工拟合。把相同模型/硬件/配置下、预热后的 concurrency-1 benchmark
结果分别交给以下命令；输入必须覆盖至少三个不同 prompt 长度，失败请求会被排除，输出
同时携带样本范围和 RMSE：

```bash
python -m hydraserve fit-router-profile \
  --collocated benchmark_output/collocated-short.json benchmark_output/collocated-long.json \
  --pd-disaggregated benchmark_output/pd-short.json benchmark_output/pd-long.json \
  --output configs/router/my-profile.json
```

拟合器对 fixed/linear/quadratic 系数施加非负约束，避免噪声生成随长度下降或最终变为负数
的延迟曲线。低负载样本拟合基础曲线；新 runner 记录 admission 时的
`route_decode_load`，可选并发 trace 会单独拟合非负 `decode_load_scale`，不会污染基础
长度曲线。当前只是一阶外部性模型，尚未显式表示 prefill queue/inflight 顺序。

成本路由使用 Schmitt-trigger 式迟滞：从当前 route 切换需要跨过收益门槛再加迟滞带，
减少边界附近抖动。在线 EWMA 修正连续达到 `drift_min_observations` 且偏离 profile 超过
`drift_ratio_threshold` 时，默认 fail closed 到 collocated；`/health` 变为 degraded，
`hydraserve_route_cost_profile_drift` 置 1。benchmark 每条结果同时记录 admission
decode load、内部 request/admission 顺序、submission→admission wait、预测与实测
prefill queue wait、以及 executor 内直接测得的 prefill service time，便于重放和重新
拟合。profile fitter 优先使用直接 service 观测；旧 JSON 才回退到 TTFT−queue。

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

采样状态贯穿 collocated、PD 与 1P+ND worker，支持 `temperature`、`top_p`、`top_k`、
`min_p`、repetition/presence/frequency penalty、逐请求 `seed`、最多 20 个
`logprobs` 以及最多四个文本 stop 序列。stop token 会参与 usage 计数但不会泄漏到普通
响应或 SSE 文本；流式接口支持 `stream_options.include_usage`。当前 API 只处理文本，
且明确拒绝多 choice、tools、logit bias 等尚未实现的字段。运行本机 benchmark：

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

runner 支持 `--warmup` 排除首次 kernel 编译，并在 warmup 后清空在线路由 EWMA/迟滞
状态，避免把 CUDA/Triton 冷编译误学成 profile drift；模型与 kernel 保持热驻留。它还支持 `burst`、固定速率和 seeded
Poisson arrival trace。常驻 PD coordinator 会异步等待 GPU0 prefill，让 GPU1 继续
推进已有 decode；GPU1 安装新请求的重算阶段仍需与 decode 串行。
benchmark 结果会记录每个请求的实际 route、route reason、worker binding、两条预测
成本、预计收益和校准置信度，并汇总 route counts。2026-08-14 的短 prompt 矩阵与 9K LongBench 实测见
[`docs/BENCHMARK_2026-08-14.md`](docs/BENCHMARK_2026-08-14.md)：在本机 SHM
PARTIAL 模式下，静态 8K 阈值会错误偏向 PD，后续路由必须纳入 KV 重算与传输成本。
运行时 decode 采用事务式状态检查点：整批失败会先回滚逻辑 KV 长度和 GDN 状态，再
二分重试隔离单请求故障。1P+ND 后端按 worker 汇总部分结果，因此一个 decode worker
失败不会丢弃其他 worker 已成功生成的 token。

服务循环可分别配置同时驻留请求数与单步 decode batch 大小。batch 选择使用优先级加权
公平分数，并通过等待老化防止低优先级请求饿死；临时容量不足的请求会回到候选队列，
不会阻塞后续可准入请求。decode 子进程退出或 RPC 超时后会先从路由摘除，重建进程与
IPC 队列，并在模型名和容量握手通过后重新加入。故障 worker 上的全部旧绑定会原子
失效；在途请求保留已输出历史，等待健康 worker 后重新绑定并用精确 replay 恢复，而
不是因为设备状态丢失直接向客户端报错。健康 worker 同批已成功的 token 不受影响。
adaptive 1P+ND coordinator 也会在 admission 和 RPC 等待期间检查 prefill 子进程；故障
时新请求立即 fail-closed 到 collocated，后台用新 IPC 队列重载模型，握手成功后自动恢复
PD 路由。`/health` 和 `/metrics` 暴露 decode/prefill 健康、恢复中状态、重启计数及 fault
suspension 数。

相同的进程监督现在也覆盖旧的单 decode-device `--adaptive` 和静态 `--pd`。adaptive
在 prefill 重载期间保持 collocated 服务；静态 PD 没有可用的 collocated 路由语义，因此
请求保留在 admission 队列并在 prefill 恢复后继续。decode 进程丢失时，两者都会使旧
KV/GDN 所有权失效，并对已经输出 token 的请求执行精确 replay，而不是等待完整 RPC
timeout 或直接丢失请求。

`--max-active-requests` 控制已准入并持有 KV/GDN 容量的请求数，必须不小于
`--max-batch-size`；后者只控制单步真正进入 decode kernel 的数量。两者分离后，调度器
可以在一个 batch 之外保留等待 decode 的活跃请求，并做 priority-weighted fairness、
老化防饿死和 deadline urgency。API 的 HydraServe 扩展字段 `timeout_ms` 是从 submit
开始计算的硬 deadline；在 admission/prefill/decode 边界过期会释放资源，非流式返回
HTTP 408，SSE 返回 `timeout_error` event。GPU kernel 不可中断，因此 deadline 是 kernel
边界协作式而非微秒级抢占。collocated 主链路会让新到达的更高优先级或更早 deadline
请求在 decode 迭代边界抢占低紧迫度请求，立即释放其 KV/GDN 容量；受害请求随后用
`prompt + generated[:-1]` 精确重算，保留已经输出的 token、采样 step 和停止序列状态。
`--max-preemptions-per-request`（默认 2，设为 0 可关闭）限制反复抢占。
`/health` 与 `/metrics` 暴露 admission、prefill、active、preempted 四层深度，以及抢占和
恢复的成功/失败累计值。异步 PD 与 1P+ND 主链路使用同一个 prefill executor 提交恢复，
decode worker 通过独立 `recover` RPC 在本地重算并安装状态，不重新采样或发出历史 token；
同一 decode worker 上的恢复与 decode 由 RPC 锁串行，不同 worker 的 decode 仍可并行。

`--kv-headroom-blocks` 从 admission 可用容量中永久保留指定物理页，防止工作集逼到最后
一页时反复准入/失败；默认 0 保持向后兼容。headroom 仍是已分配的 GPU KV tensor，
但普通 request 和 prefix cache 压力都不能侵占它。容量不足时先按成本策略淘汰未引用
prefix 页，只回收满足请求所需的数量；仍不足则事务式拒绝，现有 allocation/refcount
不变。

`--cache-tokens` 是物理 KV 的目标上限。模型权重加载后，memory planner 按实际空闲显存、
KV block 字节数、至少一个完整 GDN 事务槽、512 MiB CUDA hard reserve 和 64 MiB allocator
guard 计算可安全分配的页数；不能兑现时明确缩容并通过 health/metrics 暴露 requested、
planned、allocated bytes 和 clamped 状态。FP8 loader 会提前按该目标增加最少的
host-streamed 投影，尽量先兑现用户的 cache 配置。PD prefill worker 也使用同一套物理
Paged KV，不再以逐 chunk `torch.cat` 累积 dense K/V；每个请求发送完状态后立即释放页。

allocator 维护物理/可准入页、高水位、allocation failure、active allocation、共享页、
总引用、logical/reserved token 和 block-tail 内部碎片。Paged KV `audit()` 对账
request block references + prefix owners = allocator total references，并检查 free list、
refcount、reservation 和 prefix ownership。Prefix Cache 指标区分 active pressure、cache
capacity、manual 淘汰，以及 frequency/capacity/size/length 拒绝；`/health` 返回完整快照，
`/metrics` 暴露稳定低基数标签。

当前 PCIe fallback 使用不经 pickle 的 typed ndarray 单-envelope SHM，header 在内容写完后
才发布；GDN 状态通过 pinned host staging 搬运。decode worker 按实时剩余显存计算可保证
的 FP32 state slot 数，并预分配 layer-major 连续 GPU pool。每轮 decode 只构造一次
Paged KV metadata；页表和长度先在 host 打包，再以两个连续 tensor 上传，避免随 batch
线性增长的小 CUDA update。full-attention 层使用单次 batched Triton KV scatter；Paged
Attention 按 16-token tile 做 online softmax。RTX 3090 微基准中 batched scatter 相比逐请求
launch 在 batch 1/8/32 分别为 1.15×/10.25×/42.79×；连续 metadata 构造相比旧逐行更新
在 batch 1/8/32/64 分别为 1.52×/10.55×/19.00×/34.50×。这些是微基准，不代表端到端吞吐。
GDN decode 使用按最大 decode batch 预分配的跨层事务工作区，以整批 gather/commit 代替
每层 `cat` 和逐请求回拷；最终 logits 成功前不发布状态。真实 4B state shape 下 batch
1/4/8/16 的搬运加速为 1.83×/1.39×/1.36×/1.22×，并消除每步
52/204/408/816 MiB 临时分配。工作区槽数和 storage/workspace 字节数可在监控中查看。
N−1 replay 使用 prefill 端已传输的首 token 作为权威输出；跨 GPU 浮点 argmax 漂移不会
错误终止请求，但会计入 `hydraserve_pd_replay_mismatches_total` 供诊断。

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
