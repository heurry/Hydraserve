# HydraServe 升级、优化与性能排障日志（2026-08-22）

本文记录本轮 HydraServe 仓库同步、代码优化、正确性验证、单卡性能测试、异常现象、
根因分析及后续建议。它是一次工程过程日志，不只记录最终成功项，也保留中间出现的
冷启动、基准口径和 CUDA Graph 形状碎片化问题，供后续复现和继续优化。

## 1. 本轮目标与最终结论

本轮工作起点是根据仓库根目录的
[对比升级.md](../对比升级.md) 所列建议，优先处理能够安全落地且可以在本机验证的
高影响问题：

1. 将本地 `main` 快进同步到远端最新代码；
2. 修复 INT8 KV 写入阻断 CUDA Graph 的问题；
3. 合并 QKV、GDN 和 MLP 的兼容投影，减少 decode 热路径 GEMM 次数；
4. 补充 CPU/CUDA 正确性测试；
5. 使用本地历史结果复测单卡 C1/C4，并解释 C4 的性能下降。

最终结论：

- 仓库已同步到 `origin/main` 的 `23ea9c0`；
- INT8 KV decode 已可以进入 CUDA Graph，固定 128-token C1 测试中吞吐提升
  **5.7%**、TPOT P50 降低 **7.1%**；
- 投影融合在当前代码 C1 直接 A/B 中带来 **2.4%** 吞吐提升，TPOT P50 降低
  **3.9%**；
- C4 相对历史基线的 `-14.7%` 并不是投影融合本身造成，主要原因是短测试在计时区间
  内首次捕获多种 CUDA Graph；同一当前代码关闭 Graph 后达到 **148.22 tok/s**，
  比历史 C4 的 124.88 tok/s 高 **18.7%**；
- 原分析中“INT8 CUDA Graph 可带来 2～3 倍整体提升”属于方向性估计，本轮实测没有
  支持这一幅度。当前单卡短上下文的净收益约为 5%～7%，不能把预估当作验收结果。

## 2. 仓库同步与变更边界

### 2.1 同步结果

- 远端：`https://github.com/heurry/Hydraserve.git`
- 分支：`main`
- 同步后的提交：`23ea9c09b49c84fd785947cd9982a460cdba3408`
- 提交摘要：`Add vLLM comparison: 4xDP on same load A beats HydraServe`
- 当前状态：`main...origin/main`，没有落后或领先远端的提交。

同步时保留了工作区已有的修改和未跟踪基准文件，没有执行 `reset --hard`、清理工作区
或覆盖用户文件。本轮实现目前仍是未提交修改。

### 2.2 本轮实现文件

运行时代码：

- [hydraserve/cache/paged_kv.py](../hydraserve/cache/paged_kv.py)
- [hydraserve/kernels/activation.py](../hydraserve/kernels/activation.py)
- [hydraserve/kernels/kv_cache.py](../hydraserve/kernels/kv_cache.py)
- [hydraserve/model/runtime.py](../hydraserve/model/runtime.py)

新增或扩展的测试：

- [tests/test_paged_kv.py](../tests/test_paged_kv.py)
- [tests/test_runtime.py](../tests/test_runtime.py)
- [tests/test_triton_kernels.py](../tests/test_triton_kernels.py)

`docs/LEARNING_PATH.md` 在本轮开始前已经存在本地修改，本轮运行时优化没有覆盖或回退
它。`benchmark_output/` 下原有历史结果也全部保留。

## 3. 改进一：INT8 KV CUDA Graph 兼容

### 3.1 原问题

INT8 KV decode 写入原先分成两部分：

1. CUDA kernel 写入量化后的 key/value；
2. Python 按请求循环，通过 `positions[row].item()` 和
   `block_table[...].item()` 求物理页，再写 key/value scale。

`.item()` 会把 GPU tensor 值同步回主机，形成 host sync；Python 循环本身也不能被
CUDA Graph 捕获。因此 runtime 对 `kv_quant == "int8"` 做了显式绕过，INT8 decode
始终退回 eager 路径。

### 3.2 实施改进

新增 `write_paged_kv_batch_quantized` Triton 路径：

- grid 使用 `(batch, heads)`；
- 从设备端 `positions` 和 `block_table` 直接计算物理块与块内偏移；
- 同一次 kernel launch 写入 INT8 key、INT8 value、key scale 和 value scale；
- 移除 CUDA 热路径中的逐请求 `.item()` 与 Python scale scatter；
- CPU fallback 改为 tensor 化索引写入，不再逐行调用 `write()`。

同时扩展 `PagedKVCache.raw_layer_cache()`：

- BF16/FP16 KV 返回 key/value 两个物理页 tensor；
- INT8 KV 返回 key/value 和两份 scale，共四个 tensor；
- CUDA Graph 首次捕获前后的快照/恢复因此会同时覆盖数据页和 scale 页，避免捕获
  warmup 污染真实请求状态。

最后移除 runtime 中 `kv_quant != "int8"` 的 Graph 入口限制，使 INT8 在满足其余条件时
可以使用 `_decode_batch_graph()`。

### 3.3 正确性覆盖

新增测试覆盖：

- CPU/CUDA 两端的 INT8 batched KV scatter；
- 量化误差范围内 read-back 与原始 key/value 一致；
- raw cache 包含 INT8 数据页和 FP32 scale 页；
- Triton 量化 scatter 能被 CUDA Graph 捕获并重放；
- 一个同时包含 GDN 和完整注意力层的小模型，在 INT8 KV 下 Graph 与 eager 的
  logits、循环状态一致；
- 测试同时断言 Graph 确实捕获成功，而不是静默 fallback。

### 3.4 收益与限制

固定 synthetic 128-token prompt、128 output tokens、C1 的实测：

| 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|
| INT8 eager / Graph off | 53.001 | 57.95 ms | 18.55 ms | 2414.03 ms |
| INT8 Graph on | 56.047 | 57.60 ms | 17.24 ms | 2249.83 ms |
| 变化 | **+5.7%** | -0.6% | **-7.1%** | **-6.8%** |

这证明修复有效，但 Graph 收益取决于能否复用已经捕获的形状。首次使用新形状时仍然
会支付完整捕获成本；这一点后来成为 C4 回退的主要原因。

## 4. 改进二：投影 GEMM 合并

### 4.1 原问题

Qwen3.5/3.6 decode 中多个线性层共享同一个输入，但原实现分别发起 GEMM：

- 完整注意力：Q、K、V 共 3 次 GEMM；
- GDN/线性注意力：QKV、Z、B、A 共 4 次 GEMM；
- MLP：gate、up 共 2 次 GEMM。

decode 的矩阵高度通常只有 batch size，这些小 GEMM 更容易受 kernel launch 和矩阵
形状效率影响。

### 4.2 实施改进

runtime 初始化时构造兼容的合并权重：

- 完整注意力 `q + k + v`：3 次 GEMM 降为 1 次，输出后按宽度 split；
- GDN `qkv + z`、`b + a`：4 次 GEMM 降为 2 次；
- MLP `gate + up`：2 次 GEMM 降为 1 次。

兼容性设计：

- dense BF16/FP16 权重直接沿输出维拼接；
- `BlockScaledFP8Weight` 拼接 data 与 scale，不重新量化；
- `PackedInt4Weight` 拼接 packed data、scale 和 zero-point，不重新量化；
- 不兼容的设备、dtype、block size、group size 或形状会自动保留 unfused 路径；
- 原 checkpoint 权重名仍保留为 fused storage 的轻量 view，便于形状检查和诊断；
- `fuse_projections=False` 可作为正确性、性能和兼容性逃生开关。

### 4.3 正确性覆盖

新增测试验证：

- fused 与 unfused 完整前向 logits 一致；
- GDN recurrent 和 convolution state 一致；
- 完整注意力 QKV 确实只调用一次 `_linear()`；
- GDN 两组投影只调用两次 `_linear()`；
- MLP 包含 down projection 在内只调用两次 `_linear()`；
- FP8/INT4 合并再 split 后，数据、scale 和 zero-point 与原权重一致。

### 4.4 实际收益

当前代码的直接 fused/unfused A/B：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | unfused | 59.191 | 35.04 ms | 14.35 ms | 255.50 ms |
| C1 | fused，两次稳态均值 | 60.621 | 36.27 ms | 13.79 ms | 247.64 ms |
| C1 | fused 变化 | **+2.4%** | +3.5% | **-3.9%** | **-3.1%** |
| C4 | unfused，两次均值 | 109.115 | 142.44 ms | 29.90 ms | 580.09 ms |
| C4 | fused，两次稳态均值 | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms |
| C4 | fused 变化 | -2.3% | **-8.7%** | **-3.9%** | **-0.6%** |

C4 中 fused 的中位 TPOT 和请求延迟更好，但总吞吐略低。这一矛盾后来定位为 Graph
首次捕获和动态调度改变尾请求完成时刻，并不表示 fused GEMM 在 batch 4 普遍更慢。

## 5. 测试验证记录

### 5.1 全量非实卡测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xdu/anaconda3/envs/deepseek/bin/python -m pytest -q
```

结果：

```text
187 passed, 76 skipped
```

连续运行两次，分别在 1.72 秒和 1.67 秒完成，结果一致。

### 5.2 关键 CUDA 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xdu/anaconda3/envs/deepseek/bin/python -m pytest \
  tests/test_triton_kernels.py::test_quantized_paged_kv_scatter_is_cuda_graph_safe -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xdu/anaconda3/envs/deepseek/bin/python -m pytest \
  tests/test_triton_kernels.py::test_int8_kv_cuda_graph_decode_matches_eager -q
```

两项均通过。第一项验证 kernel capture/replay，第二项验证完整 decode 的 Graph/eager
数值与状态一致。

## 6. 单卡基准口径

### 6.1 硬件与软件

- GPU：RTX 3090 24 GiB，使用 GPU 0；排查时未发现 GPU 0 有其他计算负载；
- Python：`/home/xdu/anaconda3/envs/deepseek/bin/python`；
- 模型：`/mnt/nvme-data/models/LLM_model/Qwen3.5-4B`，BF16；
- 数据集：`/mnt/nvme-data/datasets/benchmark` 下 GSM8K；
- FlashAttention：关闭，以保持历史口径一致；
- cache tokens：8192；
- measured：8 条请求，每条最多 16 个输出 token；
- warmup：2 条请求；
- prompt cap：512。

历史与当前 C1/C4 的基础命令只有 `--concurrency` 不同：

```bash
/home/xdu/anaconda3/envs/deepseek/bin/python -m hydraserve benchmark \
  /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  /mnt/nvme-data/datasets/benchmark \
  --dataset gsm8k --limit 8 --warmup 2 \
  --max-new-tokens 16 --max-prompt-tokens 512 \
  --concurrency 1 \
  --cache-tokens 8192 --no-flash-attention \
  --output benchmark_output/<result>.json
```

### 6.2 历史基线与当前总体结果

| 单卡并发 | 版本 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | 2026-08-14 历史基线 | 47.330 | 48.87 ms | 17.46 ms | 310.93 ms |
| C1 | 当前 fused 稳态 | 60.621 | 36.27 ms | 13.79 ms | 247.64 ms |
| C1 | 相对历史变化 | **+28.1%** | **-25.8%** | **-21.0%** | **-20.4%** |
| C4 | 2026-08-14 历史基线 | 124.878 | 108.13 ms | 25.33 ms | 501.79 ms |
| C4 | 当前 fused + Graph | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms |
| C4 | 相对历史变化 | **-14.7%** | +20.2% | +13.4% | +14.9% |

注意：历史对比包含 8 月 14 日到当前之间的所有仓库变化，不能把 C1 的 28.1% 全部
归因于本轮投影融合。本轮融合的净收益应以前一节同代码 fused/unfused A/B 为准。

## 7. 遇到的问题与排查过程

### 7.1 问题一：首次优化运行异常慢

第一次 C1 fused 运行只有 39.875 tok/s，但随后两次分别为 60.570 和 60.672 tok/s。

原因是首次进入新权重形状、Triton kernel 和 CUDA Graph 路径时需要编译、autotune 或
捕获；这些成本污染了短基准。后两次结果几乎一致，因此报告使用后两次均值作为当前
稳态值。

经验：必须区分以下三种状态，不能只报一次数字：

- 全新进程、全新 kernel cache 的冷启动；
- kernel 已编译但 Graph 尚未覆盖当前形状；
- kernel 和常用 Graph 形状都已预热的长驻服务稳态。

### 7.2 问题二：第一次 INT8 Graph A/B 几乎没有吞吐提升

最初用长度不同的 GSM8K prompt、最多 128 output tokens 做 C1 A/B：

| 模式 | output tok/s | TTFT P50 | TTFT P95 | TPOT P50 |
|---|---:|---:|---:|---:|
| Graph off | 53.542 | 40.37 ms | 52.29 ms | 18.48 ms |
| Graph on | 53.642 | 47.34 ms | 358.56 ms | 17.18 ms |

Graph 的 TPOT 已改善，但首次捕获不同 block-table width 抬高 TTFT 尾部，抵消了总吞吐
收益。随后改用固定 128-token synthetic prompt 隔离 replay 收益，得到稳定的 5.7%
吞吐提升。

经验：CUDA Graph 性能测试必须控制 shape；变长 workload 需要把捕获成本单独报告。

### 7.3 问题三：C4 相对历史下降 14.7%

初始假设包括：

- 合并后的大 GEMM 在 RTX 3090、小 batch 下效率更差；
- fused/unfused 改变调度相位和尾请求；
- CUDA Graph 捕获失败，回退 eager；
- GPU 0 有其他任务竞争；
- 历史与当前代码跨版本，不是隔离对比。

逐项排查结果如下。

#### 7.3.1 投影微基准排除“batch 4 GEMM 普遍变慢”

使用 Qwen3.5-4B 真实 BF16 权重、RTX 3090、CUDA event 测量；每项预热 30 次、测量
300 次。下表为 fused 相对 separate 的速度变化，正值表示 fused 更快：

| batch | Full QKV | MLP gate+up | GDN QKV+Z | GDN B+A |
|---:|---:|---:|---:|---:|
| 1 | -1.45% | +9.19% | -4.13% | +103.36% |
| 2 | +31.49% | +2.39% | +16.44% | +98.58% |
| 3 | +21.37% | +2.84% | +16.58% | +100.14% |
| 4 | +21.52% | +2.76% | +16.30% | +101.17% |

只有 batch 1 的 Full QKV 和 GDN QKV+Z 略慢；batch 2～4 的主要投影均受益，因此它
无法解释 C4 的 14.7% 总回退。

#### 7.3.2 Graph 捕获全部成功，没有静默 fallback

对一次 C4 运行增加只读跟踪后，共观察到以下 Graph key：

```text
(1, 4), (1, 3), (1, 5), (4, 5),
(3, 5), (3, 8), (4, 8), (1, 6)
```

key 的含义是 `(batch_size, block_table_width)`。所有捕获都成功，因此不是捕获失败后
反复 fallback 的问题，而是成功捕获的形状太多。

#### 7.3.3 根因：warmup 不覆盖 C4，捕获成本进入计时区间

benchmark runner 的 `--warmup 2` 会串行提交两条请求，不能形成 batch 3/4。因此它只
能预热 batch 1 图。正式 C4 测量采用 burst 提交，ContinuousGenerationLoop 又在每轮
先 admission/prefill、再从当前 active 请求中选 decode batch，于是实际 batch 会在
1、3、4 之间变化；不同 prompt 长度进一步产生多个 block-table width。

第一轮代码中，每个新 Graph key 的首次捕获会：

1. clone recurrent、convolution、KV 数据页和 INT8 scale 页；
2. 完整执行 3 次 decode warmup；
3. 再完整执行 1 次 graph capture；
4. 恢复所有被 warmup 修改的状态；
5. 最后才 replay 真实请求。

8 个观测 key 中，前两个 batch-1 key 来自串行 warmup；其余 6 个在正式计时阶段首次
出现，相当于额外执行最多 24 次完整 decode transaction，尚未计算状态 clone/copy
成本。对于只有 8 请求 × 16 output tokens 的测试，这个成本无法摊薄。

#### 7.3.4 Graph on/off 隔离结果

同一当前 fused 代码，仅设置 `HYDRASERVE_CUDA_GRAPH=0`：

| C4 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---:|---:|---:|---:|---:|
| 当前 fused + Graph | 106.578 | 130.01 ms | 28.72 ms | 576.35 ms | 约 1.20 s |
| 当前 fused + Graph off | **148.225** | **92.45 ms** | **21.42 ms** | **423.69 ms** | **0.864 s** |
| 历史 2026-08-14 | 124.878 | 108.13 ms | 25.33 ms | 501.79 ms | 1.025 s |

当前 Graph on 相对 Graph off 吞吐低约 **28.1%**；Graph off 又比历史高约 **18.7%**。
这直接证明 C4 的历史回退主要来自 Graph 冷捕获。

历史 C4 文件生成于 8 月 14 日，而 CUDA Graph decode 路径在 8 月 16 日的提交
`ac871a1` 才加入，所以历史结果没有这项捕获成本。

#### 7.3.5 为什么 TPOT 中位改善但吞吐仍下降

C4 fused/unfused 的逐请求比较中，8 条请求有 6 条在 fused 下 TPOT 或延迟更好；主要
异常集中在 sample 6 和 sample 9。更快的局部 step 改变了 prefill admission 与 decode
batch 的时间相位，使最后一波请求进入了不同的 Graph key/批次组合。

benchmark 的 output throughput 是：

```text
所有成功输出 token 数 / 从提交第一条到最后一条完成的总 wall time
```

因此它由最后完成的尾请求决定，而不是由 TPOT P50 决定。出现“多数请求更快、P50
更好，但最后两条拖长导致总吞吐更低”是可能的。

### 7.4 问题四：临时 CUDA 微基准首次无法导入仓库包

投影微基准脚本放在 `/tmp` 后，Python 的模块搜索路径不包含仓库根目录，首次运行报
`ModuleNotFoundError: hydraserve`。在临时脚本中显式加入仓库路径后完成测试。该问题
没有修改仓库代码，也没有影响正式结果。

### 7.5 已排除的外部原因

- GPU 0 基准期间没有发现竞争计算负载；
- GPU 1 的桌面/其他负载与 GPU 0 基准无关；
- C4 的所有 Graph shape 都捕获成功；
- fused 与 unfused 数值正确性测试通过；
- 当前 Graph-off 结果显著超过历史基线，说明硬件、模型或数据路径没有整体退化。

## 8. 当前结论的适用范围

本轮性能结果只代表：

- 单张 RTX 3090；
- Qwen3.5-4B BF16；
- GSM8K 短 prompt 或固定 128-token synthetic；
- FlashAttention 关闭；
- C1/C4、小样本 burst；
- 本机当前驱动、PyTorch、Triton 和 kernel cache 状态。

它不能直接外推到：

- 4 卡 DP 或 2P+2D；
- Qwen3.6-27B AWQ/FP8；
- 长上下文或持续高负载；
- FlashAttention 开启；
- 已完整预捕获全部 Graph bucket 的长驻服务。

特别是 Graph 在长生成、固定 shape 或长驻服务中可能在摊薄捕获成本后获益，不能根据
这次 16-token C4 结果断言“C4 永远应该关闭 Graph”。

## 9. 本轮未实施的建议

以下建议仍然有效，但因为影响面、依赖或缺少独立正确性基线，本轮没有直接实施：

- fused residual add + RMSNorm；
- fused SiLU-and-mul；
- `torch.compile`/Inductor 全图编译；
- 更成熟的 FlashAttention/paged decode attention 路径；
- eager 路径 positions、metadata 和临时 tensor 的完全预分配；
- 热路径 `_weight()` 查找和 Python type dispatch 缓存；
- continuous batching 的统一 token-budget 调度；
- DP rank 间 wave/padding/Graph 同步；
- PD chunk/layer transfer overlap；
- staging buffer fused gather/scatter；
- CPU-GPU overlap scheduler；
- HiCache、TP-aware transfer 和 bootstrap/control-plane 重构。

这些项目应逐项建立同代码 A/B 与正确性测试，不能将分析报告中的估计收益直接相加。

## 10. 后续优先级建议

### P0：修复 CUDA Graph 生命周期与基准预热

1. 增加并发感知 warmup，让 C4 warmup 实际形成 batch 2/3/4；
2. 预热常见 `(batch_size, block_table_width)` bucket；
3. **已完成：**新 shape 第一次出现时先走 eager，在出现次数达到阈值后再捕获；
4. 考虑后台捕获，避免在真实请求关键路径中执行 3+1 次完整模型；
5. **已完成：**对 block-table width 做 2 的幂 bucket 化，降低 key 数量；
6. 分别报告 cold、kernel-warm/graph-cold、fully-warm 三种性能。

短突发 C4 在修复前可临时设置：

```bash
HYDRASERVE_CUDA_GRAPH=0
```

### P1：扩大性能验收负载

- measured 请求至少 100 条；
- output tokens 至少 128；
- 同时测试 burst 和稳定 arrival rate；
- 记录每 step 的 batch-size 直方图、block-table width、graph hit/miss/capture 时间；
- 固定 GPU 时钟/功耗状态，并记录驱动、PyTorch、Triton 版本；
- 每种配置至少运行 3～5 次，报告均值和标准差。

### P2：继续评估投影融合

真实 GEMM 微基准显示 batch 2～4 总体受益，暂时没有证据支持在 C4 关闭融合。建议先
解决 Graph 捕获问题，再用端到端长稳态负载评估是否需要：

- 对极小 batch 的某些 QKV shape 使用 unfused；
- 为 fused GEMM 做 shape-specific autotune；
- 进一步融合 SiLU-and-mul、residual+RMSNorm。

### P3：INT8 与多卡验证

- 运行 INT8 C4 的 Graph on/off 长稳态对照；
- 复测原报告中的 4×DP Load A/Load B；
- 复测 2P+2D 的 TTFT/TPOT 和状态传输；
- 检查不同 DP rank 的 Graph shape 是否需要 padding/synchronization。

## 11. 复现命令

### 11.1 C4 Graph-off 根因对照

```bash
HYDRASERVE_CUDA_GRAPH=0 \
  /home/xdu/anaconda3/envs/deepseek/bin/python -m hydraserve benchmark \
  /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  /mnt/nvme-data/datasets/benchmark \
  --dataset gsm8k --limit 8 --warmup 2 \
  --max-new-tokens 16 --max-prompt-tokens 512 \
  --concurrency 4 --cache-tokens 8192 --no-flash-attention \
  --output benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_fused_graph_off.json
```

### 11.2 INT8 固定形状 Graph-on

```bash
/home/xdu/anaconda3/envs/deepseek/bin/python -m hydraserve benchmark \
  /mnt/nvme-data/models/LLM_model/Qwen3.5-4B \
  /mnt/nvme-data/datasets/benchmark \
  --dataset synthetic --num-short 8 --short-tokens 128 \
  --warmup 2 --max-new-tokens 128 --concurrency 1 \
  --cache-tokens 8192 --kv-quant int8 --no-flash-attention --seed 42 \
  --output benchmark_output/2026-08-22_synthetic128_4b_c1_int8_graph_on.json
```

Graph-off 对照在同一命令前增加 `HYDRASERVE_CUDA_GRAPH=0`。

## 12. 结果与文档索引

总体单卡报告：

- [2026-08-22_single_gpu_optimization_report.md](../benchmark_output/2026-08-22_single_gpu_optimization_report.md)

历史基线：

- [C1 historical](../benchmark_output/2026-08-14_gsm8k_4b_collocated_c1.json)
- [C4 historical](../benchmark_output/2026-08-14_gsm8k_4b_collocated_c4.json)

当前 fused 稳态：

- [C1 run 2](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c1_optimized_run2.json)
- [C1 run 3](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c1_optimized_run3.json)
- [C4 run 2](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_optimized_run2.json)
- [C4 run 3](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_optimized_run3.json)

当前 unfused 对照：

- [C1 unfused](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c1_current_unfused.json)
- [C4 unfused run 1](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_current_unfused.json)
- [C4 unfused run 2](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_current_unfused_run2.json)

Graph/INT8 对照：

- [C4 fused Graph-off](../benchmark_output/2026-08-22_gsm8k_4b_collocated_c4_fused_graph_off.json)
- [INT8 fixed Graph-on](../benchmark_output/2026-08-22_synthetic128_4b_c1_int8_graph_on.json)
- [INT8 fixed Graph-off](../benchmark_output/2026-08-22_synthetic128_4b_c1_int8_graph_off.json)

## 13. 第二轮继续优化：Graph 策略与融合 activation

完成 C4 根因分析后，继续按高优先级清单实施并验证了下一批优化。

### 13.1 Graph shape bucket

`PagedKVCache.batch_metadata()` 新增可选的 `bucket_width`：

- eager 和普通 prefill 默认仍使用精确宽度；
- Graph 路径把 block-table width 向上归并到 2 的幂；
- 观测到的 width 3/4/5/6/8 因此归并为 4/8；
- padding 区保持 `-1`，attention 根据真实 sequence length 只访问有效逻辑块。
- 延迟捕获阶段将已构造的 bucket metadata 直接交给 eager transaction，避免同一步重复
  构造和传输 table/lengths。

这样减少 Graph key 数量，又不需要每个 decode step 在 GPU 上额外 pad/copy；bucket 在
host metadata 构造阶段一次完成。

### 13.2 延迟 Graph 捕获

runtime 新增每个 `(batch_size, bucketed_width)` 的观察计数：

- 默认同一 shape 观察 16 次后才捕获；
- 低频、一次性动态 shape 直接走 eager，不再阻塞真实请求做完整捕获；
- 高频长稳态 shape 达到阈值后转入 Graph replay；
- capture 前完整 warmup 从 3 次降为 1 次，降低首次捕获成本。

可调环境变量：

```bash
HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER=16
HYDRASERVE_CUDA_GRAPH_WARMUP_STEPS=1
```

两个值都必须是正整数。设置 `HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER=1` 可恢复“首次
出现立即捕获”，主要用于 Graph 正确性测试和特殊的固定 shape 服务。

### 13.3 Fused SiLU-and-mul

新增 Triton `silu_and_mul(gate, up)`，将 MLP 中：

```text
gate * sigmoid(gate)
再乘 up
```

合并为一次 kernel launch，FP32 中间计算后写回输入 dtype。CPU/reference 路径保持原
PyTorch 实现。

### 13.4 Fused GDN gating

新增 Triton `gdn_gating(beta, step, A_log, dt_bias)`，一次 kernel 同时计算：

```text
beta_out = sigmoid(beta)
decay = -exp(A_log) * softplus(step + dt_bias)
```

softplus 使用稳定形式 `max(x, 0) + log(1 + exp(-abs(x)))`，输出 beta/decay 均为
FP32，直接供 GDN recurrent kernel 使用。

### 13.5 未实施清单 #8 的原因

`torch.split` 对连续 tensor 通常返回 view，不会像报告所述自动产生“多次 copy kernel”。
因此没有为 QKV split 增加复杂的自定义 kernel；后续只有在 profiler 证明 reshape、head
展开和数据布局确实产生 materialization 时才值得融合。

### 13.6 第二轮测试结果

- 新增 power-of-two metadata bucket 测试；
- 新增 Graph 默认阈值、环境变量覆盖和非法值测试；
- 新增两种 shape 的 SiLU-and-mul CUDA 对齐测试；
- 新增两种 shape 的 GDN gating CUDA 对齐测试；
- INT8 CUDA Graph 端到端测试显式设置 capture-after=1，继续验证真实捕获；
- CPU 相关测试、6 个 CUDA 定向用例和全量测试均通过；
- `git diff --check` 通过。

### 13.7 第二轮端到端性能

默认配置两次稳定均值：

| 并发 | 第一轮默认 | 第二轮最终当前 | 变化 |
|---|---:|---:|---:|
| C1 output tok/s | 60.621 | **65.940** | **+8.8%** |
| C1 TTFT P50 | 36.27 ms | **35.04 ms** | **-3.4%** |
| C1 TPOT P50 | 13.79 ms | **13.43 ms** | **-2.5%** |
| C1 Latency P50 | 247.64 ms | **243.46 ms** | **-1.7%** |
| C4 output tok/s | 106.578 | **146.272** | **+37.2%** |
| C4 TTFT P50 | 130.01 ms | **92.41 ms** | **-28.9%** |
| C4 TPOT P50 | 28.72 ms | **21.73 ms** | **-24.3%** |
| C4 Latency P50 | 576.35 ms | **428.88 ms** | **-25.6%** |

第二轮 C4 默认结果已接近显式 Graph-off。新增 fused activation/GDN kernel 的隔离
Graph-off A/B 为：

| C4 Graph-off | output tok/s | TPOT P50 | Latency P95 |
|---|---:|---:|---:|
| 第一轮投影融合 | 148.225 | 21.42 ms | 449.49 ms |
| 加 activation/GDN 融合 | **150.226** | **21.14 ms** | **443.15 ms** |

kernel 本身贡献约 **1.35%** 吞吐提升；C4 的主要 37.2% 改善来自 Graph 捕获策略。

相对 2026-08-14 历史基线，最终当前 C1/C4 吞吐分别提高约 **39.3%/17.1%**。

### 13.8 INT8 长稳态复测

固定 128-token synthetic C1：

| 最终当前模式 | output tok/s | TTFT P50 | TTFT P95 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|---:|
| Graph off | 54.651 | 56.51 ms | 62.27 ms | 17.98 ms | 2339.84 ms |
| Graph on | **58.262** | 56.54 ms | 62.07 ms | **16.84 ms** | **2195.14 ms** |

长稳态中 Graph 净吞吐收益为 **6.6%**。与第一轮 Graph-on 相比，吞吐从 56.047
提高到 58.262 tok/s；更重要的是 TTFT P95 从 244.62 ms 降到 62.07 ms，说明低频
shape 不再在测量请求中制造捕获长尾。

新增原始结果：

- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_run1.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_run2.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_run1.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_run2.json`
- `2026-08-22_gsm8k_4b_c4_fused_kernels_graph_off.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_policy_fused_kernels_on.json`
- `2026-08-22_synthetic128_4b_c1_int8_graph_policy_fused_kernels_off.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_final.json`
- `2026-08-22_gsm8k_4b_c1_graph_policy_fused_kernels_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_final.json`
- `2026-08-22_gsm8k_4b_c4_graph_policy_fused_kernels_final_run2.json`

## 14. 工程经验总结

1. 优化必须同时有正确性测试、同代码开关 A/B 和历史总体对比；三者回答的问题不同。
2. CUDA Graph “捕获成功”不等于“当前负载更快”，捕获生命周期和 shape 命中率同样重要。
3. 小样本吞吐容易被最后一条请求和首次编译/捕获支配，P50 变好不保证总 wall time 变短。
4. kernel 微基准用于判断算子本身，端到端基准用于判断调度、捕获和尾延迟；两者不能互相替代。
5. 对比报告中的收益数字应视为待验证假设，不能在没有实测时直接作为完成标准。
6. 当前最值得优先修复的不是继续堆更多融合，而是 Graph 的并发预热、shape 管理和捕获时机。

## 15. 第三轮：残差归一化候选与采样热路径

### 15.1 fused residual-add + RMSNorm 候选被否决

按升级清单 #5 实现并验证了 in-place residual add + RMSNorm Triton 候选。正确性测试和
全量回归均通过，但同代码开关 A/B 没有显示稳定的端到端收益：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 |
|---|---|---:|---:|---:|---:|
| C1 | unfused | 65.721 | 35.11 ms | 13.47 ms | 245.22 ms |
| C1 | fused | 65.969 | 34.92 ms | 13.42 ms | 243.28 ms |
| C4 | unfused（两次均值） | **147.577** | **91.14 ms** | **21.49 ms** | **424.59 ms** |
| C4 | fused（两次均值） | 147.526 | 91.52 ms | 21.64 ms | 427.55 ms |

C1 只有约 0.38% 吞吐差异；C4 吞吐持平，但 TPOT 和 latency 均回退约 0.7%。因此不把
这个 kernel 留在默认代码中，相关源码和测试已回退，只保留 benchmark JSON 作为否决证据。
可能原因是当前逐层 RMSNorm 的行数很小，省掉一次 residual-add launch 的收益不足以覆盖
in-place 写 residual 带来的依赖和访存限制。

原始结果：

- `2026-08-22_gsm8k_4b_c1_add_rmsnorm_unfused.json`
- `2026-08-22_gsm8k_4b_c1_add_rmsnorm_fused.json`
- `2026-08-22_gsm8k_4b_c4_add_rmsnorm_unfused.json`
- `2026-08-22_gsm8k_4b_c4_add_rmsnorm_unfused_run2.json`
- `2026-08-22_gsm8k_4b_c4_add_rmsnorm_fused.json`
- `2026-08-22_gsm8k_4b_c4_add_rmsnorm_fused_run2.json`

### 15.2 默认贪心采样批处理

升级清单 #21～23 指出的采样问题在默认 benchmark 上全部存在。旧逻辑对 batch 中每一行：

1. clone 完整的 151,936 维 logits；
2. 即使惩罚参数全为默认值，也遍历 history 并构造 GPU token/frequency tensor；
3. 即使没有请求 logprobs，也计算全词表 `log_softmax`；
4. 分别执行 `int(scores.argmax())`，每行产生一次 host sync。

当前实现为纯贪心请求增加批量快速路径。仅当 temperature=0、无 repetition/presence/
frequency penalty 且未请求 logprobs 时，直接对 `[batch, vocab]` 做一次批量 `argmax`，再用
一次 `tolist()` 回传全部 token。带温度采样、惩罚和 logprobs 的请求继续走原语义路径；
其中也把 `log_softmax` 延迟到确实请求 logprobs 时才执行，并跳过默认值的惩罚准备。

可通过下列开关恢复旧逐行路径，主要用于 A/B 和应急回退：

```bash
HYDRASERVE_BATCHED_GREEDY=0
```

151,936 词表、RTX 3090 的 100 次采样微基准：

| batch | 旧逐行路径 | 批量快速路径 | 变化 |
|---:|---:|---:|---:|
| 1 | 0.0333 ms | **0.0229 ms** | **-31.2%** |
| 4 | 0.1177 ms | **0.0238 ms** | **-79.8%** |

GSM8K 端到端严格开关 A/B（每种模式两次均值）：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---|---:|---:|---:|---:|---:|
| C1 | 旧逐行 | 66.720 | 35.14 ms | 13.24 ms | 240.52 ms | 1.9185 s |
| C1 | 批量 | 66.742 | 35.23 ms | 13.24 ms | 240.53 ms | 1.9178 s |
| C4 | 旧逐行 | 149.798 | 92.50 ms | 20.99 ms | 418.64 ms | 0.8545 s |
| C4 | 批量 | **151.039** | 93.64 ms | **20.86 ms** | **418.41 ms** | **0.8475 s** |

C1 吞吐变化 +0.03%，可视为持平；C4 吞吐提高 **0.83%**、TPOT P50 降低 **0.58%**、
wall time 降低 **0.82%**。TTFT P50 的 +1.14 ms 与采样微基准方向不一致，而且两次短测的
运行间噪声大于这一差值；吞吐、TPOT 和 wall time 的方向与微基准一致，因此保留快速路径
默认开启。

测试新增覆盖：批量快速路径选择、环境变量回退、无需 logprobs 时不执行 `log_softmax`；
采样定向测试 6 个及全量测试均通过。

新增原始结果：

- `2026-08-22_gsm8k_4b_c1_batched_greedy_off.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_off_run2.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_on.json`
- `2026-08-22_gsm8k_4b_c1_batched_greedy_on_run2.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_off.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_off_run2.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_on.json`
- `2026-08-22_gsm8k_4b_c4_batched_greedy_on_run2.json`

## 16. P1 一次性收敛：GDN blocked、decode specialization 与 fused split

### 16.1 P1 逐项审计

原对比文档将 P1 标为“每 step 叠加 80 层”，但本次实际模型 Qwen3.5-4B 为 32 层：
24 个 linear-attention/GDN 层和 8 个 full-attention 层。逐项状态如下：

| # | 项目 | 最终状态 |
|---:|---|---|
| 2 | 标准 attention QKV 三 GEMM | 已合并为一次 fused QKV projection |
| 3 | GDN QKV+Z+B+A 四 GEMM | 已合并为 QKVZ + BA 两次 projection |
| 4 | MLP gate/up 两 GEMM | 已合并为一次 projection |
| 5 | residual-add + RMSNorm | 实现并测试过；C4 无净收益且延迟回退，源码已回退 |
| 6 | SiLU-and-mul | 已使用 Triton fused kernel |
| 7 | GDN beta/decay | 已使用 Triton fused gating kernel |
| 8 | conv 后 Q/K/V split/layout | 本轮改为 conv kernel 直接写三个连续输出 |
| 9 | GDN token recurrence | 本轮新增 decode sequence=1 专用 kernel；长 prefill 仍精确串行 |
| 10 | causal-conv 过细 grid | 本轮改为 256-channel blocked grid |

因此本轮没有重新启用已经被数据否决的 #5，也没有把 `torch.split` view 错报为 copy；而是
进一步消除了 view 后为满足 recurrent 连续布局产生的真实 copy。

### 16.2 blocked causal-conv

Qwen3.5-4B 的 linear conv width 是 6,912。旧 kernel 每个 program 只计算一个
`(batch, token, channel)` 标量，decode sequence=1 时主 kernel 和 state kernel 合计为每个
请求、每层 13,824 个 program 实例。新 kernel 每个 program 向量化处理 256 channels：

- channel blocks 从 6,912 降为 `ceil(6912/256)=27`；
- 两个 kernel 合计从 13,824 降为 54 个 program 实例，即减少 256 倍；
- kernel launch 数仍是两次，但调度粒度和连续访存显著改善；
- kernel 同时可直接写连续 Q、K、V tensor，不再先写完整 mixed tensor 再复制布局。

### 16.3 compact heads 与 decode recurrent specialization

模型有 16 个 Q/K heads、32 个 value heads。旧路径对 Q/K 各执行一次
`repeat_interleave(ratio=2)`，物理扩展到 32 heads。新 recurrent kernel 直接用
`value_head // ratio` 映射 compact Q/K head：

- Q/K 搬运量减半；
- 去掉两次 repeat-interleave 物理复制；
- state、decay、beta 和 value 仍按 32 value heads 更新，数学语义不变；
- sequence=1 decode 使用无 `tl.range` 动态循环的专用 kernel；
- sequence>1 prefill 继续使用原精确 recurrence。真正的 chunk-parallel affine scan 需要单独
  的算法级实现和长上下文验证，本轮没有虚报为已完成。

整组旧路径可通过以下进程级环境变量恢复：

```bash
HYDRASERVE_GDN_KERNEL=legacy
```

该开关同时恢复标量 conv grid、mixed+split 布局、repeat-interleave 和动态 recurrent kernel，
用于 A/B 与应急回退。

### 16.4 正确性与 Graph 验证

新增和扩展 CUDA 测试覆盖：

- causal-conv sequence=1/2/7 与 PyTorch reference 对齐；
- blocked conv 直接写三个连续 split，sequence=1/7 对齐；
- recurrent sequence=1/3/17 对齐；
- compact 2→4 head 映射在 sequence=1/5 与显式 repeat-interleave 对齐；
- supplied next-state buffer 继续原位复用；
- INT8 KV CUDA Graph 捕获、replay 和 eager 对齐继续通过；
- 完整 Triton 测试通过（仅未安装 FlashAttention 的用例跳过），全量 CPU 测试通过。

### 16.5 最终同代码 A/B

GSM8K、2 warmup + 8 measured、16 output tokens，每种模式两次反序均值：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---|---:|---:|---:|---:|---:|
| C1 | legacy | 66.690 | 34.81 ms | 13.28 ms | 240.92 ms | 1.9193 s |
| C1 | P1 final | **71.654** | **24.55 ms** | **12.89 ms** | **220.85 ms** | **1.7864 s** |
| C1 | 变化 | **+7.44%** | **-29.47%** | **-2.92%** | **-8.33%** | **-6.93%** |
| C4 | legacy | 150.284 | 93.60 ms | 20.96 ms | 417.04 ms | 0.8517 s |
| C4 | P1 final | **176.147** | **68.43 ms** | **18.93 ms** | **353.59 ms** | **0.7267 s** |
| C4 | 变化 | **+17.21%** | **-26.90%** | **-9.70%** | **-15.21%** | **-14.68%** |

相对 2026-08-14 历史基线，当前最终 C1/C4 吞吐分别累计提高约 **51.4%/41.1%**。
本轮收益在 C4 更大，符合旧 scalar conv program 数和 repeat-interleave 搬运随 batch 放大的
预期。

原始结果：

- `2026-08-22_gsm8k_4b_c1_gdn_p1_legacy.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_legacy_run2.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_final.json`
- `2026-08-22_gsm8k_4b_c1_gdn_p1_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_legacy.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_legacy_run2.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_final.json`
- `2026-08-22_gsm8k_4b_c4_gdn_p1_final_run2.json`

中间隔离结果 `*_gdn_p1_blocked.json` 尚未包含 fused conv split，用于确认 split/layout
继续贡献了小幅增益，不作为最终汇总值。

## 17. P2 第一批：runtime 固定开销与 P2P stream wait

### 17.1 P2 范围复核

P2 #11～#23 混合了三类问题：collocated decode 固定开销、采样开销和每请求一次的 PD
传输开销。第一批优先完成能直接作用于每个 decode step、且可独立验证的项目：

| # | 项目 | 第一批状态 |
|---:|---|---|
| 11 | 每 step 构造 CUDA metadata tensor | positions、state slot_ids 已复用固定 buffer；KV table 暂不全局复用 |
| 12 | `_weight()` 热路径字典查找 | 已绑定为初始化期 per-layer slots cache |
| 13 | `_norm/_linear` 分支与动态 import | 热路径 kernel import 已移除；量化 weight 类型 dispatch 保留 |
| 14 | GQA `repeat_interleave` | 已由 P1 compact GDN head mapping 完成 |
| 15 | 冗余 `contiguous()` | 已移除可证明的 no-op；真实 split/layout copy 保留或由 fused split 消除 |
| 16 | logits 无条件 `.float()` | 已改为 native dtype；采样慢路径按需 FP32 |
| 17～19 | SHM codec/poll 与 state host staging | 留作独立 PD 传输协议阶段 |
| 20 | P2P receive CPU event sync | 已改为 CUDA stream `wait_event` |
| 21～23 | 采样逐行、clone/logsoftmax、逐行 sync | 已在上一轮批量贪心优化完成 |

原报告把 #17～#19 也归为“每 step 固定开销”不准确：它们位于 PD 状态/KV 传输边界，通常
每请求一次，而不是每个 decode token 一次。直接去掉 #19 的同步会让随后 `.numpy()` 在 DMA
完成前读取数据，属于正确性错误，因此需要 future/event-aware bundle API 后才能异步化。

### 17.2 native logits 与按需 FP32

旧 runtime 在每次 output projection 后把整个词表 logits 转为 FP32。当前 Qwen3.5-4B 词表
为 248,320，batch=4 时单步额外写入约 3.8 MiB FP32 logits。新路径默认保留 projection 的
BF16 dtype：

- 纯贪心快速路径直接对 BF16 logits 做 batched argmax；
- 温度采样、惩罚和 logprobs 路径仍在 `_sample_row(row.float())` 中按需转 FP32；
- CUDA Graph static logits buffer 同步改为模型 dtype；
- `HYDRASERVE_FP32_LOGITS=1` 可恢复旧行为。

最终代码严格开关 A/B（每种模式两次反序均值）：

| 并发 | logits | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---|---:|---:|---:|---:|---:|
| C1 | FP32 | 71.573 | 24.81 ms | 12.92 ms | 221.76 ms | 1.7884 s |
| C1 | native BF16 | **71.788** | **24.62 ms** | **12.88 ms** | **220.67 ms** | **1.7830 s** |
| C1 | 变化 | **+0.30%** | **-0.77%** | **-0.35%** | **-0.50%** | **-0.30%** |
| C4 | FP32 | **177.918** | 69.56 ms | **18.81 ms** | **351.38 ms** | **0.7194 s** |
| C4 | native BF16 | 177.836 | **68.84 ms** | 18.81 ms | 351.46 ms | 0.7198 s |
| C4 | 变化 | -0.05% | **-1.04%** | +0.02% | +0.02% | +0.05% |

C4 吞吐/TPOT 在测量噪声内持平，但移除了固定转换和一半 static logits 显存，因此保留 native
dtype 默认路径。

### 17.3 per-layer weight slots 与固定 metadata buffer

runtime 初始化完成 projection fusion 后，权重不再变化。新增 slots dataclass 把每层 input/
post norm、MLP、full attention 或 linear attention 权重一次性绑定：

- decode 热路径不再拼接权重名称字符串；
- 不再查询总权重 dict 或判断 fused weight 名称是否存在；
- `_weight()` 只在初始化、融合和 checkpoint 校验阶段使用；
- 原 `weights` mapping 继续保留，诊断和 checkpoint 名称兼容性不变。

positions 按 batch size 复用 CPU staging + device buffer；state pool 的 slot_ids 按 workspace
capacity 复用同类 buffer。复用指针测试、异构 sequence value 更新和 CUDA Graph replay 均通过。
KV block table/lengths 没有贸然放入共享全局 buffer，因为不同 CUDA stream 并行 decode 时会
形成写后读竞争，后续应设计 stream-local workspace。

### 17.4 P2P receive 不再阻塞 CPU

旧 `CudaP2PTransferBackend.receive(stream=None)` 调用 `event.synchronize()`。新实现选择调用方
指定 stream，未指定时选择目标设备 current stream，并执行 `stream.wait_event(event)`：

- 数据依赖留在 GPU stream 上；
- CPU 可继续调度后续工作；
- 消费 kernel 在同一 stream 上仍严格等待 copy 完成。

当前两张 RTX 3090 的 `torch.cuda.can_device_access_peer(0, 1)` 为 False，不能在本机跑真实
P2P copy；测试通过受控 event/stream 验证默认路径只建立 stream dependency。正式 P2P
吞吐/overlap 需要在支持 peer access 的拓扑复测。

### 17.5 第一批总体结果

最终 native 两次均值相对上一轮 P1 final 两次均值：

| 并发 | P1 final | P2 第一批 | 吞吐变化 |
|---|---:|---:|---:|
| C1 | 71.654 tok/s | **71.788 tok/s** | **+0.19%** |
| C4 | 176.147 tok/s | **177.836 tok/s** | **+0.96%** |

P2 固定开销的端到端收益明显小于 P1 kernel 重构，符合模型计算仍占主导的预期。相对
2026-08-14 历史基线，当前 C1/C4 吞吐累计提高约 **51.7%/42.4%**。

新增原始结果：

- `2026-08-22_gsm8k_4b_c1_p2_final_fp32.json`
- `2026-08-22_gsm8k_4b_c1_p2_final_fp32_run2.json`
- `2026-08-22_gsm8k_4b_c1_p2_buffers.json`
- `2026-08-22_gsm8k_4b_c1_p2_final_native_run2.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_fp32.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_fp32_run2.json`
- `2026-08-22_gsm8k_4b_c4_p2_buffers.json`
- `2026-08-22_gsm8k_4b_c4_p2_final_native_run2.json`

`*_p2_native_logits.json`、`*_p2_runtime_cleanup.json` 和 `*_p2_weight_cache.json` 为逐项
隔离过程结果，不作为最终双次均值。

## 18. P3：边际清理、硬件边界与负收益候选回退

### 18.1 实际完成项

P3 清单经源码复核后只合入两项低风险修改：

- SSE 实际实现位于 `hydraserve/api/server.py::_sse`，而不是旧报告所写的
  `scripts/dp_proxy.py`。`BaseHTTPRequestHandler` 默认 `wbufsize=0`，使用无缓冲
  `_SocketWriter`；其 `write()` 直接发送，`flush()` 为空操作。删除逐事件 flush 后，每个
  token 仍单独写入 socket，流结束时仍保留最终 flush，因此没有引入定时批量造成的额外延迟。
- runtime 权重预算、paged KV memory planner 和 `PagedKVCache._bytes_per_block()` 中的
  `torch.empty((), dtype=dtype).element_size()` 全部改为 `dtype.itemsize`。这去掉初始化阶段
  为查询 dtype 大小创建临时 tensor 的开销，不影响每 token 数值路径。

### 18.2 GDN output 预分配：实测否决并撤回

曾实现 shape-local 常驻 workspace，把 causal-conv 的三个 Q/K/V 输出和 recurrent 输出作为
目标 tensor 传入 Triton kernel。使用同一模型、数据、warmup 和生成长度，按“普通分配 →
预分配 → 预分配 → 普通分配”的反序顺序各测两次，均值如下：

| 并发 | 模式 | output tok/s | TTFT P50 | TPOT P50 | Latency P50 | wall time |
|---|---|---:|---:|---:|---:|---:|
| C1 | caching allocator | **71.910** | **24.48 ms** | **12.87 ms** | **220.05 ms** | **1.7800 s** |
| C1 | GDN 预分配 | 71.580 | 24.70 ms | 12.95 ms | 221.13 ms | 1.7882 s |
| C1 | 变化 | **-0.46%** | +0.92% | +0.63% | +0.49% | +0.46% |
| C4 | caching allocator | **178.211** | **69.87 ms** | **18.77 ms** | **353.56 ms** | **0.7183 s** |
| C4 | GDN 预分配 | 175.784 | 76.59 ms | 18.82 ms | 357.52 ms | 0.7282 s |
| C4 | 变化 | **-1.36%** | +9.62% | +0.30% | +1.12% | +1.38% |

结论：PyTorch CUDA caching allocator 已复用这些短生命周期 allocation；额外的每层 workspace
查询和常驻多形状 buffer 没有抵消分配成本，反而增加固定开销和显存生命周期。该实现、环境
开关及专用测试已经完整撤回，A/B JSON 保留作为负结果证据。

### 18.3 未在当前硬件伪实现的项目

- 原生 FP8 `tl.dot` 方向依赖 Hopper 级 FP8 Tensor Core。当前两张 GPU 均为 RTX 3090、
  compute capability 8.6；现有 BF16 activation + FP8 weight 解量化 fallback 保留。
- Marlin 是面向 INT4/AWQ 的架构专用 GEMM，不是对现有 Triton kernel 的小型局部替换。
  当前端到端基准使用 BF16 Qwen3.5-4B，也不会执行 AWQ kernel。后续应以真实 AWQ checkpoint、
  独立正确性 oracle 和 Ampere/Hopper 分架构 benchmark 作为专项实施条件。

### 18.4 最终性能与验证

撤回 GDN 预分配后的最终代码各运行两次：

| 并发 | P2 第一批 | P3 最终 | 吞吐变化 | TTFT P50 | TPOT P50 | Latency P50 |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 71.788 tok/s | 71.475 tok/s | -0.44% | 25.72 ms | 12.95 ms | 222.20 ms |
| C4 | 177.836 tok/s | 177.895 tok/s | +0.03% | 68.26 ms | 18.82 ms | 350.97 ms |

P3 两项保留修改分别作用于 HTTP SSE Python 调用和初始化期 dtype 查询，理论上不改变模型
decode 吞吐；C1/C4 的 -0.44%/+0.03% 视为短基准噪声，不声明吞吐收益。完整 pytest 以及
Triton CUDA 回归均通过。

新增原始结果：

- `2026-08-22_gsm8k_4b_c1_p3_gdn_alloc.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_alloc_run2.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_prealloc.json`
- `2026-08-22_gsm8k_4b_c1_p3_gdn_prealloc_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_alloc.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_alloc_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_prealloc.json`
- `2026-08-22_gsm8k_4b_c4_p3_gdn_prealloc_run2.json`
- `2026-08-22_gsm8k_4b_c1_p3_final.json`
- `2026-08-22_gsm8k_4b_c1_p3_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_p3_final.json`
- `2026-08-22_gsm8k_4b_c4_p3_final_run2.json`

## 19. 第五节 14 项改进的完整实施

### 19.1 Kernel 与 runtime（#1-4）

- #1、#2 已在前述 P0/P1 阶段完成：INT8 KV Graph-safe scatter，以及 QKV/gate_up/GDN
  projection weight fusion。
- #3 新增 opt-in `torch.compile`。对手写 Triton、paged cache 和 Python state transaction
  直接 fullgraph 会触发 Dynamo graph break、CUDA Graph overwritten-output guard，且 4B Inductor
  冷编译超过三分钟。因此最终采用分层策略：Triton production runtime 编译纯 tensor dense MLP
  子图；无 Triton runtime 可选择完整 forward/decode transaction scope。默认关闭，并用
  `HYDRASERVE_TORCH_COMPILE_BACKEND=eager` 完成真实 4B GPU 接线 smoke。失败/中止的
  `compile_smoke_run4` 至 `run7` 保留为边界证据，不计入性能结果。
- #4 修正 `paged_flash_prefill`：FlashAttention 直接接收物理
  `[num_blocks, block_size, kv_heads, dim]` cache 和 int32 block table，删除此前会复制整个 cache 的
  `unsqueeze/expand/contiguous`。首 chunk 走 varlen FA，所有 continuation chunks 走 paged FA。
  后续已安装 `flash-attn 2.8.3.post1`，varlen、物理 cache 零复制和 paged output 数值对照三条
  RTX 3090 CUDA 测试全部通过；完整安装与 A/B 见第 20 节。

### 19.2 PD 流水线（#5-7、#12）

主路径协议改为 manifest-first：

1. prefill 在计算前通过 bootstrap 发布完整 chunk ranges；
2. 每个 chunk forward 完成后，立刻 fused gather 对应 KV range 并发送；
3. decode prepare 与 prefill RPC 同时启动，逐块 receive/scatter；
4. 最终 bundle 只携带 GDN recurrent/conv state、首 token 和校验 descriptor；
5. active decode 先执行，再回收 prefill future，避免 CPU future handling 挡在 decode 前面。

固定 1P1D 与多 prefill/decode-worker path 均已接入。`HYDRASERVE_CHUNKED_TRANSFER=0` 保留整体
bundle 回退。独立 `BootstrapServer` 只承载 JSON metadata，KV/GDN payload 继续走 SHM/CUDA
data plane；无法绑定 loopback socket 的受限 sandbox 使用 SHM manifest fallback。

新增 `hydraserve/kernels/staging.py` 的 Triton gather/scatter 将 layer × K/V × token 三维工作网格
合为一次 launch。完整 CUDA kernel suite 为 **43 collected，42 passed，1 skipped**。

真实 1P1D A/B：Qwen3.5-4B BF16，synthetic 1024 tokens，chunk 256，1 warmup + 2 measured，
max-new-tokens=2，Graph off，FlashAttention off：

| chunk transfer | output tok/s | TTFT P50 | TPOT P50 | latency P50 | failed |
|---|---:|---:|---:|---:|---:|
| off | 2.107 | 815.77 ms | 129.78 ms | 945.54 ms | 0 |
| on | **3.813** | **499.97 ms** | **21.60 ms** | **521.57 ms** | 0 |

TTFT P50 改善 38.7%。只有两条 measured 请求，TPOT 差异包含 worker/allocator 抖动，不把它当成
稳定收益声明。

### 19.3 调度、通信与缓存（#8-10）

- #8：`pyzmq` 加入 serve extra；新增 ROUTER/DEALER broker、wave-ready/load state，以及
  `scripts/dp_zmq_proxy.py`。原 `dp_proxy.py` 默认无 `--backends` 时进入 ZMQ；传入 `--backends`
  仍可运行 legacy HTTP，便于逐步迁移客户端。
- #9：production serving loop 新增统一 step token budget。active decode 每序列占一个 token，
  prefill admission 消耗剩余预算，decode width 也不超过同一上限。serve/benchmark CLI 均提供
  `--max-step-tokens`。
- #10：新增 CPU bounded-LRU HiCache L2。首次 full/stream transfer 后 decode offload KV；相同
  model+token prefix 再次 admission 命中时，控制面让 prefill 只传 GDN state，decode 直接恢复
  host KV。CLI 通过 `--host-prefix-cache-gb` 分配容量，默认 0 保持历史内存占用。

### 19.4 架构与分布式扩展（#11-14）

- #11：StateType 覆盖 full attention、SWA、DSA、MLA、linear SSM/conv；旧 RegionType 是兼容
  alias；StateHandlerRegistry 允许新架构注册安装/编码 handler。
- #13：RegionDescriptor wire format 加入 `src_tp_rank`、`dst_tp_rank`、`tp_world_size`；旧 descriptor
  缺字段时按 rank0/world1 读取。`compute_head_slice_params` 覆盖普通均分与 KV replication。
- #14：新增 `synchronize_dp_token_count`（distributed MAX all-reduce）、`pad_dp_batch` 和 valid mask；
  production serving loop 在 `--dp-graph-sync` 下要求 backend `decode_padded`，从而禁止各 rank
  悄悄用不同 Graph shape。

当前没有多 rank NCCL job，TP/DP 项完成单元测试和协议 round-trip，仍需目标集群验收。

### 19.5 最终回归与单卡稳态

- 全仓：**299 collected**，通过（硬件/可选依赖项按 marker skip）。
- CUDA/Triton：**43 passed，1 skipped**；FlashAttention 定向测试 **3 passed**。
- 真实 1P1D smoke：1 request succeeded，0 failed，证明 bootstrap + chunk receive/install + N-1
  replay 无死锁。
- 单卡第二次稳态：C1 **71.529 tok/s**、TTFT P50 24.70 ms、TPOT P50 12.91 ms；C4
  **176.533 tok/s**、TTFT P50 68.14 ms、TPOT P50 18.87 ms。相对 P3 final 的 71.373/
  178.731 tok/s，分别为 +0.22%/-1.23%，视为噪声，说明默认关闭的可选架构能力没有改变单卡
  hot path。首次复测出现 capture 冷态离群点，已保留 `all14_final.json`，不作为稳态结果。

新增结果：

- `2026-08-22_gsm8k_4b_pd_chunked_bootstrap_smoke.json`
- `2026-08-22_synthetic1024_4b_pd_chunked_off.json`
- `2026-08-22_synthetic1024_4b_pd_chunked_on.json`
- `2026-08-22_gsm8k_4b_c1_all14_final.json`
- `2026-08-22_gsm8k_4b_c1_all14_final_run2.json`
- `2026-08-22_gsm8k_4b_c4_all14_final.json`
- `2026-08-22_gsm8k_4b_c4_all14_final_run2.json`

## 20. FlashAttention 安装、兼容修复与真实 A/B

### 20.1 安装过程与问题

目标环境为 Python 3.12、Torch 2.9.1+cu128、CUDA 12.8、RTX 3090。PyPI 没有匹配该组合的
预编译 wheel，因此 `flash-attn 2.8.3.post1` 必须从源码构建。首次构建虽然设置了
`FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4`，但直接调用 deepseek 环境中的绝对 `pip` 路径不会把
该环境的 `bin` 加入 `PATH`；PyTorch 因而找不到已安装的 ninja，静默退化成单进程编译。

排查依据：构建树中没有 ninja，只有一个 nvcc/cicc；`is_ninja_available()` 返回 false。停止该次
构建、显式 `activate deepseek` 后，`is_ninja_available()` 返回 true，进程树显示
`ninja -v -j 4`。SM80 的主 attention 与 split-KV 共约 72 个 CUDA 对象，四路构建期间仍有
40 GiB 左右可用内存，最终成功生成并安装 57 MB wheel。

安装命令：

```bash
source /home/xdu/anaconda3/bin/activate deepseek
FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=4 \
  pip install 'flash-attn==2.8.3.post1' --no-build-isolation
```

ABI 验证：`flash-attn 2.8.3.post1` 可在 Torch 2.9.1+cu128 中导入，GPU 为 RTX 3090。

### 20.2 真实 kernel 与端到端兼容修复

新增真实 `paged_flash_prefill` 测试使用 256-token page、乱序物理 block table 和 GQA 4Q/2KV
heads，将 FlashAttention 输出与 FP32 `causal_gqa_attention` oracle 对照；连同 varlen smoke、
零复制传参测试全部通过。

第一次端到端基准在 warmup 报错：`block_t must be a power of two and a multiple of block_size`。
原因不是 FlashAttention，而是其 paged KV 要求 page size 256，HydraServe decode split-K 默认
`block_t=64`。修复为默认 `block_t=max(64, block_size)`，显式传入非法 tile 仍报错；新增
256-page decode 与 reference 对照测试。最终定向 GPU 回归为 **4 passed**。

安装后全仓 GPU 回归还暴露出 tiny CPU oracle 为 FP32、GPU runtime 按设计返回 BF16，而测试要求
dtype 完全相同；断言改为 `check_dtype=False`，原有 2e-2 数值容差保持不变。最终全仓
**299 tests collected** 并通过。

### 20.3 Qwen3.5-4B 最终 A/B

统一口径：单卡 GPU 0、BF16、Graph off、4 条 1024-token synthetic prompt、chunk 256、page 256、
2 条 warmup、每条 16 output tokens、C1。第一次 Flash-off 短测受首次 Triton 256-tile 编译污染，
原始文件保留但不用于结论；先热 cache 后完成最终 A/B。

| 模式 | output tok/s | TTFT P50 | TPOT P50 | latency P50 | failed |
|---|---:|---:|---:|---:|---:|
| FlashAttention off | 37.533 | 212.03 ms | 14.065 ms | 422.73 ms | 0 |
| FlashAttention on | **38.416** | **201.52 ms** | 14.080 ms | **413.21 ms** | 0 |
| 变化 | **+2.35%** | **-4.96%** | +0.11% | **-2.25%** | — |

结论：该 workload 的收益集中在长 prompt prefill/TTFT；decode TPOT 持平。4 条 measured 仍是小样本，
应在生产 prompt 长度分布和并发下复测，不能把 2.35% 外推为所有数据集的固定收益。

新增结果：

- `2026-08-22_synthetic1024_4b_flash_ab_off.json`（首次 256-tile 编译污染，排除）
- `2026-08-22_synthetic1024_4b_flash_ab_off_run2.json`（2×2-token 热态复核）
- `2026-08-22_synthetic1024_4b_flash_ab_on.json`（2×2-token 热态复核）
- `2026-08-22_synthetic1024_4b_flash_ab_off_final.json`（最终口径）
- `2026-08-22_synthetic1024_4b_flash_ab_on_final.json`（最终口径）

## 21. PD 传输与 Host prefix 深化优化

### 21.1 起点与根因

进一步审计第三节 PD 路径后，确认原有“14 项完成”仍存在四个生产热路径问题：

1. `SharedMemoryTransferBackend` 每个 KV chunk 都 create/unlink 一个 POSIX SHM object；
2. chunk callback 在 prefill 线程同步 gather、D2H 和 send，且每请求创建 executor；
3. QUANTIZED 模式只有 CPU INT4 codec，生产 worker 固定 FULL SHM，无法使用 GPU 侧传输压缩；
4. `HostPrefixCache` 实际是 `OrderedDict[(namespace, full_token_tuple)]` 精确命中，put/get 都
   `deepcopy`，stream 接收结束后还把完整 KV 从 decode GPU 再读回 CPU。

GPU L1 `PrefixCache` 原本已经是 block radix tree，本轮没有替换它。修改对象是 PD decode 侧的
Host L2。

### 21.2 数据面与 overlap 修改

- 新增 `SharedMemoryRingTransferBackend`：固定数量、固定大小的 SHM slot 在 worker 生命周期内复用；
  header-last 发布，READY/FREE 状态提供有界背压。配置值是 slot 下限，生产 factory 会根据模型
  recurrent+conv bundle 自动放大，避免 27B 状态超过默认 64 MiB 时启动后才失败。
- 第一版 ring 在整次 send/receive 上调用 `fcntl.flock`。1P+1D 真实 A/B 为负，因此没有停在该
  版本；但后续四卡验证证明“namespace 是单生产者”的假设不成立：2P+2D 中两个 P 进程可以向
  同一个 D namespace 并发写入。最终实现只在 `FREE -> WRITING` 的单字节槽位认领期间持有
  per-slot 文件锁，payload copy、READY 发布和 receive 均不持锁；既保证 MPSC 正确性，也不把
  大块内存复制串行到临界区内。
- prefill worker 持久复用单线程 transfer executor 和 dedicated CUDA stream；chunk 完成后记录
  CUDA event，传输 stream 等待该 event，最多保留可配置数量的 in-flight chunk。
- decode worker 使用 dedicated install stream；后台接收线程等待 install completion event，
  active decode 的 default stream 不再被整段 receive/install 阻塞。后台 prepare 改为有界 executor，
  不再为每个请求创建 daemon thread。
- `RuntimeStateCodec` 加入按 shape/device/dtype 复用的 pinned staging；D2H 只等待该 copy event，
  不再 `current_stream.synchronize()` 清空整条 stream。
- adaptive chunk 同时考虑 page size、模型 KV bytes/token、目标 payload、FULL/INT4/INT8 wire mode；
  INT8 的 chunk 不再沿用 BF16 字节模型。

### 21.3 GPU INT8 wire transfer

新增 `TransferMode.INT8_TRANSFER` 和 `Int8Tensor`：

- source GPU 对 fused-gather 后的连续 KV 做 group-64 symmetric INT8 quantization；
- D2H 只复制 INT8 values 和 FP32 scales；
- SHM typed codec 可直接编码/解码 INT4/INT8 structured payload；
- destination GPU 先搬压缩数据，再在 GPU 上反量化并 fused-scatter 到 paged KV；
- recurrent/conv state仍保持 FP32，不做有损压缩。

CLI：`--pd-transfer-quant int8`。它与 `--kv-quant int8` 含义不同：前者只控制 P→D wire payload，
后者控制常驻 paged KV cache。短 prompt 的量化成本可能大于拷贝收益，因此 wire INT8 不默认开启。

### 21.4 Host L2 从 exact LRU 到 block radix

最终 `HostPrefixCache` 的逻辑为：

1. token 按物理 KV block/page 分段；不足一页的尾部不作为共享前缀发布；
2. 每个 radix 节点记录能提供该节点 KV 的 backing entries；
3. 查询沿 block path 前进，即使缓存叶子是 `[A+B]`、查询是 `[A+C]`，也能返回 A；
4. admission 传递的不再是 `host_cache_hit: bool`，而是 `host_prefix_tokens: int`；
5. decode 先安装 L2 prefix，prefill manifest 从该 token offset 开始，数据面只发送 suffix；
6. ndarray 命中返回 backing payload 的 view，不做命中 deepcopy；首次接收直接用 CPU chunk 建新
   L2 entry，不再执行完整 decode GPU→CPU readback。

真实双卡测试用两个共享前 256 tokens、后 256 tokens 分叉的 prompt，第二个请求准确报告
`matched_tokens=256` 并完成 suffix-only transfer/generation。

### 21.5 真实 A/B 与负收益记录

环境：2×RTX 3090、Qwen3.5-4B BF16、FlashAttention on、block/page=256、C1、2 output tokens。
1K 为 2 warmup + 5 measured；8K 为 1 warmup + 3 measured。

| prompt / 数据面 | output tok/s | TTFT P50 | TPOT P50 | latency P50 | failed |
|---|---:|---:|---:|---:|---:|
| 1K one-shot SHM BF16 | 5.662 | 330.75 ms | 19.95 ms | 353.32 ms | 0 |
| 1K ring + per-op flock（否决） | 5.585 | 333.70 ms | 19.88 ms | 359.30 ms | 0 |
| 1K lock-free SPSC ring BF16 | **6.355** | **287.25 ms** | **19.26 ms** | **307.24 ms** | 0 |
| 1K lock-free ring INT8 | 6.304 | 289.65 ms | 20.45 ms | 312.76 ms | 0 |
| 8K ring BF16 | 0.899 | 1805.73 ms | 25.84 ms | 1831.56 ms | 0 |
| 8K ring INT8（沿用 BF16 chunk 模型） | 1.083 | 1790.62 ms | 22.48 ms | 1813.11 ms | 0 |
| 8K ring INT8 + mode-aware adaptive chunk | **1.195** | **1608.47 ms** | 24.21 ms | **1632.68 ms** | 0 |

结论：

- per-op `flock` 版本比 one-shot SHM 吞吐低 1.36%，已修正；
- 当时测得的 1P+1D lock-free SPSC ring 在 1K 相对 one-shot SHM 吞吐 +12.2%、TTFT -13.2%；
  该数字不能直接外推到修复后的 2P+2D MPSC 路径，需重新 A/B；
- INT8 在 1K 无净收益，保持 opt-in；
- 8K INT8 + adaptive chunk 相对无损 ring 吞吐 +32.9%、TTFT -10.9%；
- 样本仍小，不能外推成生产 SLA，下一轮应补 C4 与真实长文本分布。

结果文件：

- `2026-08-22_pd_shm_full_c1_run2.json`
- `2026-08-22_pd_ring_full_c1_run2.json`（per-op flock，否决）
- `2026-08-22_pd_ring_lockfree_c1.json`
- `2026-08-22_pd_ring_int8_c1.json`
- `2026-08-22_pd_ring_full_8k_c1.json`
- `2026-08-22_pd_ring_int8_8k_c1.json`
- `2026-08-22_pd_ring_int8_8k_adaptive_c1.json`

### 21.6 回归

- CPU/协议全仓：**310 tests collected**，全部通过或按硬件/可选依赖 marker skip；
- 新增 SHM ring slot reuse、INT8 typed codec、INT8 size/error、adaptive chunk、Host radix 分叉、
  suffix-only end-to-end 等测试；
- 真实 1P1D Host radix test：通过；
- 真实 A/B 所有 measured 请求：0 failed。

边界：当前机器无 GPU peer access，CUDA P2P/NCCL/NIXL 仍不能做目标拓扑验收；生产 SHM 主路径
采用 all-layer bulk chunk，而不是把 bulk chunk 拆成逐层小消息。`LayerTransferPipeline` 继续作为
高带宽 transport 接口保留。

## 22. V3 四卡压测前置修复（未执行测试）

拉取 `dbc570b` 后审计新增的 P0 harness 和 engine-only 4×DP backend，发现直接开跑会污染结论：

1. trace 虽记录 `ignore_eos`，但 generation loop 仍按全局 EOS 提前结束；
2. 4×DP worker 选择只观察瞬时 RPC pending，reserve 完成后归零，burst 会反复选择 GPU 0；
3. coordinator 按 worker 串行调用 decode，四张卡没有在同一步并行执行；
4. 多 worker decode 的 local group/global position 混用，交错 request 顺序可能映射错误；
5. trace CLI 无法表达 W1 的 long=16、short=128 以及定时注入，warmup 还会吃掉正式 trace；
6. 原 trace-wide hash 只拼接 prompt 文本，没有覆盖输出长度、到达时刻和 EOS 策略。

本轮完成的前置实现：

- `ServingRequest` 增加逐请求 `ignore_eos`，进程内 runner 和 HTTP API 都传递该字段；
- 4×DP 从 worker 选择到 request release 持久计数 active assignment，等负载 round-robin；
- coordinator 使用持久线程池同时派发各 worker decode，并修正 request/result 映射；
- trace-out 支持分类输出长度、long 固定注入时刻和 short Poisson 到达率；重放按绝对到达时刻排序；
- trace 固化实际 re-encode token 数、token-ID/record/全文件 SHA256，并写同名 `.meta.json`；
- CLI 为 trace 生成独立 synthetic warmup，正式结果不再丢掉前 N 条请求；
- benchmark metadata 增加 trace hash、模型文件 manifest、逐卡属性、`nvidia-smi` 状态和 topology；
- `BENCHMARK_PLAN_V3.md` 已改为 engine-only D0/P0 主路径，并补充 W1 命令和十项四卡门禁。

按用户要求，本节修改没有执行单元测试、GPU 测试或性能压测。新增测试用例只作为下一次验证入口；
当前状态是“代码和实验前置已准备，尚未验证”，禁止据此产生或更新性能数字。

## 23. Conditional PD、RPC 解耦与 P 卡 work-conserving（2026-08-23）

在拉取 `d13bfe1` 后完成五项组合优化；本轮只做实现和单元验证，未运行四卡性能压测：

1. `PDClusterConfig`/CLI 增加 `prefill_short_policy=never|work-conserving`，固定 1P+3D
   基线可明确禁止 P 卡服务 short；
2. 增加 `conditional_pd_tokens` 确定性路由，阈值以下 short 在 D collocated，阈值以上
   long 走 P→D；P 不健康时 fail closed 到 D；
3. V3 W1 固化 32-token 与 128-token 两份输出负载，分别观察 prefill-interference 和
   decode-heavy 边界；
4. prefill RPC 增加关联 ID 和并发 waiter，不再持 worker-wide lock 等待 long response；
5. `PrefillWorker.process` 在 runtime chunk callback 暴露一致性边界，P worker 每个边界有界
   执行 short reserve/prepare/decode/release；新增 P short 复用与 chunk 抢占 Prometheus 计数器。

正确性约束：只在完整 chunk 完成、KV page table 与 recurrent/conv state 一致后让出；单次最多
处理 8 个短操作，防止连续 short 流量饿死 long。新增单测覆盖确定性路由、固定 P 开关、乱序
RPC 响应分发与 chunk yield 计数；相关 CPU/模型小配置测试通过，真实 4×3090 数据待 V3 执行。

## 24. W1 short-32 口径修复与 P/D prefill 执行槽解耦（2026-08-23）

四卡W1试跑暴露出两个会直接污染结论的问题：short-32的TPOT把最后token之后的同步KV release、
worker RPC和清理时间一起除以31；1P+3D conditional虽然把short路由到D collocated，上层
`ContinuousGenerationLoop`却仍按P卡数量创建prefill线程池，导致1P拓扑只有一个host执行槽，
long-PD可以把short-D prefill一起串行化。

本轮修改：

- benchmark逐请求记录`last_token_at`，TPOT只覆盖首末token；新增`itl_ms`、
  `release_tail_ms`、`client_queue_ms`和`e2e_ttft_ms`；
- engine-only token event携带实际decode batch size，使“batch稀疏”可以由数据验证，不再作为
  未测量的默认解释；
- SLO的TTFT门禁改用包含客户端排队的`e2e_ttft_ms`；历史engine TTFT继续保留用于归因；
- 增加`route_counts_all`、`route_failure_counts`与`throughput_valid`。出现OOM/超时的run仍保存
  原始数据，但吞吐不得进入headline比较；
- serving loop支持backend声明多个独立prefill executor；multi-worker backend将P-prefill和
  D-collocated划为独立池，并分别按P/D设备数设置并行度；
- token budget遇到暂时放不下的long时只defer该请求，继续扫描后续可执行short，消除host侧
  head-of-line blocking，同时保留admission aging；
- 新增同步release隔离、客户端排队、decode batch审计、P/D executor并发和conditional分组测试。

相关CPU回归通过；本轮没有运行四卡性能压测。旧W1 short-32表中的TPOT/SLO仍采用历史污染口径，
只能作为问题定位材料，修复后结果必须使用新JSON字段并重新生成，禁止覆盖或混用旧数值。

## 25. V3 调度生命周期与审计口径补强（2026-08-23，未跑GPU压测）

在第24节CPU回归之后继续做纯代码审计，发现此前的“独立P/D executor”仍不足以保证物理资源
有界，另外有几项观测字段会误导short-32归因：

1. 请求完成时，generation线程同步等待worker `release()`；一个慢KV释放/RPC会暂停所有请求的
   decode，短输出请求频繁结束时尤其容易把其他请求ITL抬高；
2. `ThreadPoolExecutor`队列无界，long在排队取得P执行线程前已经admit并reserve D KV；1P/2P
   可以让大量不能开始prefill的long提前吃光D缓存；
3. P-worker collocated short在serving loop和backend `prefill()`各admit一次，会重复reserve和
   增加统计；并发线程的P选择与pending加一分离，可能同时选择同一张P卡；
4. engine事件记录的是coordinator总decode batch，而多GPU真正执行的是各worker子batch；P与D
   的整数worker ID也会在`per_worker`中碰撞；
5. HTTP流最后的tokenizer flush可能产生一个有文本但没有新模型token的SSE chunk，旧runner会
   把它计入completion tokens；V3 gate脚本的P0 hash检查还把四卡切成了实际1P+1D。

本轮实现：

- 引入有界异步release executor；active调度立即继续，但terminal event仍等待release完成，因此
  TPOT不含清理、`release_tail_ms`仍可观察清理成本，吞吐wall口径也没有偷偷删除release；
- release RPC完成前保留worker binding和host侧KV reservation，防止active槽提前释放后容量统计
  虚增；
- backend提供admission前的确定性executor group hint，serving loop按P/D物理worker数限制
  outstanding prefill；没有P槽的long留在admission队列，不提前reserve D KV，同时继续扫描short；
- token budget只在请求实际admit并提交prefill后扣减；因容量或物理槽defer的long不再虚耗本轮
  budget并阻塞队列后方short；
- P-short admission幂等；P dispatch增加原子claim，选择时使用`pending + claims`避免并发herding；
- Prometheus新增prefill物理槽defer、release pending/complete/failure计数，便于确认优化是在消除
  排队与主循环阻塞，而不是把开销隐藏到不可见线程；
- `decode_batch_sizes`改为请求所在物理P/D worker的子batch；请求新增`worker_pool`，结果按
  `prefill:N`/`decode:N`/`collocated:N`聚合；
- HydraServe SSE token chunk增加`hydraserve_token_id`扩展，HTTP runner据此跳过decoder flush；
- V3 gate的adaptive拓扑按完整四卡均分为2P+2D，并让`--datasets`参数真正传给子命令。

容量边界也已固化进计划：block=256、每D缓存131072 tokens时共有512 blocks；一条
32K prompt + 16 output long需要129 blocks，单卡4条long的516 blocks在没有任何short时也必然
超容量。新限流避免“尚未开始P prefill就占D KV”的人为OOM，但不会掩盖真实并发容量上限。

验证：定向serving/multi-worker测试52项通过；随后全仓CPU测试全部通过（硬件/可选依赖项按marker
skip）。本轮未执行GPU或四卡性能实验，不能据此更新任何吞吐、TTFT、TPOT或SLO数字。

## 26. 异步 prefill token-budget 语义修复（2026-08-23）

四卡W1-128 D0外部复核暴露出`max_step_tokens`的非连续反转：8192时64/72成功、8条long失败，
short TPOT P50/P99为151/160ms；65536时72/72成功，但short TPOT升至262/397ms。该结果不能
解释为“8192更适合short”：前者没有完成long工作量，吞吐和SLO均不能进入主比较；后者让long
按时参与后暴露了D0 collocated prefill对short decode的真实干扰。

代码根因有两层：

1. async serving loop用完整prompt长度作为一次admission demand，并从`max_step_tokens`减去active
   decode数；32K long是否能提交由一个本不应承担路由职责的全局阈值决定；
2. 单executor-pool backend没有pre-admission group hint时不做物理槽限流，D0可以在只有4个GPU
   prefill槽时提前admit/reserve更多请求，放大KV与激活内存压力。

修复后，异步prefill使用独立admission budget：每个请求只计一个runtime chunk，且cost不超过
`max_step_tokens`；active decode不再从该异步预算扣除。所有只有一个executor group的backend
即使没有route hint，也默认按其物理prefill并行度限制outstanding future。完整prefill RPC仍占用
该物理槽直到返回，因此不会因按chunk计费形成无界host队列。

这项修改消除8192/65536决定“long是否有资格运行”的错误语义，但不会刻意让D0的long prefill
不干扰short：D0 worker仍是collocated执行，long成功后short TPOT恶化属于W1要测的基线现象；
真正的隔离收益应由P0/P2-C在相同完整工作量下证明。新增单测覆盖active decode存在时long仍可
提交，以及无route-hint的单物理池在第一个prefill阻塞时不会提前reserve第二个请求。

## 27. 四卡 W1 复核：D0 收敛与 2P+2D SHM ring 并发修复（2026-08-23）

外部 4×RTX 3090 复核了第26节修复后的 W1-128（seed42、C32）。D0 在
`max_step_tokens=8192/65536` 下都完成 72/72，请求吞吐均为70.4 tok/s，short TPOT
P50/P99分别为264/396ms和268/405ms，差异小于2%。这证明8192不再错误拒绝32K long，后续固定
8192；两组short SLO均为0/64，则是完整long工作量进入后测得的collocated interference，不再是
“long OOM后short看起来更快”的失真结果。

同一轮中，P0（2P+2D、全PD）在72请求负载下运行5分41秒仍无JSON，四卡利用率均为0%；但3请求
P0 smoke通过，P2-C（1P+3D）也能结束（70/72，2条short OOM）。该组合不支持“prefill token
budget死锁”的初步猜测，而指向只在双P并发时出现的数据面问题。

根因是`SharedMemoryRingTransferBackend`错误地把每个D namespace当成SPSC。2P+2D下两个独立P
进程可能同时观察到同一FREE slot，随后都写WRITING并覆盖header/digest。D端等待请求A时看到请求B
的digest而继续等待，请求B的prepare又可能排在同一D RPC串行路径之后，最终形成GPU全部空闲的
协议死锁。1P+3D没有同namespace双生产者，3请求smoke则可能因时序未碰撞而通过。

修复为每个slot仅在`FREE -> WRITING`认领期间使用跨进程`flock`；认领后立即释放锁，大payload
copy和READY发布仍在锁外。新增真实双producer进程、单consumer、16轮槽位复用回归，验证每轮两份
不同digest/payload都能准确接收，且两个子进程正常退出。

本地没有复跑四卡GPU性能，因此P0原挂死run只能标为无效正确性结果，不能解释成性能负收益；
P2-C的70/72与44.4 tok/s同样不得进入headline比较。下一步只先复跑P0 W1-128 seed42：必须达到
72/72、无transfer timeout/长时间0% GPU空转，随后才恢复多seed性能矩阵并重新测量MPSC认领锁的
实际开销。
