# 4×RTX 3090 云端压测

1P+3D vs 4×单卡 collocated 的对比实验脚本。**先跑 smoke,通过后再跑矩阵**——
1P+ND 的 N>1 从未在真实硬件上验证过。

## 云端机器准备

1. 4×RTX 3090(≥24GB),clone 仓库后先跑一键准备(装依赖 + 自检,不装
   flash-attn,脚本默认 `--no-flash-attention` 可免编译直接跑):

   ```bash
   bash scripts/cloud-4gpu/00_setup.sh
   ```

2. 检查 `nvidia-smi topo -m`:
   - 全 x16 且有 peer access → P2P/QUANTIZED 路径可用;
   - 无 peer access → 自动回退 SHM PARTIAL(与本地开发机一致,结果可对照);
3. 上传 Qwen3.5-4B checkpoint 与数据集目录(含 `sharegpt.json`、`longbench.zip`),
   `python -m hydraserve inspect-models <模型目录>` 与
   `python -m hydraserve inspect-datasets <数据目录>` 确认就绪;
4. 若环境已装好 flash-attn,跑矩阵时设 `FA_OPTS=""` 启用(两个 arm 必须一致)。

## 运行

```bash
# 1. 门禁 smoke：拓扑、1P+3D 启动、短/长 prompt 双路径、最小 benchmark
MODEL=/path/to/Qwen3.5-4B DATASETS=/path/to/benchmark \
  bash scripts/cloud-4gpu/00_smoke_1p3d.sh

# 2. 压测矩阵（ShareGPT Poisson 扫描 + burst 峰值 + LongBench 长上下文）
MODEL=/path/to/Qwen3.5-4B DATASETS=/path/to/benchmark \
  bash scripts/cloud-4gpu/01_run_matrix.sh
```

矩阵脚本末尾自动调用 `02_merge_results.py` 输出对比表;结果 JSON 落在
`benchmark_output/cloud-4gpu/`(已被 .gitignore 覆盖),每对实验另写
`*.merged.json` 供后续分析。

## 公平性设计

- DP 基线 = 每卡一个 collocated 进程,ShareGPT 各以 λ/4 的 Poisson 到达
  (4 个独立 Poisson 之和 = Poisson λ,等价于总负载均衡器),每进程样本数 =
  总样本数/4;PD arm = 单进程 1P+3D,到达 λ、同总样本数;
- LongBench 按客户端总负载对齐:c=1 时 DP 只开 1 个进程(其余 3 卡空闲,这是
  PD 低负载下的结构性代价),c=4 时开 4 个进程 × 1 并发;
- 采样、warmup、cache-tokens、kv-headroom、max tokens 两端一致;
- DP 汇总的 wall time 取 4 进程 max,分位数在合并样本上重算(与 runner 同插值)。

## 预期(来自 main.md §1.3,不是失败)

4 卡规模上 DP 的吞吐大概率更高——1P+3D 只有 3 张卡持有 KV,且短 prompt 场景
PD 有传输开销。值得记录的是:P99 尾延迟与 prefill/decode 干扰隔离、长上下文
TTFT、以及 1P+ND 绑定/分组/并行 RPC 协议本身的首次实卡行为。
