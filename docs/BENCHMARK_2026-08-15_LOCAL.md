# 本地单卡参照实验 — 2026-08-15

目的:为云端 4 卡数据补一组**单卡参照点**,量化"prefill 干扰 decode"的
剂量-响应关系。混合负载与纯短请求负载用**完全相同的配置**各跑一遍,短请求
TPOT 的差值就是干扰剂量。

## 环境

- 硬件:本机 1×RTX 3090 24GB(cuda:0,跑时另一卡空闲);
- 环境:nanovllm conda env — torch 2.4.0+cu121、triton 3.0.0、flash-attn 2.8.3、
  tokenizers 0.21.4;代码为 `726357c` 之后的 main(含 B1/FA/FULL/grid 修复);
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
- 注意:与云端环境(torch 2.9.1 / triton 3.7.1)存在版本差,长 prefill 慢 ~20%,
  对照时只比较趋势,不比绝对值。

## 数据

- 生成脚本:`/tmp/gen_mixed_local.py`(与云端 gen_mixed.py 同配方:12 句循环
  拼接,长 130K token、短 2.1K token;纯短请求为同一批短记录);
- 混合负载:8 长 + 32 短,runner 取 `--limit 39 --warmup 1` → 实测 39;
- 纯短负载:32 短,runner 取 `--limit 31 --warmup 1` → 实测 31。

## 统一配置

```bash
python -m hydraserve benchmark <model> <data-dir> \
  --dataset sharegpt --max-prompt-tokens 131072 --max-new-tokens 128 \
  --concurrency 4 --warmup 1 --arrival-pattern burst \
  --cache-tokens 140000 --kv-headroom-blocks 128 --prefill-chunk-size 131072 \
  --device cuda:0 --output <out.json>
```

## 结果

### 单卡混合负载(8 长 + 31 短实测)

| 指标 | 值 |
|---|---|
| 成功/失败 | 39/0 |
| wall time | 970.5s |
| 吞吐 | 5.1 tok/s |
| 长请求 TTFT(min/med/max) | 79.1 / 79.8 / 80.4s(全串行) |
| 短请求 TTFT p50 | 2.18s |
| 短请求 TPOT p50 / p99 | **662 / 936ms** |

### 单卡纯短请求(31 短)

| 指标 | 值 |
|---|---|
| 成功/失败 | 31/0 |
| wall time | 55.1s |
| 吞吐 | **72.0 tok/s** |
| 短请求 TTFT p50 / p99 | 1.89 / 3.69s |
| 短请求 TPOT p50 / p99 | **40.6 / 48.1ms** |

## 解读:干扰剂量 = 16.3×(p50)/ 19.5×(p99)

| 场景 | 短请求 TPOT p50 | TPOT p99 | 吞吐 |
|---|---|---|---|
| 纯短请求(无干扰) | 40.6ms | 48.1ms | 72.0 tok/s |
| 混合负载(8 长同卡) | 662ms | 936ms | 5.1 tok/s |
| **干扰倍数** | **16.3×** | **19.5×** | **14.1×** |

与云端对照:

- 云端 4×DP(每卡 2 长)短请求 TPOT p99 822ms——介于本机纯短(48.1ms)与
  单卡全量(936ms)之间,**干扰剂量与同卡长请求数量正相关**(每卡 8 长 →
  936ms,每卡 2 长 → 822ms,0 长 → 48ms);
- 云端 1P+3D 短请求 TPOT p99 296ms——长请求不在 decode 卡上 prefill,隔离
  有效,但仍高于纯短基线(48.1ms),残余来自共享 admission、FULL 传输与
  decode 批内长请求;
- 结论:干扰剂量不是二值的("有/无"),而是与**同卡长 prefill 串行时长**
  连续相关——这支持统一池设计里"prefill-load 感知路由"的必要性:短请求应
  避开正在跑长 prefill 的卡,而不是简单区分 P/D 角色;
- 附带发现:纯短单卡 72 tok/s 是四组数据里的最高吞吐点(云端 4×DP 单卡
  5.4 tok/s 是被长请求拖低的)——短请求服务在无长请求时远未被充分利用,
  "P/D/DP 统一池"的短请求容量空间比混合数据表面看起来更大。
