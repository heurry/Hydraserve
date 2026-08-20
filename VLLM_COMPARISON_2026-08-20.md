# HydraServe vs vLLM 对比（2026-08-20）

> 同拓扑 4×DP、同负载 A（8×32K 长 + 32×2K 短，burst 16 并发，max_new_tokens 128），
> 4×RTX 3090 24GB，Qwen3.5-4B BF16 + 量化 KV。
> HydraServe 数据见 `results/LoadA_4dp_loadaware_fd_on.json`（组 1）；
> vLLM 数据见 `results/vllm/LoadA_dp4.json`。

## 一、结果对比

| 指标 | HydraServe 4×DP | vLLM DP=4 | 差异 |
|---|---|---|---|
| 吞吐 (tok/s) | 64.0 | **174.5** | +2.7× |
| TPOT p50 | 103ms | **19ms** | -82% |
| TPOT p99 | 309ms | **181ms** | -41% |
| TTFT p50 | — | **971ms** | — |
| TTFT p99 | 27.1s | **10.6s** | -61% |
| 成功率 | 40/40 | 40/40 | 持平 |

## 二、vLLM 长/短请求分布

| 请求 | tpot 中位 | tpot p99 | ttft 中位 |
|---|---|---|---|
| 长 (32K) | 58ms | 161ms | 6.8s |
| 短 (2K) | 19ms | 193ms | 0.6s |

## 三、结论

1. **vLLM 全面领先**：同拓扑、同负载下，吞吐 2.7×、TPOT P99 -41%、TTFT P99 -61%。
   这反映成熟推理引擎（FlashAttention、连续批处理、CUDA kernel 优化）vs
   HydraServe 原型的工程差距。
2. **32K 长请求的 decode 成本真实存在**：vLLM 长请求 tpot 中位 58ms（短请求 19ms），
   是 32K KV 的 scan 成本；但远低于 HydraServe（组 2 PD 长请求 ~149ms），
   说明 PD 拓扑本身不降低单 token 成本，而是隔离干扰。
3. **vLLM 配置**：`--data-parallel-size 4`（等价 4×DP）、`--no-enable-prefix-caching`
   （对齐 HydraServe 关前缀缓存）、`--kv-cache-dtype fp8`（对齐 HydraServe int8 KV）、
   `--enable-chunked-prefill --max-num-batched-tokens 32768`（对齐 HydraServe
   prefill-chunk-size 32768）、`--gpu-memory-utilization 0.7`（24GB 卡上防 OOM）。

## 四、vLLM 部署要点（本次踩坑记录）

- **版本**：vLLM 0.27.1（CUDA 13）在驱动 12.8 上无法运行（driver too old）；
  降级到 **0.19.1 + torch 2.10.0+cu128** 匹配驱动。
- **Qwen3.5-4B 是 VLM**：vLLM 无条件初始化视觉塔，依赖 flash_attn；
  纯文本部署需 patch `vllm/model_executor/models/qwen3_5.py` 跳过视觉塔，
  并从 config.json 移除 `image_token_id` 等多模态字段。
- **flash_attn ABI**：主环境 flash_attn 2.8.3 与 torch 2.10 不匹配
  （`undefined symbol: c10_cuda_check_implementation`），移除后 vLLM 走自带
  rotary 实现。
- **max-model-len**：vLLM 端对 synthetic 32K 文本 encode 长度 33665-33873
  （与 HydraServe 端 tokenizers 库 encode 32767 有 ~4.5% 库差异），需设
  max-model-len 40000 容纳。
- **显存**：`--gpu-memory-utilization 0.9` 时 8×32K 并发 prefill 激活 OOM，
  降到 0.7 稳定。
- **DP=4 模式**：`--data-parallel-size 4` 单实例，vLLM 自己路由请求到 4 卡，
  优于外部 proxy 分发（proxy 的 in-flight 均衡在快响应后端下失效导致惊群）。
