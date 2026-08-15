#!/usr/bin/env bash
# 4×RTX 3090 压测矩阵：1P+3D vs 4×单卡 collocated。
#
# 用法（先跑完 00_smoke_1p3d.sh）：
#   MODEL=/path/to/Qwen3.5-4B DATASETS=/path/to/benchmark \
#   bash scripts/cloud-4gpu/01_run_matrix.sh
#
# 两个 arm 的公平性设计：
#   - DP 基线 = 4 个独立 collocated 进程，各以 λ/4 的 Poisson 到达（4 个独立
#     Poisson 之和仍是 Poisson λ，统计上等价于一个总负载均衡器），每进程样本
#     数 = 总样本数/4；
#   - PD arm = 单个 1P+3D 进程（--adaptive --decode-devices），到达 λ，同总样本数；
#   - 采样参数、warmup、cache-tokens、kv-headroom、max tokens 两端一致；
#   - 输出 JSON 落到 benchmark_output/cloud-4gpu/（已在 .gitignore 中）。
set -euo pipefail

MODEL="${MODEL:?设置 MODEL=/path/to/Qwen3.5-4B}"
DATASETS="${DATASETS:?设置 DATASETS=/path/to/benchmark}"
OUT="${OUT:-benchmark_output/cloud-4gpu}"
mkdir -p "$OUT"

# --- ShareGPT 高并发 Poisson 扫描 -------------------------------------------
RATES="${RATES:-8 16 32 64}"          # 总到达率 req/s
SHAREGPT_LIMIT="${SHAREGPT_LIMIT:-256}"  # 每个 arm 的实测请求总数
SHAREGPT_MAX_PROMPT="${SHAREGPT_MAX_PROMPT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CONCURRENCY="${CONCURRENCY:-64}"      # 客户端并发上限 = 每 worker 准入上限
CACHE_TOKENS="${CACHE_TOKENS:-65536}"
HEADROOM="${HEADROOM:-128}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"    # 单次 benchmark 超时（秒）
# 默认禁用 FlashAttention（云端免装 flash-attn 可直接跑）；装好后设 FA_OPTS=""
FA_OPTS="${FA_OPTS:---no-flash-attention}"

BENCH_OPTS="--max-new-tokens $MAX_NEW_TOKENS --concurrency $CONCURRENCY \
  --cache-tokens $CACHE_TOKENS --kv-headroom-blocks $HEADROOM \
  --warmup 8 --arrival-pattern poisson $FA_OPTS"

echo "############ ShareGPT：4× collocated 基线（每 GPU λ/4） ############"
for rate in $RATES; do
  echo "== DP arm, λ=$rate/s =="
  pids=()
  per_gpu=$(awk "BEGIN{printf \"%g\", $rate/4}")
  for gpu in 0 1 2 3; do
    timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
      --dataset sharegpt --limit $((SHAREGPT_LIMIT / 4)) \
      --max-prompt-tokens "$SHAREGPT_MAX_PROMPT" \
      $BENCH_OPTS --request-rate "$per_gpu" --seed "$gpu" \
      --device "cuda:$gpu" \
      --output "$OUT/dp_sharegpt_r${rate}_gpu${gpu}.json" \
      >"$OUT/dp_sharegpt_r${rate}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
done

echo "############ ShareGPT：1P+3D（单进程 4 卡） ############"
for rate in $RATES; do
  echo "== PD arm, λ=$rate/s =="
  timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
    --dataset sharegpt --limit "$SHAREGPT_LIMIT" \
    --max-prompt-tokens "$SHAREGPT_MAX_PROMPT" \
    $BENCH_OPTS --request-rate "$rate" --seed 0 \
    --adaptive --device cuda:0 --decode-devices cuda:1 cuda:2 cuda:3 \
    --output "$OUT/pd_sharegpt_r${rate}.json" \
    >"$OUT/pd_sharegpt_r${rate}.log" 2>&1
done

# --- 峰值吞吐参考点（burst，无到达率限制） -----------------------------------
echo "############ ShareGPT burst：峰值吞吐参考点 ############"
for gpu in 0 1 2 3; do
  timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
    --dataset sharegpt --limit $((SHAREGPT_LIMIT / 4)) \
    --max-prompt-tokens "$SHAREGPT_MAX_PROMPT" \
    $BENCH_OPTS --arrival-pattern burst --seed "$gpu" \
    --device "cuda:$gpu" \
    --output "$OUT/dp_sharegpt_burst_gpu${gpu}.json" \
    >"$OUT/dp_sharegpt_burst_gpu${gpu}.log" 2>&1 &
done
wait
timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
  --dataset sharegpt --limit "$SHAREGPT_LIMIT" \
  --max-prompt-tokens "$SHAREGPT_MAX_PROMPT" \
  $BENCH_OPTS --arrival-pattern burst --seed 0 \
  --adaptive --device cuda:0 --decode-devices cuda:1 cuda:2 cuda:3 \
  --output "$OUT/pd_sharegpt_burst.json" >"$OUT/pd_sharegpt_burst.log" 2>&1

# --- LongBench 长上下文低并发 ------------------------------------------------
# 注意：LongBench 单样本运行时间长、子集样本少（gov_report 约 20 条），
# 分位数本身噪声大；这里测的是 TTFT/TPOT 量级与路由行为，不是精确分位数。
# 公平性：客户端总负载两边一致——c=1 时 DP 基线只开 1 个 collocated 进程
# （其余 3 卡空闲，这正是 PD 在低负载下的结构性代价）；c=4 时开 4 个进程、
# 每进程 1 并发。
LONG_SUBSETS="${LONG_SUBSETS:-gov_report}"
LONG_LIMIT="${LONG_LIMIT:-6}"          # 每 arm 实测样本（不含 warmup）
LONG_MAX_PROMPT="${LONG_MAX_PROMPT:-30000}"
LONG_MAX_NEW="${LONG_MAX_NEW:-32}"
LONG_CONCURRENCIES="${LONG_CONCURRENCIES:-1 4}"

echo "############ LongBench：collocated 基线（进程数 = 并发数） ############"
for subset in $LONG_SUBSETS; do
  for concurrency in $LONG_CONCURRENCIES; do
    echo "== DP arm, subset=$subset c=$concurrency =="
    pids=()
    per_proc_limit=$(((LONG_LIMIT + concurrency - 1) / concurrency))
    for proc in $(seq 1 "$concurrency"); do
      gpu=$((proc - 1))
      timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
        --dataset longbench --subset "$subset" --limit "$per_proc_limit" \
        --max-prompt-tokens "$LONG_MAX_PROMPT" --max-new-tokens "$LONG_MAX_NEW" \
        --concurrency 1 --warmup 1 --arrival-pattern burst --seed "$gpu" \
        --cache-tokens "$CACHE_TOKENS" --kv-headroom-blocks "$HEADROOM" \
        $FA_OPTS \
        --device "cuda:$gpu" \
        --output "$OUT/dp_longbench_${subset}_c${concurrency}_proc${proc}.json" \
        >"$OUT/dp_longbench_${subset}_c${concurrency}_proc${proc}.log" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}"
  done
done

echo "############ LongBench：1P+3D ############"
for subset in $LONG_SUBSETS; do
  for concurrency in $LONG_CONCURRENCIES; do
    echo "== PD arm, subset=$subset c=$concurrency =="
    timeout "$RUN_TIMEOUT" python -m hydraserve benchmark "$MODEL" "$DATASETS" \
      --dataset longbench --subset "$subset" --limit "$LONG_LIMIT" \
      --max-prompt-tokens "$LONG_MAX_PROMPT" --max-new-tokens "$LONG_MAX_NEW" \
      --concurrency "$concurrency" --warmup 1 --arrival-pattern burst --seed 0 \
      --cache-tokens "$CACHE_TOKENS" --kv-headroom-blocks "$HEADROOM" \
      $FA_OPTS \
      --adaptive --device cuda:0 --decode-devices cuda:1 cuda:2 cuda:3 \
      --output "$OUT/pd_longbench_${subset}_c${concurrency}.json" \
      >"$OUT/pd_longbench_${subset}_c${concurrency}.log" 2>&1
  done
done

# --- 汇总对比 -----------------------------------------------------------------
echo "############ 汇总对比表 ############"
for rate in $RATES burst; do
  python3 scripts/cloud-4gpu/02_merge_results.py \
    --name "sharegpt_r${rate}" \
    --dp "$OUT"/dp_sharegpt_r${rate}_gpu*.json \
    --pd "$OUT/pd_sharegpt_r${rate}.json"
done
for subset in $LONG_SUBSETS; do
  for concurrency in $LONG_CONCURRENCIES; do
    python3 scripts/cloud-4gpu/02_merge_results.py \
      --name "longbench_${subset}_c${concurrency}" \
      --dp "$OUT"/dp_longbench_${subset}_c${concurrency}_proc*.json \
      --pd "$OUT/pd_longbench_${subset}_c${concurrency}.json"
  done
done
