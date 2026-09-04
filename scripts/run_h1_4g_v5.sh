#!/bin/bash
# V5 official H1-4GPU matrix (1P+3D), pending DP4 comparison.
set -u
PY=${PY:-/root/autodl-tmp/hydraserve-venv/bin/python}
REPO_DIR=${REPO_DIR:-/root/autodl-tmp/Hydraserve}
MODEL=${MODEL:-/root/autodl-tmp/Qwen3.5-4B}
DATASETS=${DATASETS:-/root/autodl-tmp/data}
cd "$REPO_DIR"
export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRASERVE_STALL_DUMP_SECONDS=30
mkdir -p benchmark_output/v5 worker_logs/v5
run() {
  name=$1; trace=$2; seed=$3
  echo "=== RUN $name start $(date +%T) ==="
  "$PY" -u -m hydraserve benchmark "$MODEL" "$DATASETS" \
    --dataset synthetic --adaptive --trace "traces/v5/$trace" \
    --prefill-devices 0 --decode-devices 1 2 3 \
    --conditional-pd-tokens 6144 --hybrid-long-overflow-ms 5000 \
    --pd-prefill-token-budget 32768 --hybrid-short-max-assigned-work 8192 \
    --hybrid-long-pressure-hold-ms 1000 \
    --prefill-short-policy work-conserving --pd-schedule load-aware \
    --concurrency 16 --warmup 8 --kv-quant int8 --cache-tokens 65536 --block-size 256 \
    --prefix-cache-blocks 0 --prefill-chunk-size 16384 --max-step-tokens 8192 \
    --pd-transfer-backend shm-ring --pd-transfer-quant int8 \
    --worker-log-dir "worker_logs/v5/h1_4g_$name" \
    --output "benchmark_output/v5/h1_4g_$name.json" --seed "$seed" \
    > "benchmark_output/v5/h1_4g_$name.stdout.log" 2>&1
  rc=$?
  echo "=== RUN $name done exit=$rc $(date +%T) ==="
  return $rc
}
failed=0
run m1_s42 m1_seed42.jsonl 42 || failed=1
run m1_s43 m1_seed43.jsonl 43 || failed=1
run m1_s44 m1_seed44.jsonl 44 || failed=1
run b1_s42 b1_seed42.jsonl 42 || failed=1
echo "=== ALL H1-4GPU RUNS FINISHED ==="
exit "$failed"
