#!/bin/bash
# V5 official H1-4GPU matrix (1P+3D), pending DP4 comparison.
set -u
PY=/root/autodl-tmp/hydraserve-venv/bin/python
cd /root/autodl-tmp/Hydraserve
export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRASERVE_STALL_DUMP_SECONDS=30
mkdir -p benchmark_output/v5 worker_logs/v5
run() {
  name=$1; trace=$2; seed=$3
  echo "=== RUN $name start $(date +%T) ==="
  "$PY" -u -m hydraserve benchmark /root/autodl-tmp/Qwen3.5-4B /root/autodl-tmp/data \
    --dataset synthetic --adaptive --trace "traces/v5/$trace" \
    --prefill-devices 0 --decode-devices 1 2 3 \
    --conditional-pd-tokens 6144 --hybrid-long-overflow-ms 5000 \
    --prefill-short-policy work-conserving --pd-schedule load-aware \
    --concurrency 16 --warmup 8 --kv-quant int8 --cache-tokens 65536 --block-size 256 \
    --prefix-cache-blocks 0 --prefill-chunk-size 16384 --max-step-tokens 8192 \
    --pd-transfer-backend shm-ring --worker-log-dir "worker_logs/v5/h1_4g_$name" \
    --output "benchmark_output/v5/h1_4g_$name.json" --seed "$seed" \
    > "benchmark_output/v5/h1_4g_$name.stdout.log" 2>&1
  echo "=== RUN $name done exit=$? $(date +%T) ==="
}
run m1_s43 m1_seed43.jsonl 43
run m1_s44 m1_seed44.jsonl 44
run b1_s42 b1_seed42.jsonl 42
echo "=== ALL H1-4GPU RUNS FINISHED ==="
