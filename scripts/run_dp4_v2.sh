#!/bin/bash
# DP4 re-run on current HEAD (8949816 changed DP admission priority + load balance).
set -u
PY=${PY:-/root/autodl-tmp/hydraserve-venv/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/Qwen3.5-4B}
DATASETS=${DATASETS:-/root/autodl-tmp/data}
cd /root/autodl-tmp/Hydraserve
export OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRASERVE_STALL_DUMP_SECONDS=30
run() {
  name=$1; trace=$2; seed=$3
  echo "=== RUN $name start $(date +%T) ==="
  "$PY" -u -m hydraserve benchmark "$MODEL" "$DATASETS" \
    --dataset synthetic --trace "traces/v5/$trace" --dp-devices 0 1 2 3 \
    --concurrency 16 --warmup 8 --kv-quant int8 --cache-tokens 65536 --block-size 256 \
    --prefix-cache-blocks 0 --prefill-chunk-size 16384 --max-step-tokens 8192 \
    --worker-log-dir "worker_logs/v5/dp4_v2_$name" \
    --output "benchmark_output/v5/dp4_v2_$name.json" --seed "$seed" \
    > "benchmark_output/v5/dp4_v2_$name.stdout.log" 2>&1
  echo "=== RUN $name done exit=$? $(date +%T) ==="
}
failed=0
run m1_s42 m1_seed42.jsonl 42 || failed=1
run m1_s43 m1_seed43.jsonl 43 || failed=1
run m1_s44 m1_seed44.jsonl 44 || failed=1
run b1_s42 b1_seed42.jsonl 42 || failed=1
echo "=== ALL DP4-v2 RUNS FINISHED (failed=$failed) ==="
exit "$failed"
