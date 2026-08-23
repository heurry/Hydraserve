#!/bin/bash
# One-run P0 deadlock diagnosis. No external profiler is required: when the
# process remains alive, coordinator and worker thread stacks are dumped every
# STALL_SECONDS into the console/worker logs.
set -euo pipefail

PY=${PY:-/root/autodl-tmp/hydraserve-venv/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/Qwen3.5-4B}
DATA=${DATA:-/root/autodl-tmp/data}
TRACE=${TRACE:-traces/w1_seed42.jsonl}
OUT=${OUT:-results/v3/p0_diagnose}
TRANSFER_BACKEND=${TRANSFER_BACKEND:-shm}
STALL_SECONDS=${STALL_SECONDS:-60}

mkdir -p "$OUT/workers"
export HYDRASERVE_STALL_DUMP_SECONDS="$STALL_SECONDS"
export PYTHONFAULTHANDLER=1

"$PY" -m hydraserve benchmark "$MODEL" "$DATA" --dataset synthetic \
  --trace "$TRACE" --adaptive --force-pd-tokens 1 \
  --prefill-short-policy never \
  --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware \
  --concurrency 32 --warmup 8 --arrival-pattern burst \
  --kv-quant int8 --prefix-cache-blocks 0 --cache-tokens 131072 \
  --block-size 256 --prefill-chunk-size 16384 --max-step-tokens 8192 \
  --pd-transfer-backend "$TRANSFER_BACKEND" --pd-transfer-target-mb 8 \
  --pd-transfer-inflight 2 --shm-ring-slots 3 --shm-ring-slot-mb 64 \
  --worker-log-dir "$OUT/workers" --output "$OUT/result.json" --seed 42 \
  2>&1 | tee "$OUT/coordinator.log"
