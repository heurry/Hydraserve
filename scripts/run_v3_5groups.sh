#!/bin/bash
# Run the 5-group short-32 comparison (W1 short output 32 tokens) sequentially.
# Groups: D0 (4xDP), D-control (2xDP GPU 2,3), P0 (2P+2D all-PD),
#         P3 (2P+2D conditional), P2-C (1P+3D, long PD + short D-collocated).
# prefill-chunk 16384 avoids the 32K long-request prefill activation OOM.
set -uo pipefail

PY=/root/autodl-tmp/hydraserve-venv/bin/python
MODEL=/root/autodl-tmp/Qwen3.5-4B
DATA=/root/autodl-tmp/data
TRACE=traces/w1s32_seed42.jsonl
OUT=results/v3/new
SEED=42

COMMON=(--dataset synthetic --trace "$TRACE"
        --concurrency 32 --warmup 8 --arrival-pattern burst
        --kv-quant int8 --prefix-cache-blocks 0 --cache-tokens 131072
        --block-size 256 --prefill-chunk-size 16384
        --pd-transfer-backend shm-ring --pd-transfer-target-mb 8
        --pd-transfer-inflight 2 --shm-ring-slots 3 --shm-ring-slot-mb 64
        --seed "$SEED")

run_group() {
  local name="$1"; shift
  echo "=== GROUP $name ===" | tee -a /tmp/v3_groups.log
  "$PY" -m hydraserve benchmark "$MODEL" "$DATA" "${COMMON[@]}" "$@" \
    --worker-log-dir "$OUT/${name}_workers" \
    --output "$OUT/${name}_seed${SEED}.json" 2>&1 | tail -2 | tee -a /tmp/v3_groups.log || true
}

run_group d0 --dp-devices 0 1 2 3
run_group dctrl --dp-devices 2 3
run_group p0 --adaptive --force-pd-tokens 1 --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware
run_group p3 --adaptive --conditional-pd-tokens 8192 --prefill-short-policy never \
           --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware
run_group p2c --adaptive --conditional-pd-tokens 8192 --prefill-short-policy never \
           --prefill-devices 0 --decode-devices 1 2 3 --pd-schedule kv-aware

echo "=== ALL DONE ===" | tee -a /tmp/v3_groups.log
