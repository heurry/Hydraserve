#!/bin/bash
# Historical regression on d8f3c87: re-run seed42 with the exact old settings
# (concurrency 32, block 256, cache 131072, chunk 16384, kv int8, prefix off,
# no HTTP) plus --max-step-tokens 65536 so 32K longs are not deferred behind
# active shorts. Loads: W1-128 and W1-32; topologies D0 / P0 / P2-C.
set -uo pipefail

PY=/root/autodl-tmp/hydraserve-venv/bin/python
MODEL=/root/autodl-tmp/Qwen3.5-4B
DATA=/root/autodl-tmp/data
OUT=results/v3/regress
SEED=42
LOG=/tmp/v3_regress.log

mkdir -p "$OUT"
: > "$LOG"

COMMON=(--concurrency 32 --warmup 8 --arrival-pattern burst
        --kv-quant int8 --prefix-cache-blocks 0 --cache-tokens 131072
        --block-size 256 --prefill-chunk-size 16384 --max-step-tokens 65536
        --pd-transfer-backend shm-ring --pd-transfer-target-mb 8
        --pd-transfer-inflight 2 --shm-ring-slots 3 --shm-ring-slot-mb 64
        --seed "$SEED")

run_group() {
  local trace="$1" name="$2"; shift 2
  echo "=== GROUP $name ($(basename "$trace")) ===" | tee -a "$LOG"
  "$PY" -m hydraserve benchmark "$MODEL" "$DATA" --dataset synthetic --trace "$trace" \
    "${COMMON[@]}" "$@" \
    --worker-log-dir "$OUT/${name}_workers" \
    --output "$OUT/${name}.json" 2>&1 | tail -2 | tee -a "$LOG" || true
}

# W1-128 (short output 128)
run_group traces/w1_seed42.jsonl   w1_d0  --dp-devices 0 1 2 3
run_group traces/w1_seed42.jsonl   w1_p0  --adaptive --force-pd-tokens 1 --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware
run_group traces/w1_seed42.jsonl   w1_p2c --adaptive --conditional-pd-tokens 8192 --prefill-short-policy never --prefill-devices 0 --decode-devices 1 2 3 --pd-schedule kv-aware

# W1-32 (short output 32)
run_group traces/w1s32_seed42.jsonl w32_d0  --dp-devices 0 1 2 3
run_group traces/w1s32_seed42.jsonl w32_p0  --adaptive --force-pd-tokens 1 --prefill-devices 0 1 --decode-devices 2 3 --pd-schedule kv-aware
run_group traces/w1s32_seed42.jsonl w32_p2c --adaptive --conditional-pd-tokens 8192 --prefill-short-policy never --prefill-devices 0 --decode-devices 1 2 3 --pd-schedule kv-aware

echo "=== REGRESS DONE ===" | tee -a "$LOG"
