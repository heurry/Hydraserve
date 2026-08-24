#!/bin/bash
set -euo pipefail
MODEL_PATH=${1:-/root/autodl-tmp/Qwen3.5-4B}
OUTPUT_DIR=${2:-results/vllm_v4}
TRACES_DIR=traces
RATE=2.0
CONCURRENCY=8
SEEDS="42 43 44"
mkdir -p $OUTPUT_DIR

start_vllm() {
    local max_len=$1
    echo "Starting vLLM with max-model-len=$max_len ..."
    vllm serve $MODEL_PATH --data-parallel-size 4 --kv-cache-dtype fp8 --no-enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 16 --gpu-memory-utilization 0.7 --max-model-len $max_len --port 8000 --host 0.0.0.0 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"
    # Wait until the engine can actually serve a completion, not just metadata.
    # /v1/models responds before model weights are loaded, so a real
    # generation (min_tokens=1, max_tokens=1) is the reliable readiness probe.
    # -f is essential: plain `curl -s` returns exit 0 even on HTTP 502, so the
    # probe would pass while the engine is still cold-loading.
    for i in $(seq 1 300); do
        if curl -fsS -m 60 -o /dev/null \
            -H "Content-Type: application/json" \
            -d '{"prompt":"hello","max_tokens":1,"min_tokens":1,"stream":false}' \
            http://127.0.0.1:8000/v1/completions 2>/dev/null; then
            echo "vLLM ready (real completion OK, after ~$((i * 2))s)"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: vLLM timeout (engine not serving after ~600s)"; kill $VLLM_PID; return 1
}

stop_vllm() {
    if [ -n "${VLLM_PID:-}" ]; then kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true; sleep 5; fi
}

run_trace() {
    local tf=$1 of=$2 sd=$3
    echo "=== Running $tf seed=$sd ==="
    python scripts/vllm_trace_bench.py --endpoint http://127.0.0.1:8000 --trace $tf --output $of --request-rate $RATE --arrival-pattern poisson --concurrency $CONCURRENCY --seed $sd --warmup 8
}

echo "=== vLLM 4xDP Benchmark ==="
start_vllm 20000
for s in $SEEDS; do run_trace traces/r1_rag_qa_seed${s}.jsonl results/vllm_v4/r1_vllm_dp4_seed${s}.json $s || true; done
run_trace traces/r3_code_analysis_seed42.jsonl results/vllm_v4/r3_vllm_dp4_seed42.json 42 || true
stop_vllm
# r2 longs tokenize to ~17.1k + 1024 output = ~18.1k, so 18000 overflows;
# use 20000 like r1/r3.
start_vllm 20000
for s in $SEEDS; do run_trace traces/r2_doc_summary_seed${s}.jsonl results/vllm_v4/r2_vllm_dp4_seed${s}.json $s || true; done
stop_vllm
echo "=== Done. Compare with results/v4/ ==="

