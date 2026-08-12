#!/bin/bash
# Run vLLM baseline for comparison
# Usage: bash scripts/run_baseline.sh [collocated|tp|dp]

set -e

MODE="${1:-collocated}"
MODEL_DIR="${MODEL_DIR:-/models}"
MODEL="${MODEL:-Qwen3.5-9B-AWQ}"
PORT="${PORT:-8001}"

case "${MODE}" in
    collocated)
        echo "Running vLLM baseline: 1-GPU collocated (reference A)"
        docker run --gpus '"device=0"' --rm \
            -v "${MODEL_DIR}:/models" -p "${PORT}:8000" \
            vllm/vllm-openai:latest \
            --model "/models/${MODEL}" \
            --quantization awq \
            --gpu-memory-utilization 0.9 \
            --max-model-len 131072
        ;;
    tp)
        echo "Running vLLM baseline: TP=2 (baseline C)"
        docker run --gpus all --rm \
            -v "${MODEL_DIR}:/models" -p "${PORT}:8000" \
            vllm/vllm-openai:latest \
            --model "/models/${MODEL}" \
            --quantization awq \
            --tensor-parallel-size 2 \
            --gpu-memory-utilization 0.9 \
            --max-model-len 131072
        ;;
    dp)
        echo "Running vLLM baseline: DP (two instances, baseline B)"
        docker run --gpus '"device=0"' --rm -d \
            -v "${MODEL_DIR}:/models" -p 8001:8000 \
            --name vllm-dp-0 \
            vllm/vllm-openai:latest \
            --model "/models/${MODEL}" \
            --quantization awq \
            --gpu-memory-utilization 0.9

        docker run --gpus '"device=1"' --rm -d \
            -v "${MODEL_DIR}:/models" -p 8002:8000 \
            --name vllm-dp-1 \
            vllm/vllm-openai:latest \
            --model "/models/${MODEL}" \
            --quantization awq \
            --gpu-memory-utilization 0.9

        echo "DP instances running on ports 8001, 8002"
        echo "Stop with: docker stop vllm-dp-0 vllm-dp-1"
        ;;
    *)
        echo "Usage: $0 [collocated|tp|dp]"
        exit 1
        ;;
esac
