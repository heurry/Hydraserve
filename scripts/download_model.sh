#!/bin/bash
# Download Qwen3.5/3.6 model weights
# Usage: bash scripts/download_model.sh [4B|9B|27B|all]

set -e

MODEL_DIR="${MODEL_DIR:-/models}"

download_model() {
    local model_id=$1
    local local_dir=$2
    echo "Downloading ${model_id} → ${local_dir}..."
    huggingface-cli download "${model_id}" --local-dir "${local_dir}"
}

case "${1:-all}" in
    4B)
        download_model "Qwen/Qwen3.5-4B" "${MODEL_DIR}/Qwen3.5-4B"
        ;;
    9B)
        download_model "Qwen/Qwen3.5-9B-AWQ" "${MODEL_DIR}/Qwen3.5-9B-AWQ"
        ;;
    27B)
        download_model "Qwen/Qwen3.6-27B-AWQ" "${MODEL_DIR}/Qwen3.6-27B-AWQ"
        ;;
    all)
        download_model "Qwen/Qwen3.5-4B" "${MODEL_DIR}/Qwen3.5-4B"
        download_model "Qwen/Qwen3.5-9B-AWQ" "${MODEL_DIR}/Qwen3.5-9B-AWQ"
        download_model "Qwen/Qwen3.6-27B-AWQ" "${MODEL_DIR}/Qwen3.6-27B-AWQ"
        ;;
    *)
        echo "Usage: $0 [4B|9B|27B|all]"
        exit 1
        ;;
esac

echo "Done. Models saved to ${MODEL_DIR}/"
ls -la "${MODEL_DIR}/"
