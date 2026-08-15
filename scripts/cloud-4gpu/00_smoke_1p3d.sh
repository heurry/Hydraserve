#!/usr/bin/env bash
# 1P+3D 首次实卡 smoke：4 卡拓扑检查、multi-worker 后端启动、HTTP 服务、
# 短/长 prompt 两条路由路径，以及最小 benchmark。
#
# 用法（云端机器上，先 pip install -e '.[gpu]' '.[serve]'）：
#   MODEL=/path/to/Qwen3.5-4B DATASETS=/path/to/benchmark \
#   bash scripts/cloud-4gpu/00_smoke_1p3d.sh
#
# N>1 的 1P+ND 从未在真实硬件上验证过，此脚本是矩阵前的必经门禁。
set -euo pipefail

MODEL="${MODEL:?设置 MODEL=/path/to/Qwen3.5-4B}"
DATASETS="${DATASETS:?设置 DATASETS=/path/to/benchmark}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-benchmark_output/cloud-4gpu/smoke}"
# 默认禁用 FlashAttention（云端免装 flash-attn 可直接跑）；装好后设 FA_OPTS=""
FA_OPTS="${FA_OPTS:---no-flash-attention}"
mkdir -p "$LOG_DIR"

echo "== GPU 数量 =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "错误：nvidia-smi 不可用，当前可能是无 GPU 模式；请切换到 4 卡 GPU 模式后重跑"
  exit 1
fi
nvidia-smi -L | tee "$LOG_DIR/gpus.txt"
[ "$(nvidia-smi -L | wc -l)" -ge 4 ] || { echo "错误：需要至少 4 张 GPU"; exit 1; }

echo "== 拓扑（重点看 P2P 列与 x16） =="
nvidia-smi topo -m | tee "$LOG_DIR/topo.txt"

echo "== 启动 1P+3D 服务（prefill=cuda:0, decode=cuda:1..3） =="
python -m hydraserve serve "$MODEL" \
  --adaptive --device cuda:0 --decode-devices cuda:1 cuda:2 cuda:3 \
  --port "$PORT" \
  --max-batch-size 64 --max-active-requests 128 \
  --max-queue-size 1024 --max-queue-tokens 1048576 \
  --cache-tokens 65536 --kv-headroom-blocks 128 \
  $FA_OPTS \
  >"$LOG_DIR/serve.log" 2>&1 &
SERVER_PID=$!
stop_server() { kill "$SERVER_PID" 2>/dev/null || true; }
trap stop_server EXIT

echo "== 等待 /health（3 个 decode worker 加载模型，可能需数分钟） =="
ready=0
for _ in $(seq 1 150); do
  if curl -sf "http://127.0.0.1:$PORT/health" >"$LOG_DIR/health.json" 2>/dev/null; then
    ready=1; break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "服务未就绪，serve.log 尾部："; tail -50 "$LOG_DIR/serve.log"; exit 1
fi
cat "$LOG_DIR/health.json"

echo "== 短 prompt（预期 collocated 路由） =="
curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
echo

echo "== 长 prompt（约 4K token，预期 PD 路由） =="
python3 - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
prompt = "HydraServe long-context smoke test. " * 400  # ~4.4K tokens
body = json.dumps({"model": "Qwen3.5-4B", "prompt": prompt, "max_tokens": 16, "temperature": 0}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as resp:
    out = json.load(resp)
print("choices:", out["choices"][0]["text"][:80], "| usage:", out["usage"])
PY

echo "== 关闭 HTTP 服务，跑进程内最小 benchmark（c=1 与 c=4） =="
stop_server
sleep 5

for concurrency in 1 4; do
  python -m hydraserve benchmark "$MODEL" "$DATASETS" \
    --dataset gsm8k --limit 8 --warmup 2 --max-new-tokens 16 \
    --concurrency "$concurrency" \
    --adaptive --device cuda:0 --decode-devices cuda:1 cuda:2 cuda:3 \
    $FA_OPTS \
    --output "$LOG_DIR/gsm8k_c${concurrency}.json"
done

echo "== smoke 完成。检查各 JSON 的 route_counts（应出现 worker 0/1/2 的绑定） =="
python3 - "$LOG_DIR" <<'PY'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("gsm8k_c*.json")):
    data = json.loads(path.read_text())
    print(path.name, "succeeded:", data["succeeded"], "failed:", data["failed"],
          "routes:", data["route_counts"])
PY
