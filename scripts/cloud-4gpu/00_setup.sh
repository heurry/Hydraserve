#!/usr/bin/env bash
# 云端一键准备：clone 之后先跑本脚本，装依赖并自检环境。
# 不安装 flash-attn（脚本默认 --no-flash-attention，可免编译直接跑）；
# 若环境已装好 flash-attn，跑矩阵时设 FA_OPTS="" 即可启用。
set -euo pipefail

echo "== 安装 HydraServe 与 GPU 依赖 =="
python -m pip install -e '.[gpu]' '.[serve]'

echo "== 环境自检 =="
python -c 'import torch; print("torch", torch.__version__, "| cuda", torch.version.cuda, "| devices", torch.cuda.device_count())'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L
else
  echo "当前为无 GPU 模式（nvidia-smi 不可用）：环境安装本身不受影响，"
  echo "也可以顺手跑 CPU 测试与模型/数据检查："
  echo "  python -m pytest"
  echo "  python -m hydraserve inspect-models <模型目录>"
  echo "  python -m hydraserve inspect-datasets <数据目录>"
  echo "完成后切到 GPU 模式再跑 smoke 与矩阵。"
fi
python -m hydraserve --help >/dev/null && echo "hydraserve CLI OK"

echo
echo "接下来："
echo "  1) 上传模型与数据集后自检："
echo "     python -m hydraserve inspect-models <模型目录>"
echo "     python -m hydraserve inspect-datasets <数据目录>"
echo "  2) 门禁 smoke："
echo "     MODEL=<模型路径> DATASETS=<数据路径> bash scripts/cloud-4gpu/00_smoke_1p3d.sh"
echo "  3) 压测矩阵："
echo "     MODEL=<模型路径> DATASETS=<数据路径> bash scripts/cloud-4gpu/01_run_matrix.sh"
