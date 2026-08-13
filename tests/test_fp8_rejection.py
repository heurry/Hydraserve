from pathlib import Path

import pytest

pytest.importorskip("torch")

from hydraserve.model.runtime import QwenTextRuntime


def test_block_scaled_fp8_is_rejected_before_tensor_loading() -> None:
    model_dir = Path("/mnt/nvme-data/models/LLM_model/Qwen3.6-27B-FP8")
    if not model_dir.is_dir():
        pytest.skip("local FP8 checkpoint is absent")
    with pytest.raises(NotImplementedError, match="FP8 GEMM"):
        QwenTextRuntime.from_checkpoint(model_dir, device="cpu")
