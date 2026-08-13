from pathlib import Path

import pytest

from hydraserve.model.weights import ShardedSafeTensorLoader, layer_prefix


def test_checkpoint_index_and_shapes() -> None:
    model_dir = "/mnt/nvme-data/models/LLM_model/Qwen3.5-4B"
    if not Path(model_dir).is_dir():
        pytest.skip("local Qwen3.5-4B checkpoint is not available")
    loader = ShardedSafeTensorLoader(model_dir)
    assert loader.tensor_shape("model.language_model.embed_tokens.weight") == (248320, 2560)
    assert loader.tensor_shape(f"{layer_prefix(0)}.linear_attn.in_proj_qkv.weight") == (8192, 2560)
    assert loader.tensor_shape(f"{layer_prefix(3)}.self_attn.q_proj.weight") == (8192, 2560)
