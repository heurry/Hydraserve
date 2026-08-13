from __future__ import annotations

import pytest

from hydraserve.config import ModelConfig


@pytest.fixture
def tiny_model() -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "name": "tiny-qwen-hybrid",
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 4,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 3,
            "intermediate_size": 64,
            "vocab_size": 64,
        }
    )
