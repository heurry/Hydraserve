from __future__ import annotations

import json

import pytest

from hydraserve.config import LayerKind, get_model_config, load_model_config


def test_presets_have_architecture_driven_state_sizes() -> None:
    model = get_model_config("Qwen3.5-9B-AWQ")
    assert model.num_linear_layers == 24
    assert model.num_full_attention_layers == 8
    assert model.kv_bytes_per_token_bf16 == 32 * 1024
    assert 25_000_000 < model.recurrent_state_bytes < 27_000_000


def test_loads_arbitrary_size_from_nested_hf_config(tmp_path) -> None:
    layer_types = ["linear_attention"] * 47 + ["full_attention"]
    path = tmp_path / "Qwen-custom-13B" / "config.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "hidden_size": 4608,
                    "num_hidden_layers": 48,
                    "layer_types": layer_types,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 6,
                    "head_dim": 192,
                    "linear_num_key_heads": 18,
                    "linear_key_head_dim": 96,
                    "linear_num_value_heads": 36,
                    "linear_value_head_dim": 96,
                    "linear_conv_kernel_dim": 4,
                    "mamba_ssm_dtype": "float32",
                },
            }
        ),
        encoding="utf-8",
    )
    model = load_model_config(path.parent)
    assert model.name == "Qwen-custom-13B"
    assert model.num_hidden_layers == 48
    assert model.num_kv_heads == 6
    assert model.layer_types[-1] is LayerKind.FULL_ATTENTION
    assert model.ssm_state_shape == (47, 18, 96, 96)


def test_rejects_unknown_architecture(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model architecture"):
        load_model_config(path)
