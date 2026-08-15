from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.model import ModelAdapter, QwenHybridAdapter, QwenTextRuntime
from tests.test_runtime import make_weights


def test_qwen_adapter_is_the_production_runtime(tiny_model) -> None:
    runtime = QwenHybridAdapter(
        tiny_model,
        make_weights(tiny_model),
        use_triton=False,
        use_flash_attention=False,
    )
    assert isinstance(runtime, QwenTextRuntime)
    assert isinstance(runtime, ModelAdapter)
    logits, state = runtime.prefill(torch.tensor([[1, 2, 3]]), chunk_size=2)
    assert logits.shape == (1, 1, tiny_model.vocab_size)
    assert state.sequence_length == 3


def test_model_adapter_protocol_rejects_configuration_only_objects(tiny_model) -> None:
    class ConfigurationOnly:
        config = tiny_model

    assert not isinstance(ConfigurationOnly(), ModelAdapter)
