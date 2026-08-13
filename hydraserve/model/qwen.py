from __future__ import annotations

from typing import Any, Iterable

from hydraserve.config import ModelConfig, get_model_config, load_model_config
from hydraserve.model.adapter import ModelAdapter


class QwenHybridAdapter(ModelAdapter):
    """Configuration-complete Qwen adapter with runtime hooks left explicit.

    This class establishes the engine/model contract without pretending that a
    Transformers checkpoint has been loaded. GPU execution is added behind these
    hooks in the kernel phase.
    """

    def __init__(self, model: str | ModelConfig) -> None:
        if isinstance(model, ModelConfig):
            config = model
        else:
            try:
                config = get_model_config(model)
            except KeyError:
                config = load_model_config(model)
        super().__init__(config)

    def prefill_layer(
        self, layer_index: int, hidden_states: Any, positions: Iterable[int]
    ) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError("Qwen weight loading and GPU prefill are not implemented yet")

    def decode_token(
        self, token_id: int, kv_cache: Any, recurrent_state: Any
    ) -> tuple[Any, Any]:
        raise NotImplementedError("Qwen GPU decode is not implemented yet")
