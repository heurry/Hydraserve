from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from hydraserve.config import LayerKind, ModelConfig


class ModelAdapter(ABC):
    """Boundary between engine scheduling and a concrete model runtime."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def layer_type(self, layer_index: int) -> LayerKind:
        if not 0 <= layer_index < self.config.num_hidden_layers:
            raise IndexError(f"layer {layer_index} is out of range")
        return self.config.layer_types[layer_index]

    @abstractmethod
    def prefill_layer(
        self, layer_index: int, hidden_states: Any, positions: Iterable[int]
    ) -> tuple[Any, dict[str, Any]]:
        """Return next hidden states and the transferable state for this layer."""

    @abstractmethod
    def decode_token(
        self, token_id: int, kv_cache: Any, recurrent_state: Any
    ) -> tuple[Any, Any]:
        """Return logits and updated recurrent state."""
