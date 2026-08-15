from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hydraserve.config import ModelConfig


@runtime_checkable
class ModelAdapter(Protocol):
    """Structural contract consumed by HydraServe generation backends."""

    @property
    def config(self) -> ModelConfig: ...

    @property
    def device(self) -> Any: ...

    @property
    def input_device(self) -> Any: ...

    def prefill(
        self,
        input_ids,
        *,
        chunk_size: int,
        paged_cache=None,
        request_id: int | None = None,
    ): ...

    def decode_batch(self, input_ids, states, paged_cache, request_ids): ...
