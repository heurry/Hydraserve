"""Direct safetensors checkpoint access without a model framework."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class PackedInt4Weight:
    """Compressed-tensors asymmetric grouped INT4 linear weight."""

    packed: Any
    scale: Any
    zero_point: Any
    original_shape: tuple[int, int]
    group_size: int = 128

    @property
    def shape(self) -> tuple[int, int]:
        return self.original_shape


@dataclass(frozen=True, slots=True)
class BlockScaledFP8Weight:
    """E4M3 weight with a two-dimensional inverse scale per weight block."""

    data: Any
    scale_inv: Any
    original_shape: tuple[int, int]
    block_size: tuple[int, int] = (128, 128)

    @property
    def shape(self) -> tuple[int, int]:
        return self.original_shape


class ShardedSafeTensorLoader:
    """Resolve tensors from a Hugging Face sharded safetensors index.

    Safetensors is a file format dependency, not an inference backend. Tensors
    are loaded on demand so callers can control CPU/GPU placement layer by layer.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise NotADirectoryError(self.model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            self.weight_map = dict(index["weight_map"])
        elif single_path.is_file():
            from safetensors import safe_open

            with safe_open(single_path, framework="np") as handle:
                self.weight_map = {key: single_path.name for key in handle.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint found in {self.model_dir}")
        self._lock = RLock()

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self, prefix: str | None = None) -> tuple[str, ...]:
        keys = self.weight_map if prefix is None else (k for k in self.weight_map if k.startswith(prefix))
        return tuple(sorted(keys))

    def tensor(self, name: str, *, device: str | Any = "cpu", dtype: Any = None):
        """Load one tensor, optionally moving/casting after safe deserialization."""
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint does not contain {name!r}") from exc
        from safetensors import safe_open

        # safe_open handles are intentionally short-lived. This avoids retaining
        # one mmap per shard in long-running workers and is thread safe.
        with self._lock, safe_open(
            self.model_dir / shard, framework="pt", device="cpu"
        ) as handle:
            tensor = handle.get_tensor(name)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if str(device) != "cpu":
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor

    def tensors(
        self, prefix: str, *, device: str | Any = "cpu", dtype: Any = None
    ) -> Iterator[tuple[str, Any]]:
        for name in self.keys(prefix):
            yield name, self.tensor(name, device=device, dtype=dtype)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint does not contain {name!r}") from exc
        from safetensors import safe_open

        with self._lock, safe_open(self.model_dir / shard, framework="np") as handle:
            return tuple(handle.get_slice(name).get_shape())


LANGUAGE_PREFIX = "model.language_model"


def layer_prefix(layer_index: int) -> str:
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    return f"{LANGUAGE_PREFIX}.layers.{layer_index}"
