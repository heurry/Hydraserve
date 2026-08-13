"""Architecture-driven model configuration.

HydraServe is not limited to a list of parameter counts.  Presets make common
models convenient, while :func:`load_model_config` accepts any compatible local
``config.json`` without importing Transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class LayerKind(str, Enum):
    LINEAR_ATTENTION = "linear_attention"
    FULL_ATTENTION = "full_attention"


def _required(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    raise ValueError(f"model config is missing required field: {' or '.join(names)}")


def _optional(data: Mapping[str, Any], default: Any, *names: str) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def _normalize_layer_kind(value: Any) -> LayerKind:
    normalized = str(value).lower().replace("-", "_")
    if normalized in {"full", "attention", "full_attention", "self_attention"}:
        return LayerKind.FULL_ATTENTION
    if normalized in {"linear", "linear_attention", "gdn", "gated_delta_net"}:
        return LayerKind.LINEAR_ATTENTION
    raise ValueError(f"unsupported hybrid layer type {value!r}")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[LayerKind, ...]
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    max_position_embeddings: int = 262_144
    recurrent_dtype: str = "float32"

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {', '.join(invalid)}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, expected "
                f"{self.num_hidden_layers}"
            )
        if not self.full_attention_layer_indices:
            raise ValueError("a hybrid model must contain at least one full-attention layer")
        if not self.linear_layer_indices:
            raise ValueError("a hybrid model must contain at least one linear-attention layer")
        if self.recurrent_dtype != "float32":
            raise ValueError("recurrent state must use float32")

    @property
    def linear_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind is LayerKind.LINEAR_ATTENTION)

    @property
    def full_attention_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.layer_types) if kind is LayerKind.FULL_ATTENTION)

    @property
    def num_linear_layers(self) -> int:
        return len(self.linear_layer_indices)

    @property
    def num_full_attention_layers(self) -> int:
        return len(self.full_attention_layer_indices)

    @property
    def kv_bytes_per_token_bf16(self) -> int:
        return self.num_full_attention_layers * self.num_kv_heads * self.head_dim * 4

    @property
    def ssm_state_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_linear_layers,
            self.linear_num_key_heads,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
        )

    @property
    def conv_state_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_linear_layers,
            self.linear_num_key_heads,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
        )

    @property
    def recurrent_state_bytes(self) -> int:
        elements = _product(self.ssm_state_shape) + _product(self.conv_state_shape)
        return elements * 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, name: str | None = None) -> "ModelConfig":
        layers = int(_required(data, "num_hidden_layers", "n_layer"))
        raw_layer_types = _optional(
            data, None, "layer_types", "layers_block_type", "layer_type_list"
        )
        if raw_layer_types is not None:
            if not isinstance(raw_layer_types, Sequence) or isinstance(raw_layer_types, str):
                raise ValueError("layer_types must be a sequence")
            layer_types = tuple(_normalize_layer_kind(item) for item in raw_layer_types)
        else:
            interval = int(_optional(data, 4, "full_attention_interval"))
            if interval <= 1:
                raise ValueError("full_attention_interval must be greater than one")
            layer_types = tuple(
                LayerKind.FULL_ATTENTION if (index + 1) % interval == 0
                else LayerKind.LINEAR_ATTENTION
                for index in range(layers)
            )

        hidden_size = int(_required(data, "hidden_size", "n_embd"))
        attention_heads = int(_required(data, "num_attention_heads", "n_head"))
        head_dim = int(_optional(data, hidden_size // attention_heads, "head_dim"))
        linear_key_dim = int(_optional(data, 128, "linear_key_head_dim"))
        return cls(
            name=name or str(_optional(data, "custom-hybrid-model", "name", "_name_or_path")),
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            layer_types=layer_types,
            num_attention_heads=attention_heads,
            num_kv_heads=int(_required(data, "num_key_value_heads", "num_kv_heads")),
            head_dim=head_dim,
            linear_num_key_heads=int(_optional(data, attention_heads, "linear_num_key_heads")),
            linear_key_head_dim=linear_key_dim,
            linear_num_value_heads=int(
                _optional(data, attention_heads, "linear_num_value_heads")
            ),
            linear_value_head_dim=int(
                _optional(data, linear_key_dim, "linear_value_head_dim")
            ),
            linear_conv_kernel_dim=int(
                _optional(data, 4, "linear_conv_kernel_dim", "conv_kernel")
            ),
            max_position_embeddings=int(
                _optional(data, 262_144, "max_position_embeddings", "max_sequence_length")
            ),
            recurrent_dtype=str(_optional(data, "float32", "mamba_ssm_dtype", "recurrent_dtype")),
        )


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def _preset(name: str, hidden: int, layers: int, heads: int, value_heads: int) -> ModelConfig:
    return ModelConfig.from_mapping(
        {
            "name": name,
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": value_heads,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        }
    )


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3.5-4b": _preset("Qwen3.5-4B", 2560, 32, 16, 32),
    "qwen3.5-9b": _preset("Qwen3.5-9B", 4096, 32, 16, 32),
    "qwen3.6-27b": _preset("Qwen3.6-27B", 5120, 64, 24, 48),
}


def get_model_config(name: str) -> ModelConfig:
    normalized = name.lower().rstrip("/").split("/")[-1]
    for suffix in ("-awq", "-gptq", "-fp8", "-bf16"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    try:
        return MODEL_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(
            f"unknown preset {name!r}; use load_model_config(path) for any compatible "
            f"model, or choose: {supported}"
        ) from exc


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    architecture = data.get("model_type")
    model_data = data.get("text_config", data)
    if not isinstance(model_data, Mapping):
        raise ValueError("text_config must be a JSON object")
    supported_architectures = {"qwen3_5", "qwen3_6", "qwen3_5_text", "qwen3_6_text"}
    text_architecture = model_data.get("model_type", architecture)
    if text_architecture and text_architecture not in supported_architectures:
        raise ValueError(f"unsupported model architecture {text_architecture!r}")
    model_name = str(data.get("_name_or_path") or config_path.parent.name)
    return ModelConfig.from_mapping(model_data, name=model_name)


def discover_model_configs(root: str | Path) -> dict[Path, ModelConfig | Exception]:
    """Inspect immediate model directories without loading any weights."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)
    discovered: dict[Path, ModelConfig | Exception] = {}
    for config_path in sorted(root_path.glob("*/config.json")):
        try:
            discovered[config_path.parent] = load_model_config(config_path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            discovered[config_path.parent] = exc
    return discovered
