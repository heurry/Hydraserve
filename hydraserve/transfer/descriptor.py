from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TransferMode(str, Enum):
    FULL_TRANSFER = "full"
    QUANTIZED_TRANSFER = "quant"
    PARTIAL_TRANSFER = "partial"
    INTRA_GPU = "intra"


class RegionType(str, Enum):
    FULL_ATTN_KV = "full_attn_kv"
    LINEAR_SSM = "linear_ssm"
    LINEAR_CONV = "linear_conv"


@dataclass(frozen=True, slots=True)
class RegionDescriptor:
    region_type: RegionType
    layer_indices: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    quantized: bool
    src_gpu: int
    dst_gpu: int

    def __post_init__(self) -> None:
        if not self.layer_indices or any(index < 0 for index in self.layer_indices):
            raise ValueError("region must contain non-negative layer indices")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("region dimensions must be positive")
        if self.region_type in {RegionType.LINEAR_SSM, RegionType.LINEAR_CONV}:
            if self.dtype != "float32" or self.quantized:
                raise ValueError("linear recurrent state must be unquantized float32")
        if self.quantized and self.dtype != "int4":
            raise ValueError("a quantized region must declare dtype='int4'")
        if self.src_gpu < 0 or self.dst_gpu < 0 or self.src_gpu == self.dst_gpu:
            raise ValueError("source and destination must be distinct non-negative GPU ids")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["region_type"] = self.region_type.value
        result["layer_indices"] = list(self.layer_indices)
        result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionDescriptor":
        return cls(
            region_type=RegionType(data["region_type"]),
            layer_indices=tuple(data["layer_indices"]),
            shape=tuple(data["shape"]),
            dtype=str(data["dtype"]),
            quantized=bool(data["quantized"]),
            src_gpu=int(data["src_gpu"]),
            dst_gpu=int(data["dst_gpu"]),
        )


@dataclass(frozen=True, slots=True)
class StateTransferDescriptor:
    request_id: int
    model_name: str
    prompt_length: int
    first_token_id: int | None
    mode: TransferMode
    regions: tuple[RegionDescriptor, ...]

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.prompt_length <= 0 or not self.model_name:
            raise ValueError("invalid request metadata")
        if not self.regions:
            raise ValueError("a transfer must contain at least one region")
        kv_regions = [r for r in self.regions if r.region_type is RegionType.FULL_ATTN_KV]
        if self.mode is TransferMode.PARTIAL_TRANSFER and kv_regions:
            raise ValueError("partial transfer cannot include full-attention KV")
        if self.mode is TransferMode.QUANTIZED_TRANSFER:
            if not kv_regions or any(not region.quantized for region in kv_regions):
                raise ValueError("quantized transfer requires INT4 KV regions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "prompt_length": self.prompt_length,
            "first_token_id": self.first_token_id,
            "mode": self.mode.value,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateTransferDescriptor":
        return cls(
            request_id=int(data["request_id"]),
            model_name=str(data["model_name"]),
            prompt_length=int(data["prompt_length"]),
            first_token_id=data.get("first_token_id"),
            mode=TransferMode(data["mode"]),
            regions=tuple(RegionDescriptor.from_dict(r) for r in data["regions"]),
        )
