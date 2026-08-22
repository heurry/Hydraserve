from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


def compute_head_slice_params(
    num_heads: int, tp_rank: int, tp_world_size: int, *, replication_factor: int = 1
) -> tuple[int, int]:
    """Return the contiguous local head slice for TP-aware KV transfer.

    ``replication_factor`` supports MLA/GQA layouts where the same KV shard is
    consumed by multiple tensor-parallel ranks.
    """
    if min(num_heads, tp_world_size, replication_factor) <= 0:
        raise ValueError("head topology values must be positive")
    if not 0 <= tp_rank < tp_world_size:
        raise ValueError("tp_rank is outside the topology")
    if tp_world_size % replication_factor:
        raise ValueError("replication_factor must divide tp_world_size")
    shard_count = tp_world_size // replication_factor
    if num_heads % shard_count:
        raise ValueError("KV heads must divide evenly across TP shards")
    heads_per_shard = num_heads // shard_count
    shard_rank = tp_rank // replication_factor
    return shard_rank * heads_per_shard, heads_per_shard


class TransferMode(str, Enum):
    FULL_TRANSFER = "full"
    QUANTIZED_TRANSFER = "quant"
    PARTIAL_TRANSFER = "partial"
    INTRA_GPU = "intra"


class StateType(str, Enum):
    """Transferable model state families.

    Values remain wire-compatible with the original ``RegionType`` enum while
    allowing hybrid models to declare additional attention state layouts.
    """

    FULL_ATTN_KV = "full_attn_kv"
    SLIDING_WINDOW_KV = "sliding_window_kv"
    DSA_KV = "dsa_kv"
    MLA_KV = "mla_kv"
    LINEAR_SSM = "linear_ssm"
    LINEAR_CONV = "linear_conv"


# Public compatibility alias. Existing connectors can migrate independently.
RegionType = StateType


@dataclass(frozen=True, slots=True)
class RegionDescriptor:
    region_type: StateType
    layer_indices: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    quantized: bool
    src_gpu: int
    dst_gpu: int
    src_tp_rank: int = 0
    dst_tp_rank: int = 0
    tp_world_size: int = 1

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
        if self.tp_world_size <= 0:
            raise ValueError("tp_world_size must be positive")
        if not (0 <= self.src_tp_rank < self.tp_world_size):
            raise ValueError("src_tp_rank is outside the TP topology")
        if not (0 <= self.dst_tp_rank < self.tp_world_size):
            raise ValueError("dst_tp_rank is outside the TP topology")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["region_type"] = self.region_type.value
        result["layer_indices"] = list(self.layer_indices)
        result["shape"] = list(self.shape)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionDescriptor":
        return cls(
            region_type=StateType(data["region_type"]),
            layer_indices=tuple(data["layer_indices"]),
            shape=tuple(data["shape"]),
            dtype=str(data["dtype"]),
            quantized=bool(data["quantized"]),
            src_gpu=int(data["src_gpu"]),
            dst_gpu=int(data["dst_gpu"]),
            src_tp_rank=int(data.get("src_tp_rank", 0)),
            dst_tp_rank=int(data.get("dst_tp_rank", 0)),
            tp_world_size=int(data.get("tp_world_size", 1)),
        )


@dataclass(frozen=True, slots=True)
class StateTransferDescriptor:
    request_id: int
    model_name: str
    prompt_length: int
    first_token_id: int | None
    mode: TransferMode
    regions: tuple[RegionDescriptor, ...]
    state_token_count: int | None = None
    streamed_kv: bool = False
    kv_chunk_ranges: tuple[tuple[int, int], ...] = ()
    host_cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.request_id < 0 or self.prompt_length <= 0 or not self.model_name:
            raise ValueError("invalid request metadata")
        if not self.regions:
            raise ValueError("a transfer must contain at least one region")
        if self.state_token_count is None:
            object.__setattr__(self, "state_token_count", self.prompt_length)
        if not 0 < self.state_token_count <= self.prompt_length:
            raise ValueError("state_token_count must be in [1, prompt_length]")
        if self.prompt_length - self.state_token_count > 1:
            raise ValueError("only full or N-1 prompt state is supported")
        kv_regions = [r for r in self.regions if r.region_type is RegionType.FULL_ATTN_KV]
        if self.mode is TransferMode.PARTIAL_TRANSFER and kv_regions:
            raise ValueError("partial transfer cannot include full-attention KV")
        if self.mode is TransferMode.QUANTIZED_TRANSFER:
            if not kv_regions or any(not region.quantized for region in kv_regions):
                raise ValueError("quantized transfer requires INT4 KV regions")
        previous = 0
        for start, end in self.kv_chunk_ranges:
            if start != previous or end <= start or end > self.prompt_length:
                raise ValueError("KV chunk ranges must be contiguous and within the prompt")
            previous = end
        if self.streamed_kv:
            if self.mode is TransferMode.PARTIAL_TRANSFER or not kv_regions:
                raise ValueError("streamed KV requires a KV transfer region")
            if not self.kv_chunk_ranges or previous != self.prompt_length:
                raise ValueError("streamed KV chunks must cover the complete prompt")
        elif self.kv_chunk_ranges:
            raise ValueError("KV chunk ranges require streamed_kv=True")
        if self.host_cache_hit:
            if self.streamed_kv or self.mode is TransferMode.PARTIAL_TRANSFER:
                raise ValueError("host KV restore cannot also stream or recompute KV")
            if not kv_regions:
                raise ValueError("host KV restore requires a KV region descriptor")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "prompt_length": self.prompt_length,
            "first_token_id": self.first_token_id,
            "mode": self.mode.value,
            "regions": [region.to_dict() for region in self.regions],
            "state_token_count": self.state_token_count,
            "streamed_kv": self.streamed_kv,
            "kv_chunk_ranges": [list(item) for item in self.kv_chunk_ranges],
            "host_cache_hit": self.host_cache_hit,
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
            state_token_count=data.get("state_token_count"),
            streamed_kv=bool(data.get("streamed_kv", False)),
            kv_chunk_ranges=tuple(
                (int(item[0]), int(item[1]))
                for item in data.get("kv_chunk_ranges", ())
            ),
            host_cache_hit=bool(data.get("host_cache_hit", False)),
        )
