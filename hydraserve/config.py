"""
HydraServe configuration system.

Centralized config for all components: model specs, hardware topology,
transfer backends, engine settings, and benchmark parameters.
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple


class TransferMode(Enum):
    """Transfer mode describing the strategy for state migration."""
    FULL_TRANSFER = "full"           # NVLink: full BF16 KV + recurrent states
    QUANTIZED_TRANSFER = "quant"     # PCIe P2P: INT4 KV + recurrent states
    PARTIAL_TRANSFER = "partial"     # Low bandwidth: recurrent states only + KV recompute
    INTRA_GPU = "intra"              # MPS: same-GPU separation, zero-copy


class ServingMode(Enum):
    """Top-level serving strategy."""
    COLLOCATED = "collocated"        # Single GPU prefill + decode
    PD_DISAGGREGATED = "pd_disaggregated"  # GPU 0 prefill, GPU 1 decode
    DP = "dp"                        # Two independent instances
    TP = "tp"                        # Tensor parallelism (via vLLM baseline)


class RouterDecision(Enum):
    """Routing decision per request."""
    COLLOCATED = "collocated"
    PD_DISAGGREGATED = "pd_disaggregated"


class RequestState(Enum):
    """Request lifecycle states."""
    WAITING = "waiting"
    PREFILL_RUNNING = "prefill_running"
    PREFILL_TRANSFER_PENDING = "prefill_transfer_pending"
    READY = "ready"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# ─── Model Specifications ───────────────────────────────────────────

@dataclass
class ModelSpec:
    """Specification for a supported model architecture."""
    name: str
    hidden_size: int
    num_hidden_layers: int
    full_attention_interval: int        # e.g. 4 => every 4th layer is full attention
    num_attention_heads: int
    num_key_value_heads: int            # GQA KV heads
    head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    mamba_ssm_dtype: str = "float32"    # FP32 for recurrent state
    max_position_embeddings: int = 262144
    vocab_size: int = 248320
    rope_theta: float = 1000000.0
    partial_rotary_factor: float = 0.25
    use_mrope: bool = True              # Qwen3.5/3.6 use mrope

    @property
    def num_full_attn_layers(self) -> int:
        return self.num_hidden_layers // self.full_attention_interval

    @property
    def num_linear_attn_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_attn_layers

    def get_layer_types(self) -> List[str]:
        """Return layer type for each layer index."""
        types = []
        for i in range(self.num_hidden_layers):
            if (i + 1) % self.full_attention_interval == 0:
                types.append("full")
            else:
                types.append("linear")
        return types

    def get_kv_cache_size_per_token(self) -> int:
        """KV cache bytes per token (BF16) for full attention layers only."""
        # 2 (K+V) * full_attn_layers * kv_heads * head_dim * 2 bytes
        return (2 * self.num_full_attn_layers * self.num_key_value_heads *
                self.head_dim * 2)

    def get_ssm_state_size(self) -> int:
        """Linear attention SSM state size in bytes (FP32)."""
        # linear_layers * key_heads * key_dim * val_dim * 4 bytes
        return (self.num_linear_attn_layers * self.linear_num_key_heads *
                self.linear_key_head_dim * self.linear_value_head_dim * 4)

    def get_conv_state_size(self) -> int:
        """Linear attention conv state size in bytes (FP32)."""
        return (self.num_linear_attn_layers * self.linear_num_key_heads *
                self.linear_conv_kernel_dim * self.linear_key_head_dim * 4)

    def get_ssm_state_shape(self) -> Tuple[int, int, int]:
        """SSM state shape: (linear_layers, key_heads, key_dim, val_dim)."""
        return (self.num_linear_attn_layers, self.linear_num_key_heads,
                self.linear_key_head_dim, self.linear_value_head_dim)

    def get_conv_state_shape(self) -> Tuple[int, int, int]:
        """Conv state shape: (linear_layers, key_heads, conv_kernel, key_dim)."""
        return (self.num_linear_attn_layers, self.linear_num_key_heads,
                self.linear_conv_kernel_dim, self.linear_key_head_dim)

    def estimate_weight_size_int4(self) -> float:
        """Estimate INT4 weight size in GB."""
        # Rough: hidden^2 * layers * 7 (Q/K/V/O/gate/up/down) * 0.5 bytes/param / 1e9
        params = (self.hidden_size ** 2 * self.num_hidden_layers * 7 * 0.5 / 1e9)
        return params


# ─── Pre-defined Model Specs ────────────────────────────────────────

QWEN3_5_4B_SPEC = ModelSpec(
    name="Qwen3.5-4B",
    hidden_size=2560,
    num_hidden_layers=32,
    full_attention_interval=4,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=32,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)

QWEN3_5_9B_SPEC = ModelSpec(
    name="Qwen3.5-9B",
    hidden_size=4096,
    num_hidden_layers=32,
    full_attention_interval=4,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=32,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)

QWEN3_6_27B_SPEC = ModelSpec(
    name="Qwen3.6-27B",
    hidden_size=5120,
    num_hidden_layers=64,
    full_attention_interval=4,
    num_attention_heads=24,
    num_key_value_heads=4,
    head_dim=256,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)

MODEL_SPECS: Dict[str, ModelSpec] = {
    "Qwen3.5-4B": QWEN3_5_4B_SPEC,
    "Qwen3.5-9B": QWEN3_5_9B_SPEC,
    "Qwen3.6-27B": QWEN3_6_27B_SPEC,
}


# ─── Hardware / Transfer Configuration ──────────────────────────────

@dataclass
class TransferConfig:
    """Configuration for state transfer between GPUs."""
    backend: str = "auto"                      # auto | nvlink | pcie_p2p | shm | rdma | intra_gpu
    quantize_kv: bool = True                   # Enable INT4 KV quantization for PCIe backends
    layer_pipeline: bool = True                # Layer-level async pipeline
    pipeline_depth: int = 2                    # CUDA stream pipeline depth
    first_token_seeding: bool = True           # Prefill sends first token to decode
    n1_truncation: bool = True                 # N-1 truncation for recurrent state boundary


@dataclass
class CacheConfig:
    """Configuration for dual-state memory management."""
    block_size: int = 16                       # Tokens per PagedAttention block
    gpu_memory_utilization: float = 0.90       # Fraction of GPU memory to use
    max_num_seqs: int = 256                    # Maximum concurrent sequences
    max_model_len: int = 131072                # Maximum context length
    enable_prefix_cache: bool = True           # Radix tree prefix caching
    enable_kv_quant: bool = False              # INT4 KV quantization on decode GPU
    swap_space: int = 4                        # GB of CPU swap space


@dataclass
class SchedulerConfig:
    """Configuration for prefill/decode schedulers."""
    max_num_batched_tokens: int = 8192         # Max tokens per prefill batch
    max_num_seqs_per_batch: int = 64           # Max sequences per decode batch
    chunked_prefill_size: int = 4096           # Tokens per prefill chunk
    prefill_priority: str = "FCFS"             # FCFS | shortest-first


@dataclass
class RouterConfig:
    """Configuration for adaptive routing."""
    prompt_short_threshold: int = 2048         # Below this: collocated
    prompt_long_threshold: int = 8192          # Above this: PD always
    decode_load_threshold: float = 0.8         # Decode utilization threshold
    entropy_threshold: float = 2.0             # Logits entropy for output length prediction
    enable_entropy_predictor: bool = True


@dataclass
class HydraServeConfig:
    """Master configuration for HydraServe."""
    # Model
    model_name: str = "Qwen3.5-9B"
    model_path: str = "/models/Qwen3.5-9B-AWQ"
    precision: str = "int4"                    # int4 | bf16
    trust_remote_code: bool = True

    # Serving mode
    mode: ServingMode = ServingMode.PD_DISAGGREGATED
    prefill_gpu: int = 0
    decode_gpu: int = 1

    # Sub-configs
    transfer: TransferConfig = field(default_factory=TransferConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)

    # API server
    host: str = "0.0.0.0"
    port: int = 8000
    api_workers: int = 1

    @property
    def model_spec(self) -> ModelSpec:
        return MODEL_SPECS[self.model_name]

    @classmethod
    def from_json(cls, path: str) -> "HydraServeConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
