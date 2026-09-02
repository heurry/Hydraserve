"""Independent Qwen3.5/3.6 text runtime.

This module does not construct or invoke a Transformers model. It directly
applies checkpoint tensors using HydraServe kernels and basic GEMMs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hydraserve.config import LayerKind, ModelConfig, load_model_config
from hydraserve.kernels.activation import gdn_gating as triton_gdn_gating
from hydraserve.kernels.activation import silu_and_mul as triton_silu_and_mul
from hydraserve.kernels.awq import awq_linear
from hydraserve.kernels.fp8 import fp8_linear
from hydraserve.kernels.gdn import causal_depthwise_conv as triton_causal_conv
from hydraserve.kernels.gdn import gated_delta_recurrent
from hydraserve.kernels.gdn import legacy_gdn_kernels_enabled
from hydraserve.kernels.paged_attention import paged_attention
from hydraserve.kernels.paged_attention import paged_attention_splitk
from hydraserve.kernels.reference import (
    apply_text_rope,
    causal_depthwise_conv,
    causal_gqa_attention,
    gated_delta_rule,
    gated_rms_norm,
    rms_norm as reference_rms_norm,
    silu,
)
from hydraserve.kernels.rmsnorm import gated_rms_norm as triton_gated_rms_norm
from hydraserve.kernels.rmsnorm import rms_norm as triton_rms_norm
from hydraserve.model.weights import (
    BlockScaledFP8Weight,
    LANGUAGE_PREFIX,
    PackedInt4Weight,
    ShardedSafeTensorLoader,
    layer_prefix,
)


@dataclass(slots=True)
class RuntimeState:
    sequence_length: int = 0
    recurrent: dict[int, Any] = field(default_factory=dict)
    convolution: dict[int, Any] = field(default_factory=dict)
    keys: dict[int, Any] = field(default_factory=dict)
    values: dict[int, Any] = field(default_factory=dict)
    _state_pool_ref: Any = field(default=None, repr=False, compare=False)

    def clone(self) -> "RuntimeState":
        def copied(values):
            return {key: value.clone() for key, value in values.items()}

        return RuntimeState(
            sequence_length=self.sequence_length,
            recurrent=copied(self.recurrent),
            convolution=copied(self.convolution),
            keys=copied(self.keys),
            values=copied(self.values),
        )


@dataclass(frozen=True, slots=True)
class _MLPWeightSet:
    gate_up: Any | None
    gate: Any
    up: Any
    down: Any


@dataclass(frozen=True, slots=True)
class _FullAttentionWeightSet:
    qkv: Any | None
    query: Any
    key: Any
    value: Any
    query_norm: Any
    key_norm: Any
    output: Any


@dataclass(frozen=True, slots=True)
class _LinearAttentionWeightSet:
    qkvz: Any | None
    qkv: Any
    gate: Any
    ba: Any | None
    beta: Any
    step: Any
    a_log: Any
    dt_bias: Any
    convolution: Any
    norm: Any
    output: Any


@dataclass(frozen=True, slots=True)
class _LayerWeightSet:
    input_norm: Any
    post_attention_norm: Any
    mlp: _MLPWeightSet
    full_attention: _FullAttentionWeightSet | None
    linear_attention: _LinearAttentionWeightSet | None


class QwenTextRuntime:
    def __init__(
        self,
        config: ModelConfig,
        weights: Mapping[str, Any],
        *,
        use_triton: bool = True,
        use_flash_attention: bool = True,
        fuse_projections: bool = True,
        use_torch_compile: bool | None = None,
        device: str | Any | None = None,
        _take_weights: bool = False,
    ) -> None:
        self.config = config
        self.weights = (
            weights if _take_weights and isinstance(weights, dict) else dict(weights)
        )
        self.use_triton = use_triton
        self.use_flash_attention = use_flash_attention
        self._runtime_device = device
        self._decode_graphs: dict = {}
        self._decode_graph_failed: dict = {}
        self._decode_graph_observations: dict = {}
        self._decode_position_buffers: dict = {}
        self._fp32_logits = os.environ.get("HYDRASERVE_FP32_LOGITS", "0") == "1"
        self._torch_compile_enabled = (
            os.environ.get("HYDRASERVE_TORCH_COMPILE", "0") != "0"
            if use_torch_compile is None
            else bool(use_torch_compile)
        )
        self._compiled_forward = None
        self._compiled_decode_batch_transaction = None
        self._compiled_mlp = None
        self._validate_weight_shapes()
        if fuse_projections:
            self._prepare_fused_projection_weights()
        self._prepare_runtime_weight_cache()
        if self._torch_compile_enabled:
            self._prepare_torch_compile()

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        device: str | Any = "cuda:0",
        dtype: Any = None,
        use_triton: bool = True,
        use_flash_attention: bool = True,
        fuse_projections: bool = True,
        use_torch_compile: bool | None = None,
        requested_cache_tokens: int | None = None,
    ) -> "QwenTextRuntime":
        import torch

        config = load_model_config(model_dir)
        if requested_cache_tokens is not None and requested_cache_tokens <= 0:
            raise ValueError("requested cache tokens must be positive")
        loader = ShardedSafeTensorLoader(model_dir)
        dtype = dtype or torch.bfloat16
        names = loader.keys(f"{LANGUAGE_PREFIX}.")
        if "lm_head.weight" in loader:
            names += ("lm_head.weight",)
        packed_names = tuple(name for name in names if name.endswith(".weight_packed"))
        fp8_names = tuple(
            name
            for name in names
            if name.endswith(".weight") and f"{name}_scale_inv" in loader
        )
        packed_parts = (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape")
        cpu_weight_names = set()
        if packed_names or fp8_names:
            cpu_weight_names.add(f"{LANGUAGE_PREFIX}.embed_tokens.weight")
        if fp8_names and torch.device(device).type == "cuda" and "lm_head.weight" in names:
            free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
            element_size = dtype.itemsize
            estimated_sizes = {}
            for candidate in names:
                if candidate.endswith(packed_parts):
                    continue
                elements = 1
                for dimension in loader.tensor_shape(candidate):
                    elements *= dimension
                if candidate in fp8_names:
                    bytes_per_element = 1
                elif candidate.endswith((".A_log", ".dt_bias")):
                    bytes_per_element = 4
                else:
                    bytes_per_element = element_size
                estimated_sizes[candidate] = elements * bytes_per_element
            estimated_gpu_bytes = sum(
                size
                for candidate, size in estimated_sizes.items()
                if candidate not in cpu_weight_names
            )
            # Preserve space for the recurrent-state pool, KV pages, CUDA
            # libraries and decode activations on memory-bound consumer GPUs.
            state_transaction_bytes = config.decode_state_transaction_bytes
            state_reserve_bytes = max(
                state_transaction_bytes * 2,
                state_transaction_bytes + 512 * 1024**2,
            ) + 64 * 1024**2
            reserve_bytes = max(1024**3, state_reserve_bytes)
            if requested_cache_tokens is not None:
                reserve_bytes = max(
                    reserve_bytes,
                    requested_cache_tokens * config.kv_bytes_per_token_bf16
                    + max(1024**3, state_reserve_bytes),
                )
            if estimated_gpu_bytes + reserve_bytes > free_bytes:
                cpu_weight_names.add("lm_head.weight")
                estimated_gpu_bytes -= estimated_sizes["lm_head.weight"]
            if estimated_gpu_bytes + reserve_bytes > free_bytes:
                fp8_sizes = {
                    candidate: estimated_sizes[candidate]
                    + estimated_sizes[f"{candidate}_scale_inv"]
                    for candidate in fp8_names
                }
                for candidate in sorted(
                    fp8_names, key=fp8_sizes.__getitem__, reverse=True
                ):
                    cpu_weight_names.add(candidate)
                    estimated_gpu_bytes -= fp8_sizes[candidate]
                    if estimated_gpu_bytes + reserve_bytes <= free_bytes:
                        break
            if estimated_gpu_bytes + reserve_bytes > free_bytes:
                raise MemoryError(
                    "checkpoint cannot preserve the minimum CUDA execution reserve"
                )
        weights: dict[str, Any] = {}
        for name in names:
            if name.endswith(packed_parts) or name.endswith(".weight_scale_inv"):
                continue
            if name in fp8_names:
                weight_device = "cpu" if name in cpu_weight_names else device
                weights[name] = BlockScaledFP8Weight(
                    data=loader.tensor(name, device=weight_device).contiguous(),
                    scale_inv=loader.tensor(
                        f"{name}_scale_inv", device=weight_device, dtype=dtype
                    ).contiguous(),
                    original_shape=loader.tensor_shape(name),
                )
                continue
            weight_device = (
                "cpu" if name in cpu_weight_names else device
            )
            weights[name] = loader.tensor(
                name,
                device=weight_device,
                dtype=torch.float32 if name.endswith((".A_log", ".dt_bias")) else dtype,
            )
        for packed_name in packed_names:
            base_name = packed_name.removesuffix("_packed")
            shape = tuple(
                int(value)
                for value in loader.tensor(f"{base_name}_shape", device="cpu").tolist()
            )
            weights[base_name] = PackedInt4Weight(
                packed=loader.tensor(packed_name, device=device).contiguous(),
                scale=loader.tensor(
                    f"{base_name}_scale", device=device, dtype=dtype
                ).contiguous(),
                zero_point=loader.tensor(
                    f"{base_name}_zero_point", device=device
                ).contiguous(),
                original_shape=shape,
            )
        return cls(
            config,
            weights,
            use_triton=use_triton,
            use_flash_attention=use_flash_attention,
            fuse_projections=fuse_projections,
            use_torch_compile=use_torch_compile,
            device=device,
            _take_weights=True,
        )

    @property
    def device(self):
        if self._runtime_device is not None:
            import torch

            return torch.device(self._runtime_device)
        return self._embedding_weight.device

    @property
    def dtype(self):
        return self._embedding_weight.dtype

    @property
    def input_device(self):
        """Device on which token ids should be created before embedding lookup."""
        return self._embedding_weight.device

    def forward(
        self,
        input_ids,
        state: RuntimeState | None = None,
        *,
        paged_cache=None,
        request_id: int | None = None,
        compute_logits: bool = True,
        last_position_only: bool = False,
    ):
        target = self._compiled_forward or self._forward_transaction
        if self._compiled_forward is not None:
            self._mark_torch_compile_step()
        return target(
            input_ids,
            state,
            paged_cache=paged_cache,
            request_id=request_id,
            compute_logits=compute_logits,
            last_position_only=last_position_only,
        )

    def _forward_transaction(
        self,
        input_ids,
        state: RuntimeState | None = None,
        *,
        paged_cache=None,
        request_id: int | None = None,
        compute_logits: bool = True,
        last_position_only: bool = False,
    ):
        import torch

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, tokens] tensor")
        state = state or RuntimeState()
        batch, sequence = input_ids.shape
        start = state.sequence_length
        positions = torch.arange(start, start + sequence, device=self.device)
        embedding = self._embedding_weight
        if input_ids.device != embedding.device:
            input_ids = input_ids.to(embedding.device)
        hidden = self._embedding(input_ids, embedding)

        for layer_index, layer_kind in enumerate(self.config.layer_types):
            residual = hidden
            layer_weights = self._layer_weights[layer_index]
            hidden = self._norm(hidden, layer_weights.input_norm)
            if layer_kind is LayerKind.LINEAR_ATTENTION:
                hidden = self._linear_attention(layer_index, hidden, state)
            else:
                hidden = self._full_attention(
                    layer_index,
                    hidden,
                    positions,
                    state,
                    paged_cache=paged_cache,
                    request_id=request_id,
                )
            hidden = residual + hidden
            residual = hidden
            hidden = self._norm(hidden, layer_weights.post_attention_norm)
            hidden = self._mlp(layer_index, hidden)
            hidden = residual + hidden

        hidden = self._norm(hidden, self._final_norm_weight)
        state.sequence_length += sequence
        if not compute_logits:
            return None, state
        if last_position_only:
            hidden = hidden[:, -1:]
        logits = self._finalize_logits(self._linear(hidden, self._output_weight()))
        return logits, state

    def prefill(
        self,
        input_ids,
        *,
        chunk_size: int,
        paged_cache=None,
        request_id: int | None = None,
        chunk_callback=None,
    ):
        """Process one prompt in state-carrying chunks and return final-chunk logits."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, tokens] tensor")
        state = RuntimeState()
        logits = None
        total = input_ids.shape[1]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            is_last = end >= total
            logits, state = self.forward(
                input_ids[:, start:end],
                state,
                paged_cache=paged_cache,
                request_id=request_id,
                compute_logits=is_last,
                last_position_only=True,
            )
            if chunk_callback is not None:
                chunk_callback(start, end, state)
        return logits, state

    def decode_batch(
        self,
        input_ids,
        states: list[RuntimeState],
        paged_cache,
        request_ids,
        *,
        use_cuda_graphs: bool = True,
    ):
        """Advance heterogeneous requests by one token in a shared decode batch."""
        from contextlib import nullcontext

        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_batch requires [batch, 1] token ids")
        batch = input_ids.shape[0]
        if len(states) != batch or len(request_ids) != batch:
            raise ValueError("states/request_ids must match the decode batch")
        state_pool = self._shared_state_pool(states)
        context = (
            nullcontext(None)
            if state_pool is None
            else state_pool.batch(request_ids, states)
        )
        with context as pooled_batch:
            if (
                pooled_batch is not None
                and use_cuda_graphs
                and self._use_cuda_graphs()
            ):
                return self._decode_batch_graph(
                    input_ids, states, paged_cache, request_ids, pooled_batch
                )
            return self._run_decode_batch_transaction(
                input_ids, states, paged_cache, request_ids, pooled_batch
            )

    def _run_decode_batch_transaction(self, *args, **kwargs):
        target = (
            self._compiled_decode_batch_transaction
            or self._decode_batch_transaction
        )
        if self._compiled_decode_batch_transaction is not None:
            self._mark_torch_compile_step()
        return target(*args, **kwargs)

    def _decode_batch_transaction(
        self,
        input_ids,
        states,
        paged_cache,
        request_ids,
        pooled_batch,
        *,
        static: dict | None = None,
        paged_metadata_override=None,
        advance_sequence: bool = True,
    ):
        import torch

        batch = input_ids.shape[0]
        if static is None:
            positions = self._decode_positions(states)
        else:
            positions = static["positions"]
        embedding = self._embedding_weight
        if static is None and input_ids.device != embedding.device:
            input_ids = input_ids.to(embedding.device)
        elif static is not None:
            input_ids = static["input_ids"]
        hidden = self._embedding(input_ids, embedding)
        combined = RuntimeState()
        if static is None:
            paged_metadata = (
                (
                    paged_metadata_override
                    if paged_metadata_override is not None
                    else paged_cache.batch_metadata(request_ids)
                )
                if self.config.num_full_attention_layers
                else None
            )
        else:
            paged_metadata = (
                (static["table"], static["lengths"])
                if self.config.num_full_attention_layers
                else None
            )
        logical_positions = tuple(state.sequence_length for state in states)

        for layer_index, layer_kind in enumerate(self.config.layer_types):
            layer_weights = self._layer_weights[layer_index]
            residual = hidden
            hidden = self._norm(hidden, layer_weights.input_norm)
            if layer_kind is LayerKind.LINEAR_ATTENTION:
                if any(layer_index not in state.recurrent for state in states):
                    raise RuntimeError(f"request lacks recurrent state for layer {layer_index}")
                convolution_output = None
                if pooled_batch is None:
                    combined.recurrent[layer_index] = torch.cat(
                        [state.recurrent[layer_index] for state in states], dim=0
                    ).contiguous()
                    combined.convolution[layer_index] = torch.cat(
                        [state.convolution[layer_index] for state in states], dim=0
                    ).contiguous()
                else:
                    recurrent, convolution, convolution_output = pooled_batch.layer(
                        layer_index
                    )
                    combined.recurrent[layer_index] = recurrent
                    combined.convolution[layer_index] = convolution
                hidden = self._linear_attention(
                    layer_index,
                    hidden,
                    combined,
                    convolution_output=convolution_output,
                )
            else:
                hidden = self._full_attention_batch_decode(
                    layer_index,
                    hidden,
                    positions,
                    paged_cache,
                    request_ids,
                    paged_metadata,
                    logical_positions,
                )
            hidden = residual + hidden
            residual = hidden
            hidden = self._norm(hidden, layer_weights.post_attention_norm)
            hidden = residual + self._mlp(layer_index, hidden)

        hidden = self._norm(hidden, self._final_norm_weight)
        # The output projection is part of the transaction: do not publish state
        # if it fails after all transformer layers have run.
        logits = self._finalize_logits(self._linear(hidden, self._output_weight()))
        if static is not None:
            static["logits"].copy_(logits)
        if pooled_batch is None:
            for layer_index, recurrent in combined.recurrent.items():
                convolution = combined.convolution[layer_index]
                for row, state in enumerate(states):
                    self._commit_state_tensor(
                        state.recurrent, layer_index, recurrent[row : row + 1]
                    )
                    self._commit_state_tensor(
                        state.convolution, layer_index, convolution[row : row + 1]
                    )
        else:
            for layer_index, recurrent in combined.recurrent.items():
                pooled_batch.set_layer_result(
                    layer_index, recurrent, combined.convolution[layer_index]
                )
            pooled_batch.commit()
        if advance_sequence:
            for state in states:
                state.sequence_length += 1
        if static is not None:
            return static["logits"], states
        return logits, states

    def _prepare_torch_compile(self) -> None:
        """Compile the two runtime transactions while preserving eager opt-out.

        Dynamo is intentionally configured per callable instead of decorating the
        class: checkpoint loading and cache/control-plane bookkeeping must remain
        ordinary Python. ``fullgraph`` is available as a strict validation mode;
        the production default permits graph breaks around mutable request/cache
        objects while still compiling the tensor-heavy layer segments.
        """
        import torch

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            raise RuntimeError("HYDRASERVE_TORCH_COMPILE requires torch.compile")
        backend = os.environ.get("HYDRASERVE_TORCH_COMPILE_BACKEND", "inductor")
        fullgraph = os.environ.get("HYDRASERVE_TORCH_COMPILE_FULLGRAPH", "0") != "0"
        dynamic = os.environ.get("HYDRASERVE_TORCH_COMPILE_DYNAMIC", "1") != "0"
        options = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        if backend not in {"eager", "aot_eager"}:
            options["mode"] = os.environ.get(
                "HYDRASERVE_TORCH_COMPILE_MODE", "default"
            )
        scope = os.environ.get(
            "HYDRASERVE_TORCH_COMPILE_SCOPE",
            "tensor" if self.use_triton else "transactions",
        )
        if scope == "transactions":
            self._compiled_forward = compile_fn(self._forward_transaction, **options)
            self._compiled_decode_batch_transaction = compile_fn(
                self._decode_batch_transaction, **options
            )
        elif scope == "tensor":
            # Handwritten Triton kernels are already fused and should not be
            # re-traced as Dynamo higher-order ops. Compile the stable dense MLP
            # tensor subgraph used by both prefill and decode instead. This
            # fuses SiLU-and-mul with the surrounding projection graph while
            # keeping request/cache mutation in ordinary Python.
            options["fullgraph"] = True
            self._compiled_mlp = compile_fn(self._torch_mlp_dense, **options)
        else:
            raise ValueError(
                "HYDRASERVE_TORCH_COMPILE_SCOPE must be 'tensor' or 'transactions'"
            )

    @staticmethod
    def _mark_torch_compile_step() -> None:
        """Separate inference transactions for Inductor's internal CUDA Graphs."""
        import torch

        compiler = getattr(torch, "compiler", None)
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)
        if marker is not None:
            marker()

    def _use_cuda_graphs(self) -> bool:
        import os

        import torch

        return (
            self._compiled_decode_batch_transaction is None
            and os.environ.get("HYDRASERVE_CUDA_GRAPH", "1") != "0"
            and torch.cuda.is_available()
        )

    def _decode_batch_graph(
        self, input_ids, states, paged_cache, request_ids, pooled_batch
    ):
        """Replay a captured decode step for this batch shape.

        Host-side inputs (token ids, positions, block table, lengths) are
        copied into static buffers before each replay; the captured region
        covers embedding through logits including the pooled state commit.
        """
        import torch

        batch = input_ids.shape[0]
        table, lengths = paged_cache.batch_metadata(
            request_ids, bucket_width=True
        )
        key = (batch, table.shape[1])
        entry = self._decode_graphs.get(key)
        if entry is None and not self._decode_graph_failed.get(key, False):
            observations = self._decode_graph_observations.get(key, 0) + 1
            self._decode_graph_observations[key] = observations
            if observations >= self._cuda_graph_capture_after():
                entry = self._capture_decode_graph(
                    key,
                    input_ids,
                    states,
                    paged_cache,
                    request_ids,
                    pooled_batch,
                    table,
                    lengths,
                )
                if entry is not None:
                    self._decode_graphs[key] = entry
                    self._decode_graph_observations.pop(key, None)
                else:
                    self._decode_graph_failed[key] = True
        if entry is None:
            return self._decode_batch_transaction(
                input_ids,
                states,
                paged_cache,
                request_ids,
                pooled_batch,
                paged_metadata_override=(table, lengths),
            )
        embedding = self._embedding_weight
        ids = (
            input_ids
            if input_ids.device == embedding.device
            else input_ids.to(embedding.device)
        )
        entry["input_ids"].copy_(ids)
        self._decode_positions(states, target=entry["positions"])
        entry["table"].copy_(table)
        entry["lengths"].copy_(lengths)
        entry["slot_ids"].copy_(pooled_batch.slot_ids)
        entry["graph"].replay()
        for state in states:
            state.sequence_length += 1
        return entry["logits"], states

    def _capture_decode_graph(
        self,
        key,
        input_ids,
        states,
        paged_cache,
        request_ids,
        pooled_batch,
        table=None,
        lengths=None,
    ) -> dict | None:
        import torch

        batch, width = key
        embedding = self._embedding_weight
        static = {
            "input_ids": torch.empty(
                (batch, 1), device=embedding.device, dtype=torch.long
            ),
            "positions": torch.empty(
                (batch, 1), device=self.device, dtype=torch.long
            ),
            "table": torch.empty(
                (batch, width), device=self.device, dtype=torch.int32
            ),
            "lengths": torch.empty((batch,), device=self.device, dtype=torch.int32),
            "slot_ids": torch.empty((batch,), device=self.device, dtype=torch.long),
            "logits": torch.empty(
                (batch, 1, self.config.vocab_size),
                device=self.device,
                dtype=torch.float32 if self._fp32_logits else self.dtype,
            ),
        }
        # The warmup and capture passes execute the transaction for real and
        # would corrupt live state; snapshot the touched pool slots and KV
        # pages and restore them afterwards.
        block_size = paged_cache.block_manager.block_size
        slot_snapshot = []
        for state in states:
            recurrent = {
                index: state.recurrent[index].clone()
                for index in self.config.linear_layer_indices
            }
            convolution = {
                index: state.convolution[index].clone()
                for index in self.config.linear_layer_indices
            }
            slot_snapshot.append((state, recurrent, convolution))
        kv_snapshot = []
        for state, request_id in zip(states, request_ids):
            allocation = paged_cache.block_manager.get(request_id)
            block = allocation.block_ids[
                state.sequence_length // block_size
            ]
            for layer_index in self.config.full_attention_layer_indices:
                for pages in paged_cache.raw_layer_cache(layer_index):
                    kv_snapshot.append((pages, block, pages[block].clone()))
        # The batch workspace is gathered once at context entry; the warmup
        # passes advance it in place, so it must be restored before the replay.
        workspace_snapshot = (
            pooled_batch.recurrent.clone(),
            pooled_batch.convolution.clone(),
            pooled_batch.next_convolution.clone(),
        )

        def restore():
            for state, recurrent, convolution in slot_snapshot:
                for index in self.config.linear_layer_indices:
                    state.recurrent[index].copy_(recurrent[index])
                    state.convolution[index].copy_(convolution[index])
            for pages, block, snapshot in kv_snapshot:
                pages[block].copy_(snapshot)
            pooled_batch.recurrent.copy_(workspace_snapshot[0])
            pooled_batch.convolution.copy_(workspace_snapshot[1])
            pooled_batch.next_convolution.copy_(workspace_snapshot[2])

        # Fill the static inputs with the real current values so the warmup
        # passes execute a valid step (correct positions and block tables).
        ids = (
            input_ids
            if input_ids.device == embedding.device
            else input_ids.to(embedding.device)
        )
        static["input_ids"].copy_(ids)
        self._decode_positions(states, target=static["positions"])
        if table is None or lengths is None:
            table, lengths = paged_cache.batch_metadata(request_ids)
        static["table"].copy_(table)
        static["lengths"].copy_(lengths)
        # The pooled-state commit scatters into per-request slots; those slot
        # ids vary per batch, so they must live in a static buffer the replay
        # rewrites (the fresh tensor would otherwise be captured by address and
        # read as stale memory on later replays).
        static["slot_ids"].copy_(pooled_batch.slot_ids)
        pooled_batch.slot_ids = static["slot_ids"]
        try:
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self._cuda_graph_warmup_steps()):
                    self._decode_batch_transaction(
                        input_ids,
                        states,
                        paged_cache,
                        request_ids,
                        pooled_batch,
                        static=static,
                        advance_sequence=False,
                    )
            torch.cuda.current_stream().wait_stream(stream)
            # D-side state installation runs on another host thread/stream.
            # Thread-local capture mode permits unrelated CUDA work from that
            # thread; the decode worker additionally stays eager while a
            # prepare is already known to be active.
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self._decode_batch_transaction(
                    input_ids,
                    states,
                    paged_cache,
                    request_ids,
                    pooled_batch,
                    static=static,
                    advance_sequence=False,
                )
        except Exception:
            # Any host sync or dynamic shape inside the capture region makes
            # this batch shape permanently eager. Release the failed graph's
            # private memory pool so a capture failure does not pin several GB.
            try:
                graph.reset()
            except Exception:
                pass
            torch.cuda.empty_cache()
            restore()
            return None
        restore()
        static["graph"] = graph
        return static

    @staticmethod
    def _positive_env_int(name: str, default: int) -> int:
        import os

        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def _decode_positions(self, states, *, target=None):
        """Stage heterogeneous sequence positions without per-step allocation."""
        import torch

        batch = len(states)
        buffers = self._decode_position_buffers.get(batch)
        if buffers is None:
            host = torch.empty((batch, 1), device="cpu", dtype=torch.long)
            device = torch.empty((batch, 1), device=self.device, dtype=torch.long)
            buffers = (host, host.numpy(), device)
            self._decode_position_buffers[batch] = buffers
        host, host_array, device = buffers
        for row, state in enumerate(states):
            host_array[row, 0] = state.sequence_length
        destination = device if target is None else target
        destination.copy_(host)
        return destination

    def _cuda_graph_capture_after(self) -> int:
        # Capturing costs several complete decode passes. Waiting for repeated
        # observations avoids making one-off dynamic shapes slower than eager.
        return self._positive_env_int("HYDRASERVE_CUDA_GRAPH_CAPTURE_AFTER", 16)

    def _cuda_graph_warmup_steps(self) -> int:
        return self._positive_env_int("HYDRASERVE_CUDA_GRAPH_WARMUP_STEPS", 1)

    @staticmethod
    def _shared_state_pool(states):
        references = [getattr(state, "_state_pool_ref", None) for state in states]
        if not any(references):
            return None
        if not all(references):
            raise RuntimeError("decode batch mixes pooled and standalone recurrent states")
        owners = [reference() for reference in references]
        owner = owners[0]
        if owner is None or any(candidate is not owner for candidate in owners):
            raise RuntimeError("decode batch spans incompatible recurrent-state pools")
        return owner

    @staticmethod
    def _commit_state_tensor(mapping, layer_index: int, value) -> None:
        current = mapping.get(layer_index)
        if (
            current is not None
            and current.shape == value.shape
            and current.device == value.device
            and current.dtype == value.dtype
        ):
            current.copy_(value)
        else:
            mapping[layer_index] = value.clone()

    def _full_attention_batch_decode(
        self,
        layer_index: int,
        hidden,
        positions,
        paged_cache,
        request_ids,
        paged_metadata,
        logical_positions,
    ):
        import torch

        if not hidden.is_cuda:
            raise ValueError("paged batched decode requires CUDA")
        config = self.config
        weights = self._layer_weights[layer_index].full_attention
        if weights is None:
            raise RuntimeError("full-attention layer is missing cached weights")
        batch = hidden.shape[0]
        projected, key, value = self._full_attention_projections(hidden, weights)
        projected = projected.reshape(batch, 1, config.num_attention_heads, config.head_dim * 2)
        query, output_gate = projected.chunk(2, dim=-1)
        key = key.reshape(
            batch, 1, config.num_kv_heads, config.head_dim
        )
        value = value.reshape_as(key)
        query = self._norm(query, weights.query_norm)
        key = self._norm(key, weights.key_norm)
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        query = apply_text_rope(query, positions, config.rope_theta, rotary_dim)
        key = apply_text_rope(key, positions, config.rope_theta, rotary_dim)
        table, lengths = paged_metadata
        paged_cache.write_decode_batch(
            request_ids,
            layer_index,
            positions[:, 0],
            key[:, 0],
            value[:, 0],
            table,
            logical_positions=logical_positions,
        )
        key_pages, value_pages = paged_cache.layer_cache(layer_index)
        # Decode attention kernel selection:
        #   HYDRASERVE_PAGED_ATTENTION=flash     -> flash_attn_with_kvcache
        #   HYDRASERVE_PAGED_ATTENTION=reference -> original sequential scan
        #   default                              -> FlashDecoding-style split-K
        import os

        paged_attn_mode = os.environ.get("HYDRASERVE_PAGED_ATTENTION", "splitk")
        if paged_attn_mode == "flash" and self.use_flash_attention:
            from hydraserve.kernels.flash_prefill import paged_flash_prefill

            # query is [batch, 1, heads, head_dim]; kernel keeps the T axis.
            attn = paged_flash_prefill(
                query, key_pages, value_pages, table, lengths, causal=False
            )  # [batch, 1, heads, head_dim]
            attention = attn[:, 0]  # [batch, heads, head_dim]
        elif paged_attn_mode == "reference" or config.head_dim < 16:
            attention = paged_attention(query[:, 0], key_pages, value_pages, table, lengths)
        else:
            attention = paged_attention_splitk(query[:, 0], key_pages, value_pages, table, lengths)
        attention = attention[:, None]
        attention = attention.reshape(batch, 1, -1) * torch.sigmoid(
            output_gate.reshape(batch, 1, -1)
        )
        return self._linear(attention, weights.output)

    def _linear_attention(
        self,
        layer_index: int,
        hidden,
        state: RuntimeState,
        *,
        convolution_output=None,
    ):
        import torch

        config = self.config
        weights = self._layer_weights[layer_index].linear_attention
        if weights is None:
            raise RuntimeError("linear-attention layer is missing cached weights")
        mixed, gate, beta, step = self._linear_attention_projections(hidden, weights)
        if hidden.is_cuda and self.use_triton:
            beta, decay = triton_gdn_gating(
                beta.contiguous(),
                step.contiguous(),
                weights.a_log,
                weights.dt_bias,
            )
        else:
            beta = torch.sigmoid(beta)
            decay = -weights.a_log.float().exp() * torch.nn.functional.softplus(
                step.float() + weights.dt_bias.float()
            )
        conv_weight = weights.convolution.reshape(config.linear_conv_width, -1)
        conv_state = state.convolution.get(layer_index)
        if conv_state is None:
            conv_state = torch.zeros(
                hidden.shape[0],
                config.linear_conv_width,
                config.linear_conv_kernel_dim,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        if hidden.is_cuda and self.use_triton:
            mixed, conv_state = triton_causal_conv(
                mixed.contiguous(),
                conv_weight,
                conv_state,
                next_state=convolution_output,
                split_widths=(
                    config.linear_key_width,
                    config.linear_key_width,
                    config.linear_value_width,
                ),
            )
        else:
            mixed, conv_state = causal_depthwise_conv(mixed, conv_weight, conv_state)
        state.convolution[layer_index] = conv_state
        if isinstance(mixed, tuple):
            query, key, value = mixed
        else:
            query, key, value = torch.split(
                mixed,
                (
                    config.linear_key_width,
                    config.linear_key_width,
                    config.linear_value_width,
                ),
                dim=-1,
            )
        batch, sequence, _ = query.shape
        query = query.reshape(
            batch, sequence, config.linear_num_key_heads, config.linear_key_head_dim
        )
        key = key.reshape_as(query)
        value = value.reshape(
            batch, sequence, config.linear_num_value_heads, config.linear_value_head_dim
        )
        ratio = config.linear_num_value_heads // config.linear_num_key_heads
        if ratio * config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError("GDN value heads must be divisible by key heads")
        if hidden.is_cuda and self.use_triton:
            if legacy_gdn_kernels_enabled():
                query = query.repeat_interleave(ratio, dim=2).contiguous()
                key = key.repeat_interleave(ratio, dim=2).contiguous()
            else:
                query = query.contiguous()
                key = key.contiguous()
        else:
            query = query.repeat_interleave(ratio, dim=2).contiguous()
            key = key.repeat_interleave(ratio, dim=2).contiguous()
        initial = state.recurrent.get(layer_index)
        if hidden.is_cuda and self.use_triton:
            if initial is None:
                initial = torch.zeros(
                    batch,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    device=hidden.device,
                    dtype=torch.float32,
                )
            core, recurrent = gated_delta_recurrent(
                query,
                key,
                value.contiguous(),
                decay,
                beta,
                initial,
            )
        else:
            core, recurrent = gated_delta_rule(query, key, value, decay, beta, initial)
        state.recurrent[layer_index] = recurrent
        gate = gate.reshape(
            batch, sequence, config.linear_num_value_heads, config.linear_value_head_dim
        )
        if hidden.is_cuda and self.use_triton:
            core = triton_gated_rms_norm(
                core, gate, weights.norm, config.rms_norm_eps
            )
        else:
            core = gated_rms_norm(
                core,
                gate,
                weights.norm,
                config.rms_norm_eps,
            )
        return self._linear(core.reshape(batch, sequence, -1), weights.output)

    def _full_attention(
        self,
        layer_index: int,
        hidden,
        positions,
        state: RuntimeState,
        *,
        paged_cache,
        request_id,
    ):
        import torch

        config = self.config
        weights = self._layer_weights[layer_index].full_attention
        if weights is None:
            raise RuntimeError("full-attention layer is missing cached weights")
        batch, sequence, _ = hidden.shape
        projected, key, value = self._full_attention_projections(hidden, weights)
        projected = projected.reshape(batch, sequence, config.num_attention_heads, config.head_dim * 2)
        query, output_gate = projected.chunk(2, dim=-1)
        key = key.reshape(
            batch, sequence, config.num_kv_heads, config.head_dim
        )
        value = value.reshape_as(key)
        query = self._norm(query, weights.query_norm)
        key = self._norm(key, weights.key_norm)
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        query = apply_text_rope(query, positions, config.rope_theta, rotary_dim)
        key = apply_text_rope(key, positions, config.rope_theta, rotary_dim)

        if paged_cache is not None:
            if request_id is None or batch != 1:
                raise ValueError("paged single-request runtime needs one request_id")
            paged_cache.write(request_id, layer_index, positions, key[0], value[0])

        old_key = state.keys.get(layer_index)
        old_value = state.values.get(layer_index)
        all_key = key if old_key is None else torch.cat((old_key, key), dim=1)
        all_value = value if old_value is None else torch.cat((old_value, value), dim=1)
        if paged_cache is None:
            state.keys[layer_index] = all_key
            state.values[layer_index] = all_value

        if (
            sequence > 1
            and hidden.is_cuda
            and self.use_flash_attention
            and state.sequence_length == 0
            and old_key is None
        ):
            from hydraserve.kernels.flash_prefill import flash_attention_varlen

            packed_query = query.reshape(batch * sequence, config.num_attention_heads, config.head_dim)
            packed_key = key.reshape(batch * sequence, config.num_kv_heads, config.head_dim)
            packed_value = value.reshape_as(packed_key)
            cu = torch.arange(
                0, (batch + 1) * sequence, sequence, device=hidden.device, dtype=torch.int32
            )
            attention = flash_attention_varlen(
                packed_query, packed_key, packed_value, cu, sequence
            ).reshape_as(query)
        elif sequence > 1 and paged_cache is not None:
            table, _ = paged_cache.batch_metadata(
                (request_id,),
                logical_lengths=(state.sequence_length + sequence,),
            )
            key_pages, value_pages = paged_cache.layer_cache(layer_index)
            if (
                hidden.is_cuda
                and self.use_flash_attention
                and os.environ.get("HYDRASERVE_PAGED_FLASH_PREFILL", "1") != "0"
            ):
                from hydraserve.kernels.flash_prefill import paged_flash_prefill

                attention = paged_flash_prefill(
                    query,
                    key_pages,
                    value_pages,
                    table,
                    state.sequence_length + sequence,
                )
            else:
                from hydraserve.kernels.paged_attention import paged_prefill_attention

                attention = paged_prefill_attention(
                    query,
                    key_pages,
                    value_pages,
                    table,
                    query_start=state.sequence_length,
                )
        elif sequence == 1 and paged_cache is not None:
            from hydraserve.kernels.paged_attention import paged_prefill_attention

            table, _ = paged_cache.batch_metadata(
                (request_id,), logical_lengths=(state.sequence_length + 1,)
            )
            key_pages, value_pages = paged_cache.layer_cache(layer_index)
            attention = paged_prefill_attention(
                query,
                key_pages,
                value_pages,
                table,
                query_start=state.sequence_length,
            )
        else:
            attention = causal_gqa_attention(
                query, all_key, all_value, query_start=state.sequence_length
            )
        attention = attention.reshape(batch, sequence, -1) * torch.sigmoid(
            output_gate.reshape(batch, sequence, -1)
        )
        return self._linear(attention, weights.output)

    def _mlp(self, layer_index: int, hidden):
        weights = self._layer_weights[layer_index].mlp
        if self._compiled_mlp is not None and weights.gate_up is not None:
            import torch

            if isinstance(weights.gate_up, torch.Tensor) and isinstance(
                weights.down, torch.Tensor
            ):
                return self._compiled_mlp(hidden, weights.gate_up, weights.down)
        if weights.gate_up is not None:
            gate, up = self._linear(hidden, weights.gate_up).chunk(2, dim=-1)
        else:
            gate = self._linear(hidden, weights.gate)
            up = self._linear(hidden, weights.up)
        if hidden.is_cuda and self.use_triton:
            activated = triton_silu_and_mul(gate.contiguous(), up.contiguous())
        else:
            activated = silu(gate) * up
        return self._linear(activated, weights.down)

    @staticmethod
    def _torch_mlp_dense(hidden, gate_up, down):
        """Pure tensor MLP subgraph compiled once and reused across layers."""
        import torch.nn.functional as functional

        gate, up = (hidden @ gate_up.transpose(0, 1)).chunk(2, dim=-1)
        return (functional.silu(gate) * up) @ down.transpose(0, 1)

    def _full_attention_projections(
        self, hidden, weights: _FullAttentionWeightSet
    ):
        config = self.config
        if weights.qkv is not None:
            widths = (
                config.num_attention_heads * config.head_dim * 2,
                config.num_kv_heads * config.head_dim,
                config.num_kv_heads * config.head_dim,
            )
            return self._linear(hidden, weights.qkv).split(widths, dim=-1)
        return (
            self._linear(hidden, weights.query),
            self._linear(hidden, weights.key),
            self._linear(hidden, weights.value),
        )

    def _linear_attention_projections(
        self, hidden, weights: _LinearAttentionWeightSet
    ):
        if weights.qkvz is not None:
            mixed, gate = self._linear(hidden, weights.qkvz).split(
                (self.config.linear_conv_width, self.config.linear_value_width),
                dim=-1,
            )
        else:
            mixed = self._linear(hidden, weights.qkv)
            gate = self._linear(hidden, weights.gate)
        if weights.ba is not None:
            beta, step = self._linear(hidden, weights.ba).chunk(2, dim=-1)
        else:
            beta = self._linear(hidden, weights.beta)
            step = self._linear(hidden, weights.step)
        return mixed, gate, beta, step

    def _norm(self, hidden, weight):
        if hidden.is_cuda and self.use_triton:
            return triton_rms_norm(
                hidden.contiguous(), weight, self.config.rms_norm_eps
            )
        return reference_rms_norm(hidden, weight, self.config.rms_norm_eps)

    @staticmethod
    def _linear(hidden, weight):
        if isinstance(weight, PackedInt4Weight):
            return awq_linear(hidden, weight)
        if isinstance(weight, BlockScaledFP8Weight):
            return fp8_linear(hidden, weight)
        if hidden.device != weight.device:
            hidden = hidden.to(weight.device)
        return hidden @ weight.transpose(0, 1)

    def _embedding(self, input_ids, embedding):
        hidden = embedding[input_ids]
        if hidden.device == self.device:
            return hidden
        return hidden.to(self.device)

    def _weight(self, name: str):
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"runtime weight is missing: {name}") from exc

    def _output_weight(self):
        return self._output_projection_weight

    def _finalize_logits(self, logits):
        return logits.float() if self._fp32_logits else logits

    def _prepare_runtime_weight_cache(self) -> None:
        """Bind immutable per-layer weights once for the inference hot path."""
        embedding_name = f"{LANGUAGE_PREFIX}.embed_tokens.weight"
        self._embedding_weight = self._weight(embedding_name)
        self._final_norm_weight = self._weight(f"{LANGUAGE_PREFIX}.norm.weight")
        self._output_projection_weight = self.weights.get(
            "lm_head.weight", self._embedding_weight
        )
        layers = []
        for layer_index, kind in enumerate(self.config.layer_types):
            prefix = layer_prefix(layer_index)
            mlp_prefix = f"{prefix}.mlp"
            mlp = _MLPWeightSet(
                gate_up=self.weights.get(f"{mlp_prefix}.gate_up_proj.weight"),
                gate=self._weight(f"{mlp_prefix}.gate_proj.weight"),
                up=self._weight(f"{mlp_prefix}.up_proj.weight"),
                down=self._weight(f"{mlp_prefix}.down_proj.weight"),
            )
            full_attention = None
            linear_attention = None
            if kind is LayerKind.FULL_ATTENTION:
                attention = f"{prefix}.self_attn"
                full_attention = _FullAttentionWeightSet(
                    qkv=self.weights.get(f"{attention}.qkv_proj.weight"),
                    query=self._weight(f"{attention}.q_proj.weight"),
                    key=self._weight(f"{attention}.k_proj.weight"),
                    value=self._weight(f"{attention}.v_proj.weight"),
                    query_norm=self._weight(f"{attention}.q_norm.weight"),
                    key_norm=self._weight(f"{attention}.k_norm.weight"),
                    output=self._weight(f"{attention}.o_proj.weight"),
                )
            else:
                attention = f"{prefix}.linear_attn"
                linear_attention = _LinearAttentionWeightSet(
                    qkvz=self.weights.get(f"{attention}.in_proj_qkvz.weight"),
                    qkv=self._weight(f"{attention}.in_proj_qkv.weight"),
                    gate=self._weight(f"{attention}.in_proj_z.weight"),
                    ba=self.weights.get(f"{attention}.in_proj_ba.weight"),
                    beta=self._weight(f"{attention}.in_proj_b.weight"),
                    step=self._weight(f"{attention}.in_proj_a.weight"),
                    a_log=self._weight(f"{attention}.A_log"),
                    dt_bias=self._weight(f"{attention}.dt_bias"),
                    convolution=self._weight(f"{attention}.conv1d.weight"),
                    norm=self._weight(f"{attention}.norm.weight"),
                    output=self._weight(f"{attention}.out_proj.weight"),
                )
            layers.append(
                _LayerWeightSet(
                    input_norm=self._weight(f"{prefix}.input_layernorm.weight"),
                    post_attention_norm=self._weight(
                        f"{prefix}.post_attention_layernorm.weight"
                    ),
                    mlp=mlp,
                    full_attention=full_attention,
                    linear_attention=linear_attention,
                )
            )
        self._layer_weights = tuple(layers)

    def _prepare_fused_projection_weights(self) -> None:
        """Coalesce compatible output projections without changing checkpoints."""
        for layer_index, kind in enumerate(self.config.layer_types):
            prefix = layer_prefix(layer_index)
            self._install_fused_linear(
                (
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.up_proj.weight",
                ),
                f"{prefix}.mlp.gate_up_proj.weight",
            )
            if kind is LayerKind.FULL_ATTENTION:
                attention = f"{prefix}.self_attn"
                self._install_fused_linear(
                    (
                        f"{attention}.q_proj.weight",
                        f"{attention}.k_proj.weight",
                        f"{attention}.v_proj.weight",
                    ),
                    f"{attention}.qkv_proj.weight",
                )
            else:
                attention = f"{prefix}.linear_attn"
                self._install_fused_linear(
                    (
                        f"{attention}.in_proj_qkv.weight",
                        f"{attention}.in_proj_z.weight",
                    ),
                    f"{attention}.in_proj_qkvz.weight",
                )
                self._install_fused_linear(
                    (
                        f"{attention}.in_proj_b.weight",
                        f"{attention}.in_proj_a.weight",
                    ),
                    f"{attention}.in_proj_ba.weight",
                )

    def _install_fused_linear(self, names: tuple[str, ...], fused_name: str) -> None:
        parts = tuple(self._weight(name) for name in names)
        fused = self._concatenate_linear_weights(parts)
        if fused is None:
            return
        self.weights[fused_name] = fused
        sizes = tuple(part.shape[0] for part in parts)
        for name, view in zip(
            names,
            self._split_linear_weight(fused, sizes),
            strict=True,
        ):
            # Keep the checkpoint names as lightweight views for diagnostics and
            # shape validation while releasing their independent storage.
            self.weights[name] = view

    @staticmethod
    def _concatenate_linear_weights(parts):
        import torch

        if not parts:
            return None
        if all(isinstance(part, torch.Tensor) for part in parts):
            first = parts[0]
            if not all(
                part.ndim == 2
                and part.shape[1] == first.shape[1]
                and part.device == first.device
                and part.dtype == first.dtype
                for part in parts
            ):
                return None
            return torch.cat(parts, dim=0).contiguous()
        if all(isinstance(part, BlockScaledFP8Weight) for part in parts):
            first = parts[0]
            block_n, _ = first.block_size
            if not all(
                part.original_shape[1] == first.original_shape[1]
                and part.block_size == first.block_size
                and part.original_shape[0] % block_n == 0
                and part.data.device == first.data.device
                and part.scale_inv.device == first.scale_inv.device
                and part.data.dtype == first.data.dtype
                and part.scale_inv.dtype == first.scale_inv.dtype
                for part in parts
            ):
                return None
            return BlockScaledFP8Weight(
                torch.cat(tuple(part.data for part in parts), dim=0).contiguous(),
                torch.cat(tuple(part.scale_inv for part in parts), dim=0).contiguous(),
                (
                    sum(part.original_shape[0] for part in parts),
                    first.original_shape[1],
                ),
                first.block_size,
            )
        if all(isinstance(part, PackedInt4Weight) for part in parts):
            first = parts[0]
            if not all(
                part.original_shape[1] == first.original_shape[1]
                and part.group_size == first.group_size
                and part.original_shape[0] % 8 == 0
                and part.packed.device == first.packed.device
                and part.scale.device == first.scale.device
                and part.zero_point.device == first.zero_point.device
                and part.packed.dtype == first.packed.dtype
                and part.scale.dtype == first.scale.dtype
                and part.zero_point.dtype == first.zero_point.dtype
                for part in parts
            ):
                return None
            return PackedInt4Weight(
                torch.cat(tuple(part.packed for part in parts), dim=0).contiguous(),
                torch.cat(tuple(part.scale for part in parts), dim=0).contiguous(),
                torch.cat(tuple(part.zero_point for part in parts), dim=0).contiguous(),
                (
                    sum(part.original_shape[0] for part in parts),
                    first.original_shape[1],
                ),
                first.group_size,
            )
        return None

    @staticmethod
    def _split_linear_weight(fused, sizes: tuple[int, ...]):
        import torch

        if isinstance(fused, torch.Tensor):
            return fused.split(sizes, dim=0)
        if isinstance(fused, BlockScaledFP8Weight):
            block_n, _ = fused.block_size
            result = []
            start = 0
            for size in sizes:
                result.append(
                    BlockScaledFP8Weight(
                        fused.data.narrow(0, start, size),
                        fused.scale_inv.narrow(0, start // block_n, size // block_n),
                        (size, fused.original_shape[1]),
                        fused.block_size,
                    )
                )
                start += size
            return tuple(result)
        if isinstance(fused, PackedInt4Weight):
            result = []
            start = 0
            for size in sizes:
                result.append(
                    PackedInt4Weight(
                        fused.packed.narrow(0, start, size),
                        fused.scale.narrow(0, start, size),
                        fused.zero_point.narrow(0, start // 8, size // 8),
                        (size, fused.original_shape[1]),
                        fused.group_size,
                    )
                )
                start += size
            return tuple(result)
        raise TypeError(f"unsupported fused linear weight: {type(fused)!r}")

    def _validate_weight_shapes(self) -> None:
        config = self.config
        required = {
            f"{LANGUAGE_PREFIX}.embed_tokens.weight": (config.vocab_size, config.hidden_size),
            f"{LANGUAGE_PREFIX}.norm.weight": (config.hidden_size,),
        }
        if "lm_head.weight" in self.weights:
            required["lm_head.weight"] = (config.vocab_size, config.hidden_size)
        for layer_index, kind in enumerate(config.layer_types):
            prefix = layer_prefix(layer_index)
            required.update(
                {
                    f"{prefix}.input_layernorm.weight": (config.hidden_size,),
                    f"{prefix}.post_attention_layernorm.weight": (config.hidden_size,),
                    f"{prefix}.mlp.gate_proj.weight": (config.intermediate_size, config.hidden_size),
                    f"{prefix}.mlp.up_proj.weight": (config.intermediate_size, config.hidden_size),
                    f"{prefix}.mlp.down_proj.weight": (config.hidden_size, config.intermediate_size),
                }
            )
            if kind is LayerKind.FULL_ATTENTION:
                attn = f"{prefix}.self_attn"
                required.update(
                    {
                        f"{attn}.q_proj.weight": (
                            config.num_attention_heads * config.head_dim * 2,
                            config.hidden_size,
                        ),
                        f"{attn}.k_proj.weight": (
                            config.num_kv_heads * config.head_dim,
                            config.hidden_size,
                        ),
                        f"{attn}.v_proj.weight": (
                            config.num_kv_heads * config.head_dim,
                            config.hidden_size,
                        ),
                        f"{attn}.o_proj.weight": (
                            config.hidden_size,
                            config.num_attention_heads * config.head_dim,
                        ),
                        f"{attn}.q_norm.weight": (config.head_dim,),
                        f"{attn}.k_norm.weight": (config.head_dim,),
                    }
                )
            else:
                attn = f"{prefix}.linear_attn"
                required.update(
                    {
                        f"{attn}.in_proj_qkv.weight": (config.linear_conv_width, config.hidden_size),
                        f"{attn}.in_proj_z.weight": (config.linear_value_width, config.hidden_size),
                        f"{attn}.in_proj_b.weight": (config.linear_num_value_heads, config.hidden_size),
                        f"{attn}.in_proj_a.weight": (config.linear_num_value_heads, config.hidden_size),
                        f"{attn}.conv1d.weight": (
                            config.linear_conv_width,
                            1,
                            config.linear_conv_kernel_dim,
                        ),
                        f"{attn}.A_log": (config.linear_num_value_heads,),
                        f"{attn}.dt_bias": (config.linear_num_value_heads,),
                        f"{attn}.norm.weight": (config.linear_value_head_dim,),
                        f"{attn}.out_proj.weight": (config.hidden_size, config.linear_value_width),
                    }
                )
        for name, shape in required.items():
            tensor = self._weight(name)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
