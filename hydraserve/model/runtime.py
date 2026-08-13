"""Independent Qwen3.5/3.6 text runtime.

This module does not construct or invoke a Transformers model. It directly
applies checkpoint tensors using HydraServe kernels and basic GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hydraserve.config import LayerKind, ModelConfig, load_model_config
from hydraserve.kernels.reference import (
    apply_text_rope,
    causal_depthwise_conv,
    causal_gqa_attention,
    gated_delta_rule,
    gated_rms_norm,
    rms_norm as reference_rms_norm,
    silu,
)
from hydraserve.model.weights import LANGUAGE_PREFIX, ShardedSafeTensorLoader, layer_prefix


@dataclass(slots=True)
class RuntimeState:
    sequence_length: int = 0
    recurrent: dict[int, Any] = field(default_factory=dict)
    convolution: dict[int, Any] = field(default_factory=dict)
    keys: dict[int, Any] = field(default_factory=dict)
    values: dict[int, Any] = field(default_factory=dict)

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


class QwenTextRuntime:
    def __init__(
        self,
        config: ModelConfig,
        weights: Mapping[str, Any],
        *,
        use_triton: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        self.config = config
        self.weights = weights
        self.use_triton = use_triton
        self.use_flash_attention = use_flash_attention
        self._validate_weight_shapes()

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        device: str | Any = "cuda:0",
        dtype: Any = None,
        use_triton: bool = True,
        use_flash_attention: bool = True,
    ) -> "QwenTextRuntime":
        import torch

        config = load_model_config(model_dir)
        loader = ShardedSafeTensorLoader(model_dir)
        dtype = dtype or torch.bfloat16
        names = loader.keys(f"{LANGUAGE_PREFIX}.")
        weights = {
            name: loader.tensor(
                name,
                device=device,
                dtype=torch.float32 if name.endswith((".A_log", ".dt_bias")) else dtype,
            )
            for name in names
        }
        return cls(
            config,
            weights,
            use_triton=use_triton,
            use_flash_attention=use_flash_attention,
        )

    @property
    def device(self):
        return self._weight(f"{LANGUAGE_PREFIX}.embed_tokens.weight").device

    @property
    def dtype(self):
        return self._weight(f"{LANGUAGE_PREFIX}.embed_tokens.weight").dtype

    def forward(
        self,
        input_ids,
        state: RuntimeState | None = None,
        *,
        paged_cache=None,
        request_id: int | None = None,
    ):
        import torch

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, tokens] tensor")
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        state = state or RuntimeState()
        batch, sequence = input_ids.shape
        start = state.sequence_length
        positions = torch.arange(start, start + sequence, device=self.device)
        embedding = self._weight(f"{LANGUAGE_PREFIX}.embed_tokens.weight")
        hidden = embedding[input_ids]

        for layer_index, layer_kind in enumerate(self.config.layer_types):
            residual = hidden
            prefix = layer_prefix(layer_index)
            hidden = self._norm(hidden, self._weight(f"{prefix}.input_layernorm.weight"))
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
            hidden = self._norm(hidden, self._weight(f"{prefix}.post_attention_layernorm.weight"))
            hidden = self._mlp(layer_index, hidden)
            hidden = residual + hidden

        hidden = self._norm(hidden, self._weight(f"{LANGUAGE_PREFIX}.norm.weight"))
        state.sequence_length += sequence
        logits = hidden.float() @ embedding.float().transpose(0, 1)
        return logits, state

    def decode_batch(self, input_ids, states: list[RuntimeState], paged_cache, request_ids):
        """Advance heterogeneous requests by one token in a shared decode batch."""
        import torch

        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode_batch requires [batch, 1] token ids")
        batch = input_ids.shape[0]
        if len(states) != batch or len(request_ids) != batch:
            raise ValueError("states/request_ids must match the decode batch")
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        positions = torch.tensor(
            [[state.sequence_length] for state in states],
            device=self.device,
            dtype=torch.long,
        )
        embedding = self._weight(f"{LANGUAGE_PREFIX}.embed_tokens.weight")
        hidden = embedding[input_ids]
        combined = RuntimeState()

        for layer_index, layer_kind in enumerate(self.config.layer_types):
            prefix = layer_prefix(layer_index)
            residual = hidden
            hidden = self._norm(hidden, self._weight(f"{prefix}.input_layernorm.weight"))
            if layer_kind is LayerKind.LINEAR_ATTENTION:
                if any(layer_index not in state.recurrent for state in states):
                    raise RuntimeError(f"request lacks recurrent state for layer {layer_index}")
                combined.recurrent[layer_index] = torch.cat(
                    [state.recurrent[layer_index] for state in states], dim=0
                ).contiguous()
                combined.convolution[layer_index] = torch.cat(
                    [state.convolution[layer_index] for state in states], dim=0
                ).contiguous()
                hidden = self._linear_attention(layer_index, hidden, combined)
                for row, state in enumerate(states):
                    state.recurrent[layer_index] = combined.recurrent[layer_index][row : row + 1].clone()
                    state.convolution[layer_index] = combined.convolution[layer_index][row : row + 1].clone()
            else:
                hidden = self._full_attention_batch_decode(
                    layer_index, hidden, positions, paged_cache, request_ids
                )
            hidden = residual + hidden
            residual = hidden
            hidden = self._norm(hidden, self._weight(f"{prefix}.post_attention_layernorm.weight"))
            hidden = residual + self._mlp(layer_index, hidden)

        hidden = self._norm(hidden, self._weight(f"{LANGUAGE_PREFIX}.norm.weight"))
        for state in states:
            state.sequence_length += 1
        return hidden.float() @ embedding.float().transpose(0, 1), states

    def _full_attention_batch_decode(
        self, layer_index: int, hidden, positions, paged_cache, request_ids
    ):
        import torch

        if not hidden.is_cuda:
            raise ValueError("paged batched decode requires CUDA")
        config = self.config
        prefix = f"{layer_prefix(layer_index)}.self_attn"
        batch = hidden.shape[0]
        projected = self._linear(hidden, self._weight(f"{prefix}.q_proj.weight"))
        projected = projected.reshape(batch, 1, config.num_attention_heads, config.head_dim * 2)
        query, output_gate = projected.chunk(2, dim=-1)
        key = self._linear(hidden, self._weight(f"{prefix}.k_proj.weight")).reshape(
            batch, 1, config.num_kv_heads, config.head_dim
        )
        value = self._linear(hidden, self._weight(f"{prefix}.v_proj.weight")).reshape_as(key)
        query = self._norm(query, self._weight(f"{prefix}.q_norm.weight"))
        key = self._norm(key, self._weight(f"{prefix}.k_norm.weight"))
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        query = apply_text_rope(query, positions, config.rope_theta, rotary_dim)
        key = apply_text_rope(key, positions, config.rope_theta, rotary_dim)
        for row, request_id in enumerate(request_ids):
            paged_cache.write(
                request_id,
                layer_index,
                positions[row],
                key[row],
                value[row],
            )
        table, lengths = paged_cache.batch_metadata(request_ids)
        key_pages, value_pages = paged_cache.layer_cache(layer_index)
        from hydraserve.kernels.paged_attention import paged_attention

        attention = paged_attention(query[:, 0], key_pages, value_pages, table, lengths)[:, None]
        attention = attention.reshape(batch, 1, -1) * torch.sigmoid(
            output_gate.reshape(batch, 1, -1)
        )
        return self._linear(attention, self._weight(f"{prefix}.o_proj.weight"))

    def _linear_attention(self, layer_index: int, hidden, state: RuntimeState):
        import torch

        config = self.config
        prefix = f"{layer_prefix(layer_index)}.linear_attn"
        mixed = self._linear(hidden, self._weight(f"{prefix}.in_proj_qkv.weight"))
        gate = self._linear(hidden, self._weight(f"{prefix}.in_proj_z.weight"))
        beta = torch.sigmoid(self._linear(hidden, self._weight(f"{prefix}.in_proj_b.weight")))
        step = self._linear(hidden, self._weight(f"{prefix}.in_proj_a.weight"))
        conv_weight = self._weight(f"{prefix}.conv1d.weight").reshape(config.linear_conv_width, -1)
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
            from hydraserve.kernels.gdn import causal_depthwise_conv as triton_causal_conv

            mixed, conv_state = triton_causal_conv(
                mixed.contiguous(), conv_weight.contiguous(), conv_state.contiguous()
            )
        else:
            mixed, conv_state = causal_depthwise_conv(mixed, conv_weight, conv_state)
        state.convolution[layer_index] = conv_state
        query, key, value = torch.split(
            mixed,
            (config.linear_key_width, config.linear_key_width, config.linear_value_width),
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
        query = query.repeat_interleave(ratio, dim=2).contiguous()
        key = key.repeat_interleave(ratio, dim=2).contiguous()
        decay = -self._weight(f"{prefix}.A_log").float().exp() * torch.nn.functional.softplus(
            step.float() + self._weight(f"{prefix}.dt_bias").float()
        )
        initial = state.recurrent.get(layer_index)
        if hidden.is_cuda and self.use_triton:
            from hydraserve.kernels.gdn import gated_delta_recurrent

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
                query, key, value.contiguous(), decay.contiguous(), beta.float().contiguous(), initial
            )
        else:
            core, recurrent = gated_delta_rule(query, key, value, decay, beta, initial)
        state.recurrent[layer_index] = recurrent
        gate = gate.reshape(
            batch, sequence, config.linear_num_value_heads, config.linear_value_head_dim
        )
        if hidden.is_cuda and self.use_triton:
            from hydraserve.kernels.rmsnorm import gated_rms_norm as triton_gated_rms_norm

            core = triton_gated_rms_norm(
                core, gate, self._weight(f"{prefix}.norm.weight"), config.rms_norm_eps
            )
        else:
            core = gated_rms_norm(
                core,
                gate,
                self._weight(f"{prefix}.norm.weight"),
                config.rms_norm_eps,
            )
        return self._linear(core.reshape(batch, sequence, -1), self._weight(f"{prefix}.out_proj.weight"))

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
        prefix = f"{layer_prefix(layer_index)}.self_attn"
        batch, sequence, _ = hidden.shape
        projected = self._linear(hidden, self._weight(f"{prefix}.q_proj.weight"))
        projected = projected.reshape(batch, sequence, config.num_attention_heads, config.head_dim * 2)
        query, output_gate = projected.chunk(2, dim=-1)
        key = self._linear(hidden, self._weight(f"{prefix}.k_proj.weight")).reshape(
            batch, sequence, config.num_kv_heads, config.head_dim
        )
        value = self._linear(hidden, self._weight(f"{prefix}.v_proj.weight")).reshape_as(key)
        query = self._norm(query, self._weight(f"{prefix}.q_norm.weight"))
        key = self._norm(key, self._weight(f"{prefix}.k_norm.weight"))
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

        if sequence > 1 and hidden.is_cuda and self.use_flash_attention and old_key is None:
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
        elif sequence == 1 and paged_cache is not None and hidden.is_cuda:
            from hydraserve.kernels.paged_attention import paged_attention

            table, lengths = paged_cache.batch_metadata((request_id,))
            key_pages, value_pages = paged_cache.layer_cache(layer_index)
            attention = paged_attention(query[:, 0], key_pages, value_pages, table, lengths)[:, None]
        else:
            attention = causal_gqa_attention(
                query, all_key, all_value, query_start=state.sequence_length
            )
        attention = attention.reshape(batch, sequence, -1) * torch.sigmoid(
            output_gate.reshape(batch, sequence, -1)
        )
        return self._linear(attention, self._weight(f"{prefix}.o_proj.weight"))

    def _mlp(self, layer_index: int, hidden):
        prefix = f"{layer_prefix(layer_index)}.mlp"
        gate = silu(self._linear(hidden, self._weight(f"{prefix}.gate_proj.weight")))
        up = self._linear(hidden, self._weight(f"{prefix}.up_proj.weight"))
        return self._linear(gate * up, self._weight(f"{prefix}.down_proj.weight"))

    def _norm(self, hidden, weight):
        if hidden.is_cuda and self.use_triton:
            from hydraserve.kernels.rmsnorm import rms_norm

            return rms_norm(hidden.contiguous(), weight.contiguous(), self.config.rms_norm_eps)
        return reference_rms_norm(hidden, weight, self.config.rms_norm_eps)

    @staticmethod
    def _linear(hidden, weight):
        return hidden @ weight.transpose(0, 1)

    def _weight(self, name: str):
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(f"runtime weight is missing: {name}") from exc

    def _validate_weight_shapes(self) -> None:
        config = self.config
        required = {
            f"{LANGUAGE_PREFIX}.embed_tokens.weight": (config.vocab_size, config.hidden_size),
            f"{LANGUAGE_PREFIX}.norm.weight": (config.hidden_size,),
        }
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
