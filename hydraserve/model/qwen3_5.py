"""
Qwen3.5 model adapter implementation.

Handles the Gated Delta Network (GDN) hybrid attention architecture:
- 4 layers per group: 3 linear (GDN) + 1 full (standard) attention
- Supports Qwen3.5-4B and Qwen3.5-9B variants.
"""

import math
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydraserve.model.adapter import ModelAdapter
from hydraserve.config import ModelSpec, MODEL_SPECS


class Qwen3_5Adapter(ModelAdapter):
    """
    Adapter for Qwen3.5 series models (4B, 9B).

    Architecture: 32 layers total, every 4th layer is full attention,
    rest are GDN linear attention.
    """

    def __init__(self, model_path: str, device: torch.device, precision: str = "int4",
                 model_spec: Optional[ModelSpec] = None):
        super().__init__(model_path, device, precision)
        self._spec = model_spec
        self._config: Dict[str, Any] = {}
        self._lm_head: Optional[nn.Linear] = None
        self._norm: Optional[nn.Module] = None
        self._embed_tokens: Optional[nn.Embedding] = None
        self._layers: nn.ModuleList = nn.ModuleList()

    # ─── Lifecycle ──────────────────────────────────────────────

    def load_model(self) -> None:
        """Load Qwen3.5 model from HuggingFace or local path."""
        from transformers import AutoConfig, AutoModelForCausalLM

        self._config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to_dict()

        # Populate spec from config if not provided
        if self._spec is None:
            model_name = self._config.get("_name_or_path", "Qwen3.5-9B")
            if "4B" in model_name:
                self._spec = MODEL_SPECS["Qwen3.5-4B"]
            else:
                self._spec = MODEL_SPECS["Qwen3.5-9B"]

        # Load with appropriate precision
        torch_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": None,
        }

        if self.precision == "int4":
            from autoawq import AutoAWQForCausalLM
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path, fuse_layers=True, **{k: v for k, v in load_kwargs.items()
                                                       if k != "torch_dtype"}
            )
            self.model = self.model.model  # Get the underlying nn.Module
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Cache module references
        self._embed_tokens = self.model.model.embed_tokens
        self._layers = self.model.model.layers
        self._norm = self.model.model.norm
        self._lm_head = self.model.lm_head

        # Build layer type list
        self._layer_types = self._spec.get_layer_types()

    def get_config(self) -> Dict[str, Any]:
        return self._config

    # ─── Layer Information ──────────────────────────────────────

    def get_layer_types(self) -> List[str]:
        return self._layer_types

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) % self._spec.full_attention_interval == 0

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        return not self.is_full_attention_layer(layer_idx)

    # ─── State Shapes ───────────────────────────────────────────

    def get_ssm_state_shape(self) -> Tuple[int, int, int, int]:
        return (self._spec.num_linear_attn_layers,
                self._spec.linear_num_key_heads,
                self._spec.linear_key_head_dim,
                self._spec.linear_value_head_dim)

    def get_conv_state_shape(self) -> Tuple[int, int, int, int]:
        return (self._spec.num_linear_attn_layers,
                self._spec.linear_num_key_heads,
                self._spec.linear_conv_kernel_dim,
                self._spec.linear_key_head_dim)

    def get_ssm_state_dtype(self) -> torch.dtype:
        return torch.float32

    def get_kv_cache_shape(self, n_tokens: int) -> Tuple[int, ...]:
        """KV cache: (num_full_attn_layers, 2, batch, n_kv_heads, n_tokens, head_dim)."""
        return (self._spec.num_full_attn_layers, 2,
                None,  # batch (dynamic)
                self._spec.num_key_value_heads,
                n_tokens,
                self._spec.head_dim)

    # ─── Layer Counts ───────────────────────────────────────────

    def get_num_full_attn_layers(self) -> int:
        return self._spec.num_full_attn_layers

    def get_num_linear_attn_layers(self) -> int:
        return self._spec.num_linear_attn_layers

    def get_num_hidden_layers(self) -> int:
        return self._spec.num_hidden_layers

    def get_hidden_size(self) -> int:
        return self._spec.hidden_size

    def get_num_attention_heads(self) -> int:
        return self._spec.num_attention_heads

    def get_num_key_value_heads(self) -> int:
        return self._spec.num_key_value_heads

    def get_head_dim(self) -> int:
        return self._spec.head_dim

    def get_weight_size_gb(self) -> float:
        return self._spec.estimate_weight_size_int4()

    # ─── Forward ────────────────────────────────────────────────

    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Dict[int, torch.Tensor]] = None,
        ssm_state: Optional[Dict[int, torch.Tensor]] = None,
        conv_state: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Prefill forward pass using fused GDN kernel for linear layers
        and flash_attn for full attention layers.
        """
        batch_size, seq_len = input_ids.shape
        hidden_states = self._embed_tokens(input_ids)

        new_kv_cache = {}
        new_ssm_state = {}
        new_conv_state = {}
        linear_state_idx = 0  # Track which linear state slot we're at

        for layer_idx in range(self._spec.num_hidden_layers):
            layer = self._layers[layer_idx]
            residual = hidden_states

            if self.is_full_attention_layer(layer_idx):
                # Full attention: RMSNorm -> Q/K/V proj -> RoPE -> flash_attn -> O proj
                hidden_states = self._forward_full_attn_layer(
                    layer, hidden_states, positions, batch_size, seq_len,
                    kv_cache.get(layer_idx) if kv_cache else None,
                    new_kv_cache, layer_idx
                )
            else:
                # Linear attention (GDN): use fused Triton kernel
                hidden_states = self._forward_linear_attn_layer(
                    layer, hidden_states, positions, batch_size, seq_len,
                    ssm_state.get(linear_state_idx) if ssm_state else None,
                    conv_state.get(linear_state_idx) if conv_state else None,
                    new_ssm_state, new_conv_state, linear_state_idx
                )
                linear_state_idx += 1

            # FFN (SwiGLU): RMSNorm -> gate/up -> silu*gate*up -> down
            hidden_states = self._forward_ffn(layer, hidden_states, residual)

        # Final norm
        hidden_states = self._norm(hidden_states)

        # LM head
        logits = self._lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        return {
            "logits": logits,
            "kv_cache": new_kv_cache,
            "ssm_state": new_ssm_state,
            "conv_state": new_conv_state,
        }

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Dict[int, torch.Tensor],
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
        block_tables: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Decode forward pass (single token). Uses paged attention kernel
        for full attention layers.
        """
        batch_size = input_ids.shape[0]
        hidden_states = self._embed_tokens(input_ids)  # [batch, 1, hidden]
        linear_state_idx = 0

        for layer_idx in range(self._spec.num_hidden_layers):
            layer = self._layers[layer_idx]
            residual = hidden_states

            if self.is_full_attention_layer(layer_idx):
                hidden_states = self._forward_full_attn_decode(
                    layer, hidden_states, positions, kv_cache[layer_idx],
                    block_tables.get(layer_idx) if block_tables else None
                )
            else:
                hidden_states = self._forward_linear_attn_decode(
                    layer, hidden_states,
                    ssm_state[linear_state_idx],
                    conv_state[linear_state_idx]
                )
                linear_state_idx += 1

            hidden_states = self._forward_ffn(layer, hidden_states, residual)

        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)  # [batch, 1, vocab_size]

        return {
            "logits": logits,
            "kv_cache": kv_cache,
            "ssm_state": ssm_state,
            "conv_state": conv_state,
        }

    def forward_single_token(
        self,
        token_id: int,
        position: int,
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        """Single token forward for N-1 truncation replay."""
        input_ids = torch.tensor([[token_id]], device=self.device)
        positions = torch.tensor([[position]], device=self.device)
        # Empty KV cache - we only need to advance recurrent state
        kv_cache = {}
        return self.forward_decode(input_ids, positions, kv_cache, ssm_state, conv_state)

    # ─── Internal: Full Attention Layer ────────────────────────

    def _forward_full_attn_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        seq_len: int,
        input_kv_cache: Optional[torch.Tensor],
        new_kv_cache: Dict[int, torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Full attention prefill using flash_attn_varlen_func."""
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            raise ImportError(
                "flash-attn is required for prefill. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        n_heads = self._spec.num_attention_heads
        n_kv_heads = self._spec.num_key_value_heads
        head_dim = self._spec.head_dim

        # Input norm
        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        # Q/K/V projections
        q = layer.self_attn.q_proj(normed)     # [batch, seq, n_heads * head_dim]
        k = layer.self_attn.k_proj(normed)     # [batch, seq, n_kv_heads * head_dim]
        v = layer.self_attn.v_proj(normed)     # [batch, seq, n_kv_heads * head_dim]

        q = q.view(batch_size, seq_len, n_heads, head_dim)
        k = k.view(batch_size, seq_len, n_kv_heads, head_dim)
        v = v.view(batch_size, seq_len, n_kv_heads, head_dim)

        # Apply RoPE
        q, k = self._apply_rope(q, k, positions, head_dim)

        # flash_attn expects [total_tokens, n_heads, head_dim]
        q_flat = q.reshape(batch_size * seq_len, n_heads, head_dim)
        k_flat = k.reshape(batch_size * seq_len, n_kv_heads, head_dim)
        v_flat = v.reshape(batch_size * seq_len, n_kv_heads, head_dim)

        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len,
                                  dtype=torch.int32, device=hidden_states.device)
        max_seqlen = seq_len

        attn_out = flash_attn_varlen_func(
            q_flat, k_flat, v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )

        attn_out = attn_out.view(batch_size, seq_len, n_heads * head_dim)

        # O projection
        output = layer.self_attn.o_proj(attn_out)

        # Store KV cache for this layer
        new_kv_cache[layer_idx] = torch.stack([k, v], dim=1)  # [batch, 2, seq, n_kv_heads, head_dim]

        return residual + output

    def _forward_full_attn_decode(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Full attention decode using paged attention Triton kernel."""
        from hydraserve.kernels.paged_attention import paged_attention_decode

        n_heads = self._spec.num_attention_heads
        n_kv_heads = self._spec.num_key_value_heads
        head_dim = self._spec.head_dim
        batch_size = hidden_states.shape[0]

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        q = layer.self_attn.q_proj(normed)
        k_new = layer.self_attn.k_proj(normed)
        v_new = layer.self_attn.v_proj(normed)

        q = q.view(batch_size, n_heads, head_dim)
        k_new = k_new.view(batch_size, n_kv_heads, head_dim)
        v_new = v_new.view(batch_size, n_kv_heads, head_dim)

        # Apply RoPE to new token
        q, k_new = self._apply_rope_single(q, k_new, positions, head_dim)

        # Append new K/V to cache (kv_cache: [batch, 2, n_kv_heads, seq_len, head_dim])
        # For paged attention, we'd update the block table entry instead
        # Simplified: concatenate
        k_cache = kv_cache[:, 0]  # [batch, n_kv_heads, seq_len, head_dim]
        v_cache = kv_cache[:, 1]
        k_full = torch.cat([k_cache, k_new.unsqueeze(2)], dim=2)
        v_full = torch.cat([v_cache, v_new.unsqueeze(2)], dim=2)
        kv_cache[:, 0] = k_full
        kv_cache[:, 1] = v_full

        attn_out = paged_attention_decode(
            q, k_full, v_full, block_tables,
            self._spec.block_size if hasattr(self._spec, 'block_size') else 16,
            sm_scale=1.0 / math.sqrt(head_dim)
        )

        attn_out = attn_out.view(batch_size, n_heads * head_dim)
        output = layer.self_attn.o_proj(attn_out)
        return residual + output

    # ─── Internal: Linear Attention (GDN) Layer ─────────────────

    def _forward_linear_attn_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        seq_len: int,
        input_ssm_state: Optional[torch.Tensor],
        input_conv_state: Optional[torch.Tensor],
        new_ssm_state: Dict[int, torch.Tensor],
        new_conv_state: Dict[int, torch.Tensor],
        linear_state_idx: int,
    ) -> torch.Tensor:
        """Linear attention (GDN) prefill using fused Triton kernel."""
        from hydraserve.kernels.gdn_fused import gdn_fused_prefill

        n_key_heads = self._spec.linear_num_key_heads
        n_val_heads = self._spec.linear_num_value_heads
        key_dim = self._spec.linear_key_head_dim
        val_dim = self._spec.linear_value_head_dim
        conv_kernel = self._spec.linear_conv_kernel_dim

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        # Q/K/V projections for GDN
        # Q: [batch, seq, n_key_heads * key_dim]
        q = layer.self_attn.q_proj(normed)
        k = layer.self_attn.k_proj(normed)
        v = layer.self_attn.v_proj(normed)

        q = q.view(batch_size, seq_len, n_key_heads, key_dim)
        k = k.view(batch_size, seq_len, n_key_heads, key_dim)
        v = v.view(batch_size, seq_len, n_val_heads, val_dim)

        # GDN-specific: beta, alpha, gate
        beta = layer.self_attn.beta_proj(normed) if hasattr(layer.self_attn, 'beta_proj') else torch.ones(
            batch_size, seq_len, n_key_heads, device=hidden_states.device
        )
        alpha = layer.self_attn.alpha_proj(normed) if hasattr(layer.self_attn, 'alpha_proj') else torch.ones(
            batch_size, seq_len, n_key_heads, device=hidden_states.device
        )
        gate = layer.self_attn.gate_proj(normed) if hasattr(layer.self_attn, 'gate_proj') else torch.ones(
            batch_size, seq_len, n_key_heads, device=hidden_states.device
        )

        beta = beta.view(batch_size, seq_len, n_key_heads)
        alpha = alpha.view(batch_size, seq_len, n_key_heads)
        gate = gate.view(batch_size, seq_len, n_key_heads)

        # Fused GDN kernel: state stays in SRAM for entire prefill
        output, final_ssm_state = gdn_fused_prefill(
            k, v, q, beta, alpha, gate,
            input_ssm_state,
            seq_len=seq_len,
            n_heads=n_key_heads,
            key_dim=key_dim,
            val_dim=val_dim,
        )

        output = output.view(batch_size, seq_len, n_key_heads * key_dim)

        # O projection
        output = layer.self_attn.o_proj(output)

        new_ssm_state[linear_state_idx] = final_ssm_state
        # Conv state placeholder (GDN conv is typically fused)
        new_conv_state[linear_state_idx] = (
            input_conv_state if input_conv_state is not None
            else torch.zeros(batch_size, n_key_heads, conv_kernel, key_dim,
                            dtype=torch.float32, device=hidden_states.device)
        )

        return residual + output

    def _forward_linear_attn_decode(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Linear attention (GDN) decode - single token step using delta rule."""
        from hydraserve.kernels.gdn_fused import gdn_decode_step

        n_key_heads = self._spec.linear_num_key_heads
        n_val_heads = self._spec.linear_num_value_heads
        key_dim = self._spec.linear_key_head_dim
        val_dim = self._spec.linear_value_head_dim

        residual = hidden_states
        normed = layer.input_layernorm(hidden_states)

        batch_size = hidden_states.shape[0]

        q = layer.self_attn.q_proj(normed).view(batch_size, n_key_heads, key_dim)
        k = layer.self_attn.k_proj(normed).view(batch_size, n_key_heads, key_dim)
        v = layer.self_attn.v_proj(normed).view(batch_size, n_val_heads, val_dim)

        beta = getattr(layer.self_attn, 'beta_proj', lambda x: torch.ones(
            batch_size, 1, n_key_heads, device=hidden_states.device
        ))(normed).view(batch_size, n_key_heads)

        alpha = getattr(layer.self_attn, 'alpha_proj', lambda x: torch.ones(
            batch_size, 1, n_key_heads, device=hidden_states.device
        ))(normed).view(batch_size, n_key_heads)

        gate = getattr(layer.self_attn, 'gate_proj', lambda x: torch.ones(
            batch_size, 1, n_key_heads, device=hidden_states.device
        ))(normed).view(batch_size, n_key_heads)

        output, ssm_state = gdn_decode_step(
            k, v, q, beta, alpha, gate, ssm_state,
            n_heads=n_key_heads, key_dim=key_dim, val_dim=val_dim,
        )

        output = output.view(batch_size, n_key_heads * key_dim)
        output = layer.self_attn.o_proj(output)
        return residual + output

    # ─── Internal: FFN ──────────────────────────────────────────

    def _forward_ffn(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """SwiGLU FFN."""
        normed = layer.post_attention_layernorm(hidden_states)
        gate = layer.mlp.gate_proj(normed)
        up = layer.mlp.up_proj(normed)
        # SwiGLU: silu(gate) * up
        hidden = F.silu(gate) * up
        hidden = layer.mlp.down_proj(hidden)
        return hidden_states + hidden

    # ─── Internal: RoPE ─────────────────────────────────────────

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Position Embedding (mrope for Qwen3.5)."""
        # Qwen3.5 uses partial rotary (only first 25% of head dim)
        rotary_dim = int(head_dim * self._spec.partial_rotary_factor)

        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        cos, sin = self._compute_rope_freqs(positions, rotary_dim)

        q_rot = self._rotate_half(q_rot, cos, sin)
        k_rot = self._rotate_half(k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def _apply_rope_single(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE for single token decode (no seq_len dimension in q/k)."""
        rotary_dim = int(head_dim * self._spec.partial_rotary_factor)

        q_rot = q[:, :, :rotary_dim]  # [batch, n_heads, rotary_dim]
        q_pass = q[:, :, rotary_dim:]
        k_rot = k[:, :, :rotary_dim]  # [batch, n_kv_heads, rotary_dim]
        k_pass = k[:, :, rotary_dim:]

        cos, sin = self._compute_rope_freqs(positions.squeeze(-1), rotary_dim)

        # Expand for head dimensions
        cos = cos.unsqueeze(1)  # [batch, 1, rotary_dim]
        sin = sin.unsqueeze(1)

        q_rot = self._rotate_half(q_rot, cos, sin)
        k_rot = self._rotate_half(k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def _compute_rope_freqs(
        self, positions: torch.Tensor, dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin frequencies."""
        theta = self._spec.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                               device=positions.device) / dim))
        # positions: [batch, seq_len]
        freqs = torch.outer(positions.reshape(-1).float(), freqs)
        freqs = freqs.reshape(*positions.shape, -1)
        return freqs.cos().to(self.device), freqs.sin().to(self.device)

    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    # ─── Sampling ───────────────────────────────────────────────

    @torch.no_grad()
    def sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """Sample from logits with temperature, top-p, top-k filtering."""
        # logits: [batch, vocab_size]
        if temperature > 0:
            logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k,
                                torch.full_like(logits, float('-inf')),
                                logits)

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # Sample
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return tokens, entropy.mean().item()
