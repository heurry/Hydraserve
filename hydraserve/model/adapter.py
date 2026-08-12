"""
ModelAdapter: Abstract interface for multi-model support.

Provides a unified API for the inference engine to interact with
different model architectures (Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-27B)
without hardcoding model-specific parameters.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """
    Abstract adapter for hybrid attention models.

    Each supported model provides a concrete implementation that wraps
    the HuggingFace model while exposing unified access to:
    - Layer types (linear vs full attention)
    - State shapes and dtypes
    - Forward pass with dual-state management
    """

    def __init__(self, model_path: str, device: torch.device, precision: str = "int4"):
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.model: Optional[nn.Module] = None
        self._layer_types: List[str] = []

    # ─── Lifecycle ──────────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and move to device."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration dict."""
        ...

    # ─── Layer Information ──────────────────────────────────────

    @abstractmethod
    def get_layer_types(self) -> List[str]:
        """
        Return layer type for each layer index.
        Returns list of "linear" or "full" strings.
        Example: ["linear", "linear", "linear", "full", ...]
        """
        ...

    @abstractmethod
    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses full (standard) attention."""
        ...

    @abstractmethod
    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses linear (GDN) attention."""
        ...

    # ─── State Shapes ───────────────────────────────────────────

    @abstractmethod
    def get_ssm_state_shape(self) -> Tuple[int, int, int, int]:
        """
        Return SSM state shape for ONE linear attention layer.
        Returns: (num_key_heads, key_head_dim, value_head_dim)
                 Actually (num_linear_attn_layers, num_key_heads, key_head_dim, num_value_heads)
        """
        ...

    @abstractmethod
    def get_conv_state_shape(self) -> Tuple[int, int, int, int]:
        """
        Return conv state shape for ONE linear attention layer.
        Returns: (num_key_heads, conv_kernel_dim, key_head_dim)
                 Actually (num_linear_attn_layers, num_key_heads, conv_kernel_dim, key_head_dim)
        """
        ...

    @abstractmethod
    def get_ssm_state_dtype(self) -> torch.dtype:
        """Return the dtype for SSM state (always float32)."""
        ...

    @abstractmethod
    def get_kv_cache_shape(self, n_tokens: int) -> Tuple[int, ...]:
        """Return KV cache shape for full attention layers, given n_tokens."""
        ...

    # ─── Layer Counts ───────────────────────────────────────────

    @abstractmethod
    def get_num_full_attn_layers(self) -> int:
        """Number of full (standard) attention layers."""
        ...

    @abstractmethod
    def get_num_linear_attn_layers(self) -> int:
        """Number of linear (GDN) attention layers."""
        ...

    @abstractmethod
    def get_num_hidden_layers(self) -> int:
        """Total number of transformer layers."""
        ...

    @abstractmethod
    def get_hidden_size(self) -> int:
        """Model hidden dimension."""
        ...

    @abstractmethod
    def get_num_attention_heads(self) -> int:
        """Number of query attention heads."""
        ...

    @abstractmethod
    def get_num_key_value_heads(self) -> int:
        """Number of KV attention heads (for GQA)."""
        ...

    @abstractmethod
    def get_head_dim(self) -> int:
        """Dimension per attention head."""
        ...

    # ─── Weight Info ────────────────────────────────────────────

    @abstractmethod
    def get_weight_size_gb(self) -> float:
        """Estimated model weight size in GB at current precision."""
        ...

    # ─── Forward ────────────────────────────────────────────────

    @abstractmethod
    def forward_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[Dict[int, torch.Tensor]] = None,
        ssm_state: Optional[Dict[int, torch.Tensor]] = None,
        conv_state: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Run prefill forward pass.

        Args:
            input_ids: [batch, seq_len] token ids
            positions: [batch, seq_len] position indices
            kv_cache: {layer_idx: [batch, n_kv_heads, seq_len, head_dim]} for full attn layers
            ssm_state: {layer_idx: [batch, n_key_heads, key_dim, val_dim]} for linear layers
            conv_state: {layer_idx: [batch, n_key_heads, kernel_dim, key_dim]} for linear layers

        Returns:
            Dict with:
                - logits: [batch, seq_len, vocab_size]
                - kv_cache: updated full attention KV cache
                - ssm_state: final SSM state (encode tokens 0..N-1)
                - conv_state: final conv state
                - first_token_id: sampled first token (optional)
        """
        ...

    @abstractmethod
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
        Run decode forward pass (single token per sequence).

        Args:
            input_ids: [batch, 1] token ids
            positions: [batch, 1] position indices
            kv_cache: {layer_idx: [batch, n_kv_heads, seq_len, head_dim]}
            ssm_state: {layer_idx: ...} current SSM state
            conv_state: {layer_idx: ...} current conv state
            block_tables: {layer_idx: [batch, max_blocks]} for paged attention

        Returns:
            Dict with logits, updated states.
        """
        ...

    # ─── Single Token Forward (for N-1 truncation) ──────────────

    @abstractmethod
    def forward_single_token(
        self,
        token_id: int,
        position: int,
        ssm_state: Dict[int, torch.Tensor],
        conv_state: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Run single token forward to advance recurrent state by one step.
        Used for N-1 truncation on the decode side.

        Args:
            token_id: single token id
            position: position index
            ssm_state: current SSM state (encoding 0..N-1)
            conv_state: current conv state

        Returns:
            Dict with updated ssm_state, conv_state, and logits for this token.
        """
        ...

    # ─── Sampling ───────────────────────────────────────────────

    @abstractmethod
    def sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        """
        Sample from logits, returning (token_ids, entropy).
        """
        ...
