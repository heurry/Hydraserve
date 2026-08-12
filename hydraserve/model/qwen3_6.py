"""
Qwen3.6 model adapter implementation.

Qwen3.6-27B: 64 layers (48 linear GDN + 16 full attention),
larger hidden dim (5120), more attention heads (24), more value heads (48).
Inherits Qwen3_5Adapter since architecture is same, just scaled up.
"""

import torch
from typing import Optional

from hydraserve.model.qwen3_5 import Qwen3_5Adapter
from hydraserve.config import ModelSpec, MODEL_SPECS


class Qwen3_6Adapter(Qwen3_5Adapter):
    """
    Adapter for Qwen3.6-27B.

    Same GDN + GQA hybrid architecture as Qwen3.5, but:
    - 64 layers (48 linear, 16 full attention)
    - hidden_size=5120, 24 attention heads, 48 value heads
    - Double the KV/token (64KB vs 32KB) due to 2x full attn layers
    - Double the SSM state (48MB vs 24MB) due to 2x linear layers
    """

    def __init__(self, model_path: str, device: torch.device, precision: str = "int4",
                 model_spec: Optional[ModelSpec] = None):
        if model_spec is None:
            model_spec = MODEL_SPECS["Qwen3.6-27B"]
        super().__init__(model_path, device, precision, model_spec)

    # The Qwen3.6 architecture is identical to Qwen3.5 except for
    # layer counts and dimensions, which are handled by ModelSpec.
    # All the layer computation code is inherited from Qwen3_5Adapter.

    def get_weight_size_gb(self) -> float:
        """Qwen3.6-27B INT4: ~13.5 GB."""
        return self._spec.estimate_weight_size_int4()
