"""
Linear Attention State Pool.

Manages fixed-size FP32 slots for linear attention recurrent states.
Unlike KV cache blocks, these states:
- Are fixed size (don't grow with sequence length)
- Cannot be quantized (FP32 required for numerical stability)
- Must be transferred whole (no block-level granularity)

Slot layout:
    slot[seq_id] = {
        ssm_state: [num_linear_layers, num_key_heads, key_dim, val_dim] fp32,
        conv_state: [num_linear_layers, num_key_heads, conv_kernel, key_dim] fp32,
    }

Memory per slot:
    4B/9B: ~25 MB (24 linear layers)
    27B:   ~50 MB (48 linear layers)
"""

from typing import Dict, Optional, Tuple
import torch

from hydraserve.config import ModelSpec


class StateSlot:
    """A single state slot for one sequence's linear attention state."""

    __slots__ = ('slot_id', 'seq_id', 'ssm_state', 'conv_state', 'is_active')

    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.seq_id: Optional[int] = None
        self.ssm_state: Optional[torch.Tensor] = None  # [n_linear_layers, ...]
        self.conv_state: Optional[torch.Tensor] = None
        self.is_active = False


class StatePool:
    """
    Pool allocator for linear attention recurrent states.

    States are FP32 and fixed-size. The pool pre-allocates all slots
    on the decode GPU at initialization.
    """

    def __init__(self, model_spec: ModelSpec, max_sequences: int, device: torch.device):
        self.model_spec = model_spec
        self.max_sequences = max_sequences
        self.device = device

        # State dimensions
        self.num_linear_layers = model_spec.num_linear_attn_layers
        self.num_key_heads = model_spec.linear_num_key_heads
        self.key_dim = model_spec.linear_key_head_dim
        self.val_dim = model_spec.linear_value_head_dim
        self.conv_kernel = model_spec.linear_conv_kernel_dim

        # Per-slot memory
        ssm_shape = (self.num_linear_layers, self.num_key_heads, self.key_dim, self.val_dim)
        conv_shape = (self.num_linear_layers, self.num_key_heads, self.conv_kernel, self.key_dim)

        self.ssm_bytes_per_slot = (self.num_linear_layers * self.num_key_heads *
                                    self.key_dim * self.val_dim * 4)  # FP32 = 4 bytes
        self.conv_bytes_per_slot = (self.num_linear_layers * self.num_key_heads *
                                     self.conv_kernel * self.key_dim * 4)
        self.total_bytes_per_slot = self.ssm_bytes_per_slot + self.conv_bytes_per_slot

        # Pre-allocate all slots
        self.slots: Dict[int, StateSlot] = {}
        self.free_slots: list = list(range(max_sequences))
        self.seq_to_slot: Dict[int, int] = {}

        # Physical storage: one big buffer for efficiency
        self._ssm_buffer: Optional[torch.Tensor] = None
        self._conv_buffer: Optional[torch.Tensor] = None

    def init_buffers(self) -> None:
        """Pre-allocate state buffers on GPU."""
        if self._ssm_buffer is None:
            ssm_shape = (self.max_sequences, self.num_linear_layers,
                         self.num_key_heads, self.key_dim, self.val_dim)
            conv_shape = (self.max_sequences, self.num_linear_layers,
                          self.num_key_heads, self.conv_kernel, self.key_dim)

            self._ssm_buffer = torch.zeros(ssm_shape, dtype=torch.float32, device=self.device)
            self._conv_buffer = torch.zeros(conv_shape, dtype=torch.float32, device=self.device)

    def allocate(self, seq_id: int) -> int:
        """
        Allocate a state slot for a sequence.

        Args:
            seq_id: Unique sequence identifier

        Returns:
            slot_id: The allocated slot index

        Raises:
            RuntimeError: If no free slots available
        """
        if seq_id in self.seq_to_slot:
            return self.seq_to_slot[seq_id]

        if not self.free_slots:
            # Try to free inactive slots
            self._evict_stale()
            if not self.free_slots:
                raise RuntimeError(f"StatePool full: {self.max_sequences} slots exhausted")

        slot_id = self.free_slots.pop(0)
        slot = StateSlot(slot_id)
        slot.seq_id = seq_id
        slot.is_active = True
        self.slots[slot_id] = slot
        self.seq_to_slot[seq_id] = slot_id

        self.init_buffers()
        return slot_id

    def free(self, seq_id: int) -> None:
        """Free a sequence's state slot."""
        slot_id = self.seq_to_slot.pop(seq_id, None)
        if slot_id is None:
            return
        slot = self.slots.pop(slot_id, None)
        if slot:
            slot.is_active = False
            slot.seq_id = None
        self.free_slots.append(slot_id)

    def get_slot(self, seq_id: int) -> Optional[int]:
        """Get slot id for a sequence."""
        return self.seq_to_slot.get(seq_id)

    def write_state(
        self,
        slot_id: int,
        ssm_state: torch.Tensor,   # [n_linear_layers, n_key_heads, key_dim, val_dim]
        conv_state: Optional[torch.Tensor] = None,
        layer_indices: Optional[list] = None,
    ) -> None:
        """
        Write state tensors into a slot.

        Args:
            slot_id: Target slot
            ssm_state: SSM state tensor (FP32)
            conv_state: Conv state tensor (FP32), optional
            layer_indices: Which linear layer indices these correspond to (optional)
        """
        self.init_buffers()

        if layer_indices is None:
            # Assume all layers
            self._ssm_buffer[slot_id].copy_(ssm_state)
            if conv_state is not None:
                self._conv_buffer[slot_id].copy_(conv_state)
        else:
            # Write specific layers
            for i, layer_idx in enumerate(layer_indices):
                self._ssm_buffer[slot_id, layer_idx].copy_(ssm_state[i])
                if conv_state is not None:
                    self._conv_buffer[slot_id, layer_idx].copy_(conv_state[i])

    def read_state(
        self,
        slot_id: int,
        layer_indices: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read state tensors from a slot.

        Returns:
            (ssm_state, conv_state) tensors
        """
        self.init_buffers()
        if layer_indices is None:
            return self._ssm_buffer[slot_id].clone(), self._conv_buffer[slot_id].clone()
        ssm = self._ssm_buffer[slot_id, layer_indices].clone()
        conv = self._conv_buffer[slot_id, layer_indices].clone()
        return ssm, conv

    def get_active_count(self) -> int:
        return len(self.seq_to_slot)

    def get_free_count(self) -> int:
        return len(self.free_slots)

    # ─── Internal ───────────────────────────────────────────────

    def _evict_stale(self) -> None:
        """Evict stale/inactive slots. Simplified LRU policy."""
        stale = [sid for sid, slot in self.slots.items() if not slot.is_active]
        for sid in stale:
            self.free(sid)

    # ─── Statistics ─────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return {
            "total_slots": self.max_sequences,
            "active_slots": len(self.seq_to_slot),
            "free_slots": len(self.free_slots),
            "bytes_per_slot_mb": self.total_bytes_per_slot / 1e6,
            "total_allocated_gb": (self.max_sequences * self.total_bytes_per_slot) / 1e9,
            "ssm_mb": self.ssm_bytes_per_slot / 1e6,
            "conv_mb": self.conv_bytes_per_slot / 1e6,
        }
