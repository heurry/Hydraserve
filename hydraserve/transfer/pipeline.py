"""
Transfer Pipeline: Layer-Level Async State Transfer.

Core innovation (§5.3.4): Don't wait for all prefill layers to complete
before starting transfer. Instead, transfer each layer's state as soon
as that layer's computation finishes.

    Prefill GPU (compute stream):
      Layer 0 → Layer 1 → ... → Layer 31 → done
         |         |                 |
         v         v                 v
    Transfer stream:
      [L0]     [L1]     ...     [L31]

    Decode GPU (receive stream):
      [L0]     [L1]     ...     [all ready → decode]

This pipeline hides transfer latency completely when per-layer state
is small relative to layer computation time.

Transfer time analysis (NVLink 112 GB/s):
  - Linear layer state: ~1 MB → <0.01ms (<< 2ms layer compute)
  - Full attn layer KV (BF16): 128 MB → 1.1ms (<< 5ms layer compute)
  - All 32K state (BF16): 1 GB → 9ms (<< 50-100ms prefill)
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import threading
from collections import defaultdict
import queue

from hydraserve.transfer.backend import TransferBackend
from hydraserve.transfer.descriptor import (
    StateTransferDescriptor, RegionDescriptor, RegionType
)
from hydraserve.cache.kv_quantizer import KVQuantizer


class TransferPipeline:
    """
    Layer-level asynchronous transfer pipeline.

    Manages CUDA streams for overlapped compute and transfer:
    - compute_stream: model forward on prefill GPU
    - transfer_stream: P2P send from prefill GPU
    - receive_stream: P2P receive on decode GPU

    The pipeline fires transfer for each layer's state as soon as
    the layer forward completes, without waiting for the full model.
    """

    def __init__(self, backend: TransferBackend, use_quantization: bool = True):
        self.backend = backend
        self.use_quantization = use_quantization
        self.quantizer = KVQuantizer() if use_quantization else None

        # CUDA streams for overlapped compute/transfer
        self.compute_stream: Optional[torch.cuda.Stream] = None
        self.transfer_streams: List[torch.cuda.Stream] = []
        self.receive_stream: Optional[torch.cuda.Stream] = None

        # State accumulation
        self._pending_transfers: Dict[int, List[Tuple[torch.Tensor, str, int]]] = defaultdict(list)
        self._received_states: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        self._transfer_events: Dict[int, torch.cuda.Event] = {}

        # Pipeline depth
        self.pipeline_depth = 2 if backend.supports_layer_pipeline() else 1

    def init_streams(self, src_gpu: int, dst_gpu: int) -> None:
        """Initialize CUDA streams for pipeline."""
        with torch.cuda.device(src_gpu):
            self.compute_stream = torch.cuda.Stream()
            self.transfer_streams = [
                torch.cuda.Stream() for _ in range(self.pipeline_depth)
            ]

        with torch.cuda.device(dst_gpu):
            self.receive_stream = torch.cuda.Stream()

    def send_layer_state(
        self,
        layer_idx: int,
        state: torch.Tensor,
        region_type: RegionType,
        request_id: int = 0,
        stream_idx: int = 0,
    ) -> None:
        """
        Send a single layer's state to the decode GPU.

        Called immediately after each layer's forward pass completes.
        Uses a dedicated transfer stream to overlap with next layer's compute.

        Args:
            layer_idx: The model layer index
            state: The state tensor to transfer
            region_type: Type of state (KV, SSM, conv)
            request_id: Request this state belongs to
            stream_idx: Which transfer stream to use (for pipeline depth)
        """
        if self.transfer_streams is None:
            self.init_streams(self.backend.src_gpu, self.backend.dst_gpu)

        stream_idx = stream_idx % self.pipeline_depth
        transfer_stream = self.transfer_streams[stream_idx]

        # If KV quantization is enabled and this is a KV region,
        # quantize before transfer
        if (self.use_quantization and self.quantizer is not None and
                region_type == RegionType.FULL_ATTN_KV and
                self.backend.transfer_mode.value != "full"):
            # Quantize to reduce transfer size
            if state.dim() >= 3:  # [2, n_kv_heads, n_tokens, head_dim] or similar
                k, v = state[0], state[1]
                k_q, k_s, k_z = self.quantizer.quantize_k(k)
                v_q, v_s, v_z = self.quantizer.quantize_v(v)
                # Pack quantization metadata with the tensor
                state_to_send = torch.cat([
                    k_q.flatten(), k_s.flatten(), k_z.flatten(),
                    v_q.flatten(), v_s.flatten(), v_z.flatten(),
                ])
            else:
                state_to_send = state
        else:
            state_to_send = state

        # Record compute completion event
        compute_event = torch.cuda.Event()
        compute_event.record(self.compute_stream)

        # Wait for compute to finish, then transfer
        with torch.cuda.stream(transfer_stream):
            transfer_stream.wait_event(compute_event)
            self.backend.send(state_to_send.contiguous(), self.backend.dst_gpu)

            # Record transfer completion event
            xfer_event = torch.cuda.Event()
            xfer_event.record()

        self._pending_transfers[request_id].append(
            (torch.empty_like(state_to_send), str(region_type), layer_idx)
        )
        self._transfer_events[(request_id, layer_idx)] = xfer_event

    def receive_layer_state(
        self,
        layer_idx: int,
        buffer: torch.Tensor,
        region_type: RegionType,
        request_id: int = 0,
    ) -> None:
        """
        Receive a single layer's state on the decode GPU.

        Args:
            layer_idx: The model layer index
            buffer: Pre-allocated receive buffer
            region_type: Type of state being received
            request_id: Request this state belongs to
        """
        if self.receive_stream is None:
            self.init_streams(self.backend.src_gpu, self.backend.dst_gpu)

        with torch.cuda.stream(self.receive_stream):
            self.backend.receive(buffer, self.backend.src_gpu)

        self._received_states[request_id][f"{region_type}_{layer_idx}"] = buffer

    def synchronize_transfers(self, request_id: int) -> None:
        """Wait for all pending transfers for a request to complete."""
        for (rid, lid), event in self._transfer_events.items():
            if rid == request_id:
                event.synchronize()

    def transfer_full_descriptor(
        self,
        descriptor: StateTransferDescriptor,
        state_tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer all state regions described in a descriptor.

        This is the high-level API used by the prefill engine.
        It handles the full dual-state transfer with optional quantization.

        Args:
            descriptor: StateTransferDescriptor for this request
            state_tensors: Dict mapping region identifiers to tensors

        Returns:
            Dict of received tensors on the decode GPU
        """
        received = {}

        for region in descriptor.regions:
            key = f"{region.region_type.value}_{region.layer_indices[0]}"
            tensor = state_tensors.get(key)
            if tensor is None:
                continue

            # Ensure tensor is on source GPU
            if tensor.device.index != self.backend.src_gpu:
                tensor = tensor.to(f'cuda:{self.backend.src_gpu}')

            # Allocate receive buffer on destination GPU
            if self.backend.dst_gpu != self.backend.src_gpu:
                recv_buf = torch.empty_like(tensor, device=f'cuda:{self.backend.dst_gpu}')
            else:
                recv_buf = tensor  # Intra-GPU: no copy

            # Transfer
            self.backend.send(tensor.contiguous(), self.backend.dst_gpu)

            if self.backend.dst_gpu != self.backend.src_gpu:
                self.backend.receive(recv_buf, self.backend.src_gpu)

            received[key] = recv_buf

        self.backend.synchronize()
        return received

    def estimate_transfer_time(self, descriptor: StateTransferDescriptor) -> float:
        """Estimate total transfer time for a descriptor at current bandwidth."""
        bw = self.backend.get_bandwidth()
        return descriptor.estimate_transfer_time_ms(bw)

    def cleanup(self, request_id: int) -> None:
        """Clean up transfer resources for a completed request."""
        self._pending_transfers.pop(request_id, None)
        self._received_states.pop(request_id, None)
        keys_to_remove = [k for k in self._transfer_events if k[0] == request_id]
        for k in keys_to_remove:
            del self._transfer_events[k]
