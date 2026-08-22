"""Direct CUDA peer-copy backend for topologies with peer access.

This backend deliberately refuses to run when CUDA reports no peer access; it
never labels a host-staged ``Tensor.to`` copy as P2P.
"""

from __future__ import annotations

from threading import Condition
from time import monotonic
from typing import Any

from hydraserve.transfer.backend import TransferBackend
from hydraserve.transfer.descriptor import TransferMode


class CudaP2PTransferBackend(TransferBackend):
    """Same-process CUDA P2P transport using a dedicated copy stream."""

    def __init__(
        self,
        src_gpu: int,
        dst_gpu: int,
        *,
        quantize_kv: bool = True,
        bandwidth_gbps: float | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA P2P requires PyTorch with CUDA support") from exc

        if src_gpu == dst_gpu or min(src_gpu, dst_gpu) < 0:
            raise ValueError("P2P endpoints must be distinct non-negative GPU ids")
        if not torch.cuda.is_available() or max(src_gpu, dst_gpu) >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA devices are unavailable")
        if not torch.cuda.can_device_access_peer(src_gpu, dst_gpu):
            raise RuntimeError(f"CUDA peer access is unavailable for GPU{src_gpu}->GPU{dst_gpu}")
        self.src_gpu = src_gpu
        self.dst_gpu = dst_gpu
        self.quantize_kv = quantize_kv
        self._bandwidth = bandwidth_gbps or 12.0
        self._stream = torch.cuda.Stream(device=dst_gpu)
        self._messages: dict[str, tuple[Any, Any]] = {}
        self._condition = Condition()

    @property
    def transfer_mode(self) -> TransferMode:
        return (
            TransferMode.QUANTIZED_TRANSFER
            if self.quantize_kv
            else TransferMode.FULL_TRANSFER
        )

    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None:
        import torch

        if dst != self.dst_gpu:
            raise ValueError(f"backend is bound to destination GPU{self.dst_gpu}")
        copy_stream = stream or self._stream
        with torch.cuda.device(self.dst_gpu), torch.cuda.stream(copy_stream):
            copied = self._copy_payload(payload, self.dst_gpu)
            event = torch.cuda.Event()
            event.record(copy_stream)
        with self._condition:
            if key in self._messages:
                raise RuntimeError(f"unconsumed P2P transfer exists: {key}")
            self._messages[key] = (copied, event)
            self._condition.notify_all()

    def receive(
        self, key: str, src: int, stream: Any = None, timeout: float | None = None
    ) -> Any:
        import torch

        if src != self.dst_gpu:
            raise ValueError(f"receive endpoint must be GPU{self.dst_gpu}")
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while key not in self._messages:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out waiting for P2P transfer {key}")
                self._condition.wait(remaining)
            payload, event = self._messages.pop(key)
        with torch.cuda.device(self.dst_gpu):
            consumer_stream = stream or torch.cuda.current_stream(self.dst_gpu)
            consumer_stream.wait_event(event)
        return payload

    def get_bandwidth(self) -> float:
        return self._bandwidth

    def get_latency(self) -> float:
        return 0.01

    def supports_layer_pipeline(self) -> bool:
        return True

    @classmethod
    def _copy_payload(cls, payload: Any, destination: int) -> Any:
        import torch

        if isinstance(payload, torch.Tensor):
            if payload.device.type != "cuda":
                raise TypeError("P2P payload tensors must already reside on CUDA")
            return payload.detach().to(
                device=f"cuda:{destination}", non_blocking=True, copy=True
            )
        if isinstance(payload, dict):
            return {key: cls._copy_payload(value, destination) for key, value in payload.items()}
        if isinstance(payload, tuple):
            return tuple(cls._copy_payload(value, destination) for value in payload)
        if isinstance(payload, list):
            return [cls._copy_payload(value, destination) for value in payload]
        if isinstance(payload, (str, int, float, bool, type(None))):
            return payload
        raise TypeError(f"unsupported P2P payload type: {type(payload).__name__}")
