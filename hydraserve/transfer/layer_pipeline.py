from __future__ import annotations

from threading import RLock
from typing import Any

from hydraserve.transfer.backend import TransferBackend
from hydraserve.transfer.descriptor import RegionDescriptor, StateTransferDescriptor


class LayerTransferPipeline:
    """Manifest-first protocol for overlapping layer compute and transfer."""

    def __init__(self, backend: TransferBackend, *, src_endpoint: int = 0, dst_endpoint: int = 1) -> None:
        if not backend.supports_layer_pipeline():
            raise ValueError("selected backend does not support layer-level pipelining")
        self.backend = backend
        self.src_endpoint = src_endpoint
        self.dst_endpoint = dst_endpoint
        self._sent: set[tuple[int, int, str]] = set()
        self._received: set[tuple[int, int, str]] = set()
        self._lock = RLock()

    def begin_send(self, descriptor: StateTransferDescriptor) -> None:
        self.backend.send(
            self._manifest_key(descriptor.request_id),
            descriptor.to_dict(),
            self.dst_endpoint,
        )

    def begin_receive(
        self, request_id: int, *, timeout: float | None = None
    ) -> StateTransferDescriptor:
        data = self.backend.receive(
            self._manifest_key(request_id), self.dst_endpoint, timeout=timeout
        )
        descriptor = StateTransferDescriptor.from_dict(data)
        if descriptor.request_id != request_id:
            raise RuntimeError("received layer manifest for the wrong request")
        return descriptor

    def send_region(self, request_id: int, region: RegionDescriptor, payload: Any) -> None:
        if len(region.layer_indices) != 1:
            raise ValueError("a pipelined region must describe exactly one layer")
        identity = self._identity(request_id, region)
        with self._lock:
            if identity in self._sent:
                raise RuntimeError(f"layer region was already sent: {identity}")
            self._sent.add(identity)
        self.backend.send(
            self._region_key(request_id, region), payload, self.dst_endpoint
        )

    def receive_region(
        self,
        request_id: int,
        region: RegionDescriptor,
        *,
        timeout: float | None = None,
    ) -> Any:
        if len(region.layer_indices) != 1:
            raise ValueError("a pipelined region must describe exactly one layer")
        identity = self._identity(request_id, region)
        with self._lock:
            if identity in self._received:
                raise RuntimeError(f"layer region was already received: {identity}")
            self._received.add(identity)
        return self.backend.receive(
            self._region_key(request_id, region),
            self.dst_endpoint,
            timeout=timeout,
        )

    @staticmethod
    def _manifest_key(request_id: int) -> str:
        return f"request:{request_id}:manifest"

    @staticmethod
    def _identity(request_id: int, region: RegionDescriptor) -> tuple[int, int, str]:
        return request_id, region.layer_indices[0], region.region_type.value

    @classmethod
    def _region_key(cls, request_id: int, region: RegionDescriptor) -> str:
        _, layer, region_type = cls._identity(request_id, region)
        return f"request:{request_id}:layer:{layer}:{region_type}"
