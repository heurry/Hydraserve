from __future__ import annotations

from hydraserve.transfer.backend import SharedMemoryTransferBackend, TransferBackend
from hydraserve.transfer.cuda_p2p import CudaP2PTransferBackend
from hydraserve.transfer.descriptor import TransferMode
from hydraserve.transfer.shm_ring import SharedMemoryRingTransferBackend


def select_transfer_backend(
    src_gpu: int,
    dst_gpu: int,
    *,
    backend: str = "auto",
    shm_namespace: str = "hydraserve",
    mode: TransferMode = TransferMode.FULL_TRANSFER,
    ring_slots: int = 3,
    ring_slot_bytes: int = 64 << 20,
) -> TransferBackend:
    """Select a backend without silently emulating unavailable P2P."""
    normalized = backend.lower()
    if normalized not in {"auto", "p2p", "shm", "shm-ring"}:
        raise ValueError(f"unsupported transfer backend {backend!r}")
    if normalized == "shm-ring":
        return SharedMemoryRingTransferBackend(
            namespace=shm_namespace,
            mode=mode,
            slots=ring_slots,
            slot_bytes=ring_slot_bytes,
        )
    if normalized == "shm":
        return SharedMemoryTransferBackend(namespace=shm_namespace, mode=mode)
    try:
        return CudaP2PTransferBackend(src_gpu, dst_gpu)
    except RuntimeError:
        if normalized == "p2p":
            raise
        return SharedMemoryRingTransferBackend(
            namespace=shm_namespace,
            mode=mode,
            slots=ring_slots,
            slot_bytes=ring_slot_bytes,
        )
