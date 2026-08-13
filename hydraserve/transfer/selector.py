from __future__ import annotations

from hydraserve.transfer.backend import SharedMemoryTransferBackend, TransferBackend
from hydraserve.transfer.cuda_p2p import CudaP2PTransferBackend


def select_transfer_backend(
    src_gpu: int,
    dst_gpu: int,
    *,
    backend: str = "auto",
    shm_namespace: str = "hydraserve",
) -> TransferBackend:
    """Select a backend without silently emulating unavailable P2P."""
    normalized = backend.lower()
    if normalized not in {"auto", "p2p", "shm"}:
        raise ValueError(f"unsupported transfer backend {backend!r}")
    if normalized == "shm":
        return SharedMemoryTransferBackend(namespace=shm_namespace)
    try:
        return CudaP2PTransferBackend(src_gpu, dst_gpu)
    except RuntimeError:
        if normalized == "p2p":
            raise
        return SharedMemoryTransferBackend(namespace=shm_namespace)
