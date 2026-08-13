from __future__ import annotations

import pytest

from hydraserve.transfer import (
    CudaP2PTransferBackend,
    SharedMemoryTransferBackend,
    select_transfer_backend,
)


def test_explicit_shm_selection() -> None:
    backend = select_transfer_backend(0, 1, backend="shm", shm_namespace="selector-test")
    assert isinstance(backend, SharedMemoryTransferBackend)
    backend.close()


def test_auto_falls_back_to_shm_without_cuda_peer_access() -> None:
    backend = select_transfer_backend(0, 1, backend="auto", shm_namespace="selector-auto")
    # The development host is NODE topology with no CUDA peer access. On a CI
    # host with P2P this assertion remains architecture-correct.
    try:
        import torch
    except ImportError:
        assert isinstance(backend, SharedMemoryTransferBackend)
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(0, 1):
            assert isinstance(backend, CudaP2PTransferBackend)
        else:
            assert isinstance(backend, SharedMemoryTransferBackend)
    if isinstance(backend, SharedMemoryTransferBackend):
        backend.close()


def test_explicit_p2p_never_silently_falls_back() -> None:
    try:
        import torch
    except ImportError:
        with pytest.raises(RuntimeError):
            select_transfer_backend(0, 1, backend="p2p")
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not torch.cuda.can_device_access_peer(0, 1):
        with pytest.raises(RuntimeError, match="unavailable"):
            select_transfer_backend(0, 1, backend="p2p")
