"""
TransferBackend: Abstract interface for state transfer between GPUs.

Design motivation (§5.7): PD separation's core action is state transfer from
prefill GPU to decode GPU. The transfer medium can be NVLink (single node),
PCIe P2P (no NVLink), RDMA (cross-node), or SHM (fallback). Different media
have different bandwidths, latencies, and memory registration requirements.

This abstraction isolates transfer details so upper layers (pipeline, engines)
don't need to know about the underlying transport.

Backend selection (§5.7.4):
  1. Check for P2P access → benchmark bandwidth
     - ≥ 50 GB/s → NVLinkBackend (FULL_TRANSFER)
     - ≥ 10 GB/s → PCIeP2PBackend (QUANTIZED_TRANSFER)
  2. Check for RDMA device → RDMABackend (if hardware present)
  3. Fallback → SHMBackend (QUANTIZED_TRANSFER, sync)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional
import torch


class TransferMode(str, Enum):
    """Transfer strategy for different hardware configurations."""
    FULL_TRANSFER = "full"           # NVLink/RDMA: full BF16 KV + recurrent states
    QUANTIZED_TRANSFER = "quant"     # PCIe P2P/SHM: INT4 KV + recurrent states
    PARTIAL_TRANSFER = "partial"     # Low bandwidth: only recurrent states + KV recompute
    INTRA_GPU = "intra"              # MPS: same-GPU, zero-copy


class TransferBackend(ABC):
    """
    Abstract transfer backend.

    Each backend implementation handles the physical movement of tensor
    data from source GPU to destination GPU. Upper layers call send/receive
    without knowing the transport details.
    """

    def __init__(self, src_gpu: int = 0, dst_gpu: int = 1):
        self.src_gpu = src_gpu
        self.dst_gpu = dst_gpu

    @abstractmethod
    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        """Send a tensor to the destination GPU."""
        ...

    @abstractmethod
    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        """Receive a tensor from the source GPU."""
        ...

    @abstractmethod
    def get_bandwidth(self) -> float:
        """Theoretical bandwidth in GB/s. Used for pipeline depth calculation."""
        ...

    @abstractmethod
    def requires_memory_registration(self) -> bool:
        """Whether RDMA-style memory registration is needed."""
        ...

    @abstractmethod
    def register_memory(self, tensor: torch.Tensor) -> Any:
        """Register memory for RDMA (no-op for NVLink/PCIe)."""
        ...

    @abstractmethod
    def supports_layer_pipeline(self) -> bool:
        """Whether layer-level async pipeline is beneficial (bandwidth ≥ 10 GB/s)."""
        ...

    @abstractmethod
    def get_latency(self) -> float:
        """Single-transfer startup latency in milliseconds."""
        ...

    @property
    @abstractmethod
    def transfer_mode(self) -> TransferMode:
        """The transfer strategy this backend uses."""
        ...

    def synchronize(self) -> None:
        """Synchronize all streams on this backend's devices."""
        torch.cuda.synchronize(self.src_gpu)
        torch.cuda.synchronize(self.dst_gpu)


class NVLinkBackend(TransferBackend):
    """
    NVLink backend: GPU-to-GPU P2P via NVLink bridge (112 GB/s).

    TransferMode: FULL_TRANSFER
    Full BF16 KV Cache + recurrent states, 9ms for 32K context.
    100% hidden within prefill computation time.
    Layer-level pipeline is effective (per-layer state ~1MB, < 0.01ms).
    """

    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.cuda._p2p_send(tensor.contiguous(), dst=dst)
        else:
            torch.cuda._p2p_send(tensor.contiguous(), dst=dst)

    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.cuda._p2p_recv(tensor, src=src)
        else:
            torch.cuda._p2p_recv(tensor, src=src)

    def get_bandwidth(self) -> float:
        return 112.0

    def requires_memory_registration(self) -> bool:
        return False

    def register_memory(self, tensor: torch.Tensor) -> Any:
        return None  # No registration needed

    def supports_layer_pipeline(self) -> bool:
        return True

    def get_latency(self) -> float:
        return 0.005  # ~5 µs

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.FULL_TRANSFER


class PCIeP2PBackend(TransferBackend):
    """
    PCIe P2P backend: GPU-to-GPU via dual x16 PCIe (~12-16 GB/s).

    TransferMode: QUANTIZED_TRANSFER
    INT4 KV quantization compresses 1GB → 345MB, 29ms transfer for 32K.
    Layer-level pipeline still viable at ~14 GB/s.
    This is the DEFAULT backend when NVLink is not available.
    """

    def __init__(self, src_gpu: int = 0, dst_gpu: int = 1, quantize_kv: bool = True):
        super().__init__(src_gpu, dst_gpu)
        self.quantize_kv = quantize_kv
        self._benchmarked_bw: Optional[float] = None

    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        t = tensor.contiguous()
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.cuda._p2p_send(t, dst=dst)
        else:
            torch.cuda._p2p_send(t, dst=dst)

    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        if stream is not None:
            with torch.cuda.stream(stream):
                torch.cuda._p2p_recv(tensor, src=src)
        else:
            torch.cuda._p2p_recv(tensor, src=src)

    def get_bandwidth(self) -> float:
        if self._benchmarked_bw is None:
            self._benchmarked_bw = self._benchmark_p2p_bandwidth()
        return self._benchmarked_bw

    def requires_memory_registration(self) -> bool:
        return False

    def register_memory(self, tensor: torch.Tensor) -> Any:
        return None

    def supports_layer_pipeline(self) -> bool:
        return self.get_bandwidth() >= 10.0

    def get_latency(self) -> float:
        return 0.01  # ~10 µs

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.QUANTIZED_TRANSFER if self.quantize_kv else TransferMode.FULL_TRANSFER

    def _benchmark_p2p_bandwidth(self) -> float:
        """Benchmark actual P2P bandwidth between GPUs."""
        try:
            size = 100 * 1024 * 1024  # 100 MB
            t = torch.randn(size // 2, dtype=torch.bfloat16, device=self.src_gpu)
            dst_t = torch.empty(size // 2, dtype=torch.bfloat16, device=self.dst_gpu)

            # Warmup
            for _ in range(5):
                torch.cuda._p2p_send(t, dst=self.dst_gpu)
                torch.cuda._p2p_recv(dst_t, src=self.src_gpu)

            torch.cuda.synchronize()
            import time
            start = time.perf_counter()
            for _ in range(20):
                torch.cuda._p2p_send(t, dst=self.dst_gpu)
                torch.cuda._p2p_recv(dst_t, src=self.src_gpu)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            bw = (size * 20) / elapsed / 1e9
            return bw
        except Exception:
            return 12.0  # Conservative estimate for dual x16 PCIe


class SHMBackend(TransferBackend):
    """
    Shared Memory backend: GPU → pinned CPU → SHM → other GPU.

    TransferMode: QUANTIZED_TRANSFER
    2-hop PCIe transfer, ~8-10 GB/s (dual x16).
    1GB transfer takes ~130ms — exceeds prefill time for short contexts.
    Layer-level pipeline disabled (bandwidth insufficient).
    Falls back to synchronous transfer + INT4 quantization.
    """

    def __init__(self, src_gpu: int = 0, dst_gpu: int = 1):
        super().__init__(src_gpu, dst_gpu)
        self.pinned_buffers: dict = {}

    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        # GPU → pinned CPU memory → SHM → other GPU
        shape_key = tuple(tensor.shape)
        if shape_key not in self.pinned_buffers:
            self.pinned_buffers[shape_key] = torch.empty(
                tensor.shape, dtype=tensor.dtype,
                pin_memory=True, device='cpu'
            )

        cpu_buf = self.pinned_buffers[shape_key]
        if cpu_buf.shape != tensor.shape:
            cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype,
                                  pin_memory=True, device='cpu')
            self.pinned_buffers[shape_key] = cpu_buf

        if stream is not None:
            cpu_buf.copy_(tensor, non_blocking=True)
            stream.synchronize()
        else:
            cpu_buf.copy_(tensor)

    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        # Data already in pinned memory from send; copy to destination GPU
        shape_key = tuple(tensor.shape)
        if shape_key in self.pinned_buffers:
            tensor.copy_(self.pinned_buffers[shape_key])

    def get_bandwidth(self) -> float:
        return 8.0

    def requires_memory_registration(self) -> bool:
        return False

    def register_memory(self, tensor: torch.Tensor) -> Any:
        return None

    def supports_layer_pipeline(self) -> bool:
        return False  # 8 GB/s is too slow for layer-level pipeline

    def get_latency(self) -> float:
        return 0.05  # ~50 µs (2-hop)

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.QUANTIZED_TRANSFER


class IntraGPUBackend(TransferBackend):
    """
    Intra-GPU (MPS) backend: same-GPU PD separation, zero-copy.

    TransferMode: INTRA_GPU
    Prefill and decode processes share the same GPU via CUDA MPS.
    State is passed via GPU memory pointer — no actual data transfer.
    BulletServe-inspired single-GPU disaggregation.

    Limitations:
    - SM resource contention (not physical isolation like inter-GPU)
    - No libsmctrl for precise SM partitioning
    - N-1 truncation semantics still apply
    """

    def __init__(self, gpu_id: int = 0):
        super().__init__(src_gpu=gpu_id, dst_gpu=gpu_id)

    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        # Zero-copy: tensor already on shared GPU, just pass the pointer
        # The "transfer" is a metadata operation (pointer handoff)
        pass

    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        # Data already accessible on shared GPU
        pass

    def get_bandwidth(self) -> float:
        return float('inf')  # No transfer needed

    def requires_memory_registration(self) -> bool:
        return False

    def register_memory(self, tensor: torch.Tensor) -> Any:
        return None

    def supports_layer_pipeline(self) -> bool:
        return False  # No transfer needed

    def get_latency(self) -> float:
        return 0.0

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.INTRA_GPU


class RDMABackend(TransferBackend):
    """
    RDMA backend: cross-node transfer via InfiniBand/RoCE (25 GB/s).

    TransferMode: FULL_TRANSFER (RDMA is fast enough for full BF16 transfer)
    Interface definition only — requires RDMA NIC hardware for testing.
    Uses NIXL or Mooncake transfer engine under the hood.

    Opens cross-node disaggregation: 1P+3D no longer limited to single node.
    """

    def __init__(self, config: dict = None, src_gpu: int = 0, dst_gpu: int = 1):
        super().__init__(src_gpu, dst_gpu)
        self.config = config or {}
        self.engine = None  # TransferEngine placeholder
        self._registered_memory: dict = {}

    def send(self, tensor: torch.Tensor, dst: int, stream: Optional[Any] = None) -> None:
        # Requires: mr = register_memory(tensor); engine.rdma_write(mr, dst_addr, size)
        raise NotImplementedError(
            "RDMA backend requires InfiniBand hardware. "
            "Use NVLinkBackend or PCIeP2PBackend for GPU-to-GPU transfers."
        )

    def receive(self, tensor: torch.Tensor, src: int, stream: Optional[Any] = None) -> None:
        raise NotImplementedError("RDMA backend requires InfiniBand hardware.")

    def get_bandwidth(self) -> float:
        return 25.0  # 200 Gbps

    def requires_memory_registration(self) -> bool:
        return True

    def register_memory(self, tensor: torch.Tensor) -> Any:
        tid = id(tensor)
        if tid not in self._registered_memory:
            # In production: mr = self.engine.register_memory(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            self._registered_memory[tid] = tensor
        return self._registered_memory[tid]

    def supports_layer_pipeline(self) -> bool:
        return True  # 25 GB/s is sufficient

    def get_latency(self) -> float:
        return 0.01  # ~10 µs

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.FULL_TRANSFER


# ─── Backend Selection ──────────────────────────────────────────────


def select_backend(src_gpu: int = 0, dst_gpu: int = 1) -> TransferBackend:
    """
    Auto-select the best available transfer backend (§5.7.4).

    Priority:
      1. NVLink (≥50 GB/s P2P bandwidth)
      2. PCIe P2P (≥10 GB/s P2P bandwidth)
      3. SHM (fallback)
      4. RDMA (requires separate hardware check, not auto-detected)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU-to-GPU transfer")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        # Single GPU: use intra-GPU mode
        return IntraGPUBackend(gpu_id=src_gpu)

    if src_gpu == dst_gpu:
        return IntraGPUBackend(gpu_id=src_gpu)

    # Check P2P access
    try:
        can_access = torch.cuda.can_device_access_peer(src_gpu, dst_gpu)
    except Exception:
        can_access = False

    if can_access:
        # Benchmark to determine NVLink vs PCIe P2P
        backend = PCIeP2PBackend(src_gpu, dst_gpu)
        bw = backend.get_bandwidth()  # Triggers benchmark

        if bw >= 50.0:
            return NVLinkBackend(src_gpu, dst_gpu)   # NVLink detected
        elif bw >= 10.0:
            return backend                            # PCIe P2P
        else:
            return SHMBackend(src_gpu, dst_gpu)       # Slow P2P → SHM
    else:
        return SHMBackend(src_gpu, dst_gpu)           # No P2P → SHM


def benchmark_transfer(backend: TransferBackend, sizes_mb: list = None) -> dict:
    """Benchmark a transfer backend."""
    if sizes_mb is None:
        sizes_mb = [1, 10, 50, 100, 500, 1000]

    import time
    results = {}

    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 / 2)  # BF16 = 2 bytes/element
        t = torch.randn(num_elements, dtype=torch.bfloat16, device=backend.src_gpu)
        dst_t = torch.empty(num_elements, dtype=torch.bfloat16, device=backend.dst_gpu)

        # Warmup
        for _ in range(3):
            backend.send(t, backend.dst_gpu)
            backend.receive(dst_t, backend.src_gpu)
        backend.synchronize()

        start = time.perf_counter()
        n_iter = max(2, min(50, 1000 // max(1, size_mb)))
        for _ in range(n_iter):
            backend.send(t, backend.dst_gpu)
            backend.receive(dst_t, backend.src_gpu)
        backend.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter

        results[size_mb] = {
            "time_ms": elapsed * 1000,
            "bandwidth_gb_s": (size_mb / 1024) / elapsed,
        }

    return results
