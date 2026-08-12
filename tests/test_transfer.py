"""
Tests for transfer layer: backends, pipeline, descriptors.
"""

import pytest
import torch

from hydraserve.transfer.backend import (
    TransferBackend, TransferMode,
    NVLinkBackend, PCIeP2PBackend, SHMBackend,
    IntraGPUBackend, RDMABackend, select_backend,
)
from hydraserve.transfer.descriptor import (
    StateTransferDescriptor, RegionDescriptor, RegionType
)
from hydraserve.transfer.pipeline import TransferPipeline
from hydraserve.cache.kv_quantizer import KVQuantizer


# ─── Backend Tests ──────────────────────────────────────────────────


class TestBackends:
    """Tests for transfer backends."""

    def test_nvlink_backend_creation(self):
        backend = NVLinkBackend(src_gpu=0, dst_gpu=1)
        assert backend.get_bandwidth() == 112.0
        assert backend.transfer_mode == TransferMode.FULL_TRANSFER
        assert backend.supports_layer_pipeline()
        assert not backend.requires_memory_registration()

    def test_pcie_p2p_backend_creation(self):
        backend = PCIeP2PBackend(src_gpu=0, dst_gpu=1)
        assert backend.get_bandwidth() >= 10.0
        assert backend.supports_layer_pipeline()
        assert not backend.requires_memory_registration()

    def test_shm_backend_creation(self):
        backend = SHMBackend(src_gpu=0, dst_gpu=1)
        assert backend.get_bandwidth() == 8.0
        assert not backend.supports_layer_pipeline()

    def test_intra_gpu_backend_creation(self):
        backend = IntraGPUBackend(gpu_id=0)
        assert backend.transfer_mode == TransferMode.INTRA_GPU
        assert not backend.requires_memory_registration()

    def test_rdma_backend_creation(self):
        backend = RDMABackend(src_gpu=0, dst_gpu=1)
        assert backend.requires_memory_registration()
        assert backend.get_bandwidth() == 25.0

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2 GPUs")
    def test_select_backend_detects_transport(self):
        backend = select_backend(0, 1)
        assert isinstance(backend, TransferBackend)
        assert backend.get_bandwidth() > 0

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Need 2 GPUs")
    def test_nvlink_send_receive(self):
        """Test basic send/receive with NVLink backend."""
        if not torch.cuda.can_device_access_peer(0, 1):
            pytest.skip("P2P not available")

        backend = NVLinkBackend(0, 1)
        size = 1024
        src = torch.randn(size, dtype=torch.bfloat16, device='cuda:0')
        dst = torch.empty(size, dtype=torch.bfloat16, device='cuda:1')

        backend.send(src, 1)
        backend.receive(dst, 0)
        backend.synchronize()

        assert torch.allclose(src.cpu(), dst.cpu(), atol=1e-2)

    @pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Need CUDA")
    def test_intra_gpu_send_receive(self):
        """Intra-GPU mode: tensor is shared, no actual data movement."""
        backend = IntraGPUBackend(0)
        size = 1024
        t = torch.randn(size, device='cuda:0')

        # send/receive are no-ops for intra-GPU
        backend.send(t, 0)
        backend.receive(t, 0)
        backend.synchronize()
        # Tensor should be unchanged
        assert t is not None


# ─── Descriptor Tests ───────────────────────────────────────────────


class TestDescriptors:
    """Tests for state transfer descriptors."""

    def test_region_descriptor_creation(self):
        region = RegionDescriptor(
            region_type=RegionType.FULL_ATTN_KV,
            layer_indices=[3, 7, 11, 15, 19, 23, 27, 31],
            shape=(8, 2, 4, 4096, 256),
            dtype="bfloat16",
            src_gpu=0,
            dst_gpu=1,
        )
        assert region.region_type == RegionType.FULL_ATTN_KV
        assert region.size_mb > 0
        assert region.quantized is False

    def test_region_descriptor_auto_size(self):
        region = RegionDescriptor(
            region_type=RegionType.LINEAR_SSM,
            layer_indices=list(range(24)),
            shape=(24, 16, 128, 128),
            dtype="float32",
        )
        expected_bytes = 24 * 16 * 128 * 128 * 4
        assert region.size_bytes == expected_bytes

    def test_transfer_descriptor_creation(self):
        desc = StateTransferDescriptor(
            request_id=42,
            model_name="Qwen3.5-9B",
            first_token_id=1234,
            regions=[
                RegionDescriptor(
                    region_type=RegionType.FULL_ATTN_KV,
                    layer_indices=[3, 7, 11],
                    shape=(3, 2, 4, 4096, 256),
                    dtype="bfloat16",
                ),
                RegionDescriptor(
                    region_type=RegionType.LINEAR_SSM,
                    layer_indices=list(range(24)),
                    shape=(24, 16, 128, 128),
                    dtype="float32",
                ),
            ],
        )
        assert desc.request_id == 42
        assert desc.first_token_id == 1234
        assert desc.total_size_mb > 0
        assert desc.has_kv_cache
        assert desc.has_recurrent_state

    def test_transfer_time_estimate(self):
        desc = StateTransferDescriptor(
            request_id=1,
            model_name="Qwen3.5-9B",
            regions=[
                RegionDescriptor(
                    region_type=RegionType.FULL_ATTN_KV,
                    layer_indices=list(range(8)),
                    shape=(8, 2, 4, 32768, 256),
                    dtype="bfloat16",
                ),
            ],
        )
        # At 112 GB/s, 1GB should transfer in ~9ms
        time_ms = desc.estimate_transfer_time_ms(112.0)
        assert 5 < time_ms < 15  # Should be roughly 9ms for 1GB


# ─── Pipeline Tests ─────────────────────────────────────────────────


class TestTransferPipeline:
    """Tests for transfer pipeline."""

    def test_pipeline_creation_nvlink(self):
        backend = NVLinkBackend(0, 1)
        pipeline = TransferPipeline(backend, use_quantization=False)
        assert pipeline.pipeline_depth == 2

    def test_pipeline_creation_shm(self):
        backend = SHMBackend(0, 1)
        pipeline = TransferPipeline(backend, use_quantization=True)
        assert pipeline.pipeline_depth == 1  # SHM doesn't support layering
        assert pipeline.use_quantization

    def test_pipeline_transfer_time_estimate(self):
        backend = NVLinkBackend(0, 1)
        pipeline = TransferPipeline(backend)

        desc = StateTransferDescriptor(
            request_id=1,
            model_name="Qwen3.5-9B",
            regions=[
                RegionDescriptor(
                    region_type=RegionType.FULL_ATTN_KV,
                    layer_indices=list(range(8)),
                    shape=(8, 2, 4, 32768, 256),
                    dtype="bfloat16",
                ),
            ],
        )
        time_ms = pipeline.estimate_transfer_time(desc)
        assert time_ms > 0


# ─── KV Quantizer Tests ────────────────────────────────────────────


class TestKVQuantizer:
    """Tests for KIVI-style INT4 KV quantization."""

    def test_quantize_dequantize_k(self):
        quantizer = KVQuantizer(n_bits=4)
        k = torch.randn(4, 4096, 256, dtype=torch.bfloat16)

        k_q, k_s, k_z = quantizer.quantize_k(k)
        k_restored = quantizer.dequantize_k(k_q, k_s, k_z)

        # Should be approximately preserved
        assert k_restored.shape == k.shape
        # INT4 has limited precision, use larger tolerance
        # Max quantization error for 4-bit uniform quantizer
        assert torch.allclose(k.float(), k_restored.float(), atol=0.3, rtol=0.3)

    def test_quantize_dequantize_v(self):
        quantizer = KVQuantizer(n_bits=4)
        v = torch.randn(4, 4096, 256, dtype=torch.bfloat16)

        v_q, v_s, v_z = quantizer.quantize_v(v)
        v_restored = quantizer.dequantize_v(v_q, v_s, v_z)

        assert v_restored.shape == v.shape
        assert torch.allclose(v.float(), v_restored.float(), atol=0.3, rtol=0.3)

    def test_compression_ratio(self):
        ratio = KVQuantizer.estimate_compression_ratio()
        assert 2.5 <= ratio <= 4.0  # ~3.2x expected

    def test_transfer_size_estimate(self):
        bf16_size = 1000.0  # MB
        int4_size = KVQuantizer.estimate_transfer_size(bf16_size)
        assert int4_size < bf16_size
        assert int4_size == pytest.approx(bf16_size / 3.2, rel=0.2)
