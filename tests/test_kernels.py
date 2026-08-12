"""
Tests for Triton kernels: GDN fused, paged attention, RMSNorm.
"""

import pytest
import torch
import math

from hydraserve.kernels.gdn_fused import (
    gdn_fused_prefill, gdn_decode_step, benchmark_gdn_fused
)
from hydraserve.kernels.paged_attention import (
    paged_attention_decode, benchmark_paged_attention
)
from hydraserve.kernels.rmsnorm import rmsnorm_forward


# ─── GDN Fused Kernel Tests ─────────────────────────────────────────


class TestGDNFused:
    """Tests for GDN delta rule fused kernel."""

    @pytest.fixture
    def gdn_inputs(self):
        batch, seq_len = 1, 256
        n_key_heads, n_val_heads = 16, 32
        key_dim, val_dim = 128, 128
        device = "cuda" if torch.cuda.is_available() else "cpu"

        k = torch.randn(batch, seq_len, n_key_heads, key_dim, device=device)
        v = torch.randn(batch, seq_len, n_val_heads, val_dim, device=device)
        q = torch.randn(batch, seq_len, n_key_heads, key_dim, device=device)
        beta = torch.rand(batch, seq_len, n_key_heads, device=device)
        alpha = torch.rand(batch, seq_len, n_key_heads, device=device)
        gate = torch.rand(batch, seq_len, n_key_heads, device=device)

        return k, v, q, beta, alpha, gate, n_key_heads, key_dim, val_dim

    def test_prefill_output_shape(self, gdn_inputs):
        """Output should have correct shape."""
        k, v, q, beta, alpha, gate, n_heads, kd, vd = gdn_inputs
        output, state = gdn_fused_prefill(k, v, q, beta, alpha, gate)

        assert output.shape == (1, 256, 16, 128)
        assert state.shape == (1, 16, 128, 128)
        assert state.dtype == torch.float32

    def test_prefill_no_nan(self, gdn_inputs):
        """Output should not contain NaN."""
        k, v, q, beta, alpha, gate, n_heads, kd, vd = gdn_inputs
        output, state = gdn_fused_prefill(k, v, q, beta, alpha, gate)

        assert not torch.isnan(output).any()
        assert not torch.isnan(state).any()

    def test_decode_step_no_nan(self):
        """Decode step should not produce NaN."""
        batch = 2
        n_key_heads, n_val_heads = 16, 32
        key_dim, val_dim = 128, 128
        device = "cuda" if torch.cuda.is_available() else "cpu"

        k = torch.randn(batch, n_key_heads, key_dim, device=device)
        v = torch.randn(batch, n_val_heads, val_dim, device=device)
        q = torch.randn(batch, n_key_heads, key_dim, device=device)
        beta = torch.rand(batch, n_key_heads, device=device)
        alpha = torch.rand(batch, n_key_heads, device=device)
        gate = torch.rand(batch, n_key_heads, device=device)
        state = torch.randn(batch, n_key_heads, key_dim, val_dim, device=device)

        output, new_state = gdn_decode_step(
            k, v, q, beta, alpha, gate, state, n_key_heads, key_dim, val_dim
        )

        assert output.shape == (batch, n_key_heads, key_dim)
        assert new_state.shape == (batch, n_key_heads, key_dim, val_dim)
        assert not torch.isnan(output).any()
        assert not torch.isnan(new_state).any()

    def test_state_evolution(self):
        """State should change after each decode step (not stay zero)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_key_heads, key_dim, val_dim = 16, 128, 128

        k = torch.ones(1, n_key_heads, key_dim, device=device)
        v = torch.ones(1, n_key_heads * 2, val_dim, device=device)
        q = torch.ones(1, n_key_heads, key_dim, device=device)
        beta = torch.ones(1, n_key_heads, device=device)
        alpha = torch.ones(1, n_key_heads, device=device)
        gate = torch.ones(1, n_key_heads, device=device)
        state = torch.zeros(1, n_key_heads, key_dim, val_dim, device=device)

        _, new_state = gdn_decode_step(
            k, v, q, beta, alpha, gate, state, n_key_heads, key_dim, val_dim
        )

        # State should no longer be zero
        assert not torch.allclose(new_state, torch.zeros_like(new_state))


# ─── Paged Attention Tests ──────────────────────────────────────────


class TestPagedAttention:
    """Tests for paged attention decode kernel."""

    @pytest.fixture
    def attn_inputs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_seqs, num_heads, num_kv_heads = 4, 16, 4
        head_dim = 256
        context_len = 1024

        q = torch.randn(num_seqs, num_heads, head_dim, device=device)
        k = torch.randn(num_seqs, num_kv_heads, context_len, head_dim, device=device)
        v = torch.randn(num_seqs, num_kv_heads, context_len, head_dim, device=device)
        sm_scale = 1.0 / math.sqrt(head_dim)

        return q, k, v, sm_scale

    def test_output_shape(self, attn_inputs):
        q, k, v, sm_scale = attn_inputs
        output = paged_attention_decode(q, k, v, None, sm_scale=sm_scale)

        assert output.shape == q.shape  # [num_seqs, num_heads, head_dim]

    def test_output_no_nan(self, attn_inputs):
        q, k, v, sm_scale = attn_inputs
        output = paged_attention_decode(q, k, v, None, sm_scale=sm_scale)
        assert not torch.isnan(output).any()

    def test_causal_mask_equivalent(self, attn_inputs):
        """Decode attention should be equivalent to standard scaled dot-product."""
        q, k, v, sm_scale = attn_inputs

        output = paged_attention_decode(q, k, v, None, sm_scale=sm_scale)

        # Reference: manual attention
        q_expanded = q.unsqueeze(2)  # [B, H, 1, D]
        k_expanded = k.repeat_interleave(4, dim=1)  # GQA repeat
        v_expanded = v.repeat_interleave(4, dim=1)

        scores = torch.matmul(q_expanded, k_expanded.transpose(-2, -1)) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        ref_output = torch.matmul(attn, v_expanded).squeeze(2)

        # Should be close (within numerical tolerance)
        assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-3)


# ─── RMSNorm Tests ──────────────────────────────────────────────────


class TestRMSNorm:
    """Tests for RMSNorm kernel."""

    def test_output_shape(self):
        hidden_size = 2560
        x = torch.randn(2, 128, hidden_size)
        weight = torch.ones(hidden_size)

        output = rmsnorm_forward(x, weight)

        assert output.shape == x.shape

    def test_normalization(self):
        """RMS of output should be approximately 1."""
        hidden_size = 4096
        x = torch.randn(4, 64, hidden_size)
        weight = torch.ones(hidden_size)

        output = rmsnorm_forward(x, weight)
        rms = torch.sqrt(torch.mean(output.float() ** 2, dim=-1))

        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_weight_scaling(self):
        """Output should be scaled by weight."""
        hidden_size = 256
        x = torch.ones(1, 10, hidden_size)
        weight = torch.full((hidden_size,), 2.0)

        output = rmsnorm_forward(x, weight)
        # For uniform input, RMS = 1, so output = x * 2 / 1 = 2
        assert torch.allclose(output, torch.full_like(output, 2.0), atol=1e-5)


# ─── Benchmark Tests (CUDA only) ────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestKernelBenchmarks:
    """Benchmark tests for kernels (CUDA required)."""

    def test_gdn_benchmark_runs(self):
        result = benchmark_gdn_fused(seq_len=1024, num_iter=5)
        assert result["time_ms"] > 0
        assert result["tflops"] > 0

    def test_paged_attention_benchmark_runs(self):
        result = benchmark_paged_attention(context_len=2048, num_iter=5)
        assert result["time_ms"] > 0
