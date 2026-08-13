from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from hydraserve.kernels.gdn import causal_depthwise_conv as triton_causal_conv
from hydraserve.kernels.gdn import gated_delta_recurrent
from hydraserve.kernels.paged_attention import paged_attention as triton_paged_attention
from hydraserve.kernels.reference import (
    gated_delta_rule,
    causal_depthwise_conv as reference_causal_conv,
    gated_rms_norm as reference_gated_rms_norm,
    paged_attention as reference_paged_attention,
    rms_norm as reference_rms_norm,
)
from hydraserve.kernels.rmsnorm import gated_rms_norm as triton_gated_rms_norm
from hydraserve.kernels.rmsnorm import rms_norm as triton_rms_norm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("hidden", [64, 256, 2560, 4096, 5120])
def test_triton_rms_norm_matches_reference(hidden: int) -> None:
    torch.manual_seed(1)
    x = torch.randn(7, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    expected = reference_rms_norm(x, weight)
    actual = triton_rms_norm(x, weight)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_gated_rms_norm_matches_reference() -> None:
    torch.manual_seed(8)
    x = torch.randn(4, 7, 32, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    expected = reference_gated_rms_norm(x, gate, weight)
    actual = triton_gated_rms_norm(x, gate, weight)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("sequence", [1, 2, 7])
def test_triton_causal_conv_matches_reference(sequence: int) -> None:
    torch.manual_seed(9)
    x = torch.randn(2, sequence, 17, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(17, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(2, 17, 4, device="cuda", dtype=torch.bfloat16)
    expected, expected_state = reference_causal_conv(x, weight, state)
    actual, actual_state = triton_causal_conv(x, weight, state)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.parametrize("sequence", [1, 3, 17])
def test_triton_gdn_matches_reference(sequence: int) -> None:
    torch.manual_seed(2)
    shape = (2, sequence, 4)
    q = torch.randn(*shape, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(*shape, 12, device="cuda", dtype=torch.bfloat16)
    decay = -torch.rand(*shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(*shape, device="cuda", dtype=torch.float32)
    initial = torch.randn(2, 4, 16, 12, device="cuda", dtype=torch.float32) * 0.1
    expected, expected_state = gated_delta_rule(q, k, v, decay, beta, initial)
    actual, actual_state = gated_delta_recurrent(q, k, v, decay, beta, initial.clone())
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("head_dim", [32, 128])
@pytest.mark.parametrize(
    "lengths", [[1, 5], [7, 12], [16, 13], [17, 31], [63, 129]]
)
def test_triton_paged_attention_matches_reference(
    lengths: list[int], head_dim: int
) -> None:
    torch.manual_seed(3)
    batch, query_heads, kv_heads = 2, 4, 2
    block_size = 4
    table_width = (max(lengths) + block_size - 1) // block_size
    physical_blocks = batch * table_width
    query = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(
        physical_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    table = torch.arange(
        physical_blocks, device="cuda", dtype=torch.int32
    ).reshape(batch, table_width)
    table[0] = table[0].flip(0)
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    expected = reference_paged_attention(query, key, value, table, sequence_lengths)
    actual = triton_paged_attention(query, key, value, table, sequence_lengths)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
