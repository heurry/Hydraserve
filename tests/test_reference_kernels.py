from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydraserve.kernels.reference import (
    apply_text_rope,
    causal_depthwise_conv,
    gated_delta_rule,
    paged_attention,
    rms_norm,
)


def test_zero_centered_rms_norm() -> None:
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    weight = torch.zeros(2)
    output = rms_norm(x, weight, eps=0.0)
    expected = x / torch.sqrt(torch.tensor(12.5))
    torch.testing.assert_close(output, expected)


def test_causal_conv_chunk_state_matches_full_sequence() -> None:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(2, 9, 5, generator=generator)
    weight = torch.randn(5, 4, generator=generator)
    full, full_state = causal_depthwise_conv(x, weight)
    first, state = causal_depthwise_conv(x[:, :4], weight)
    second, state = causal_depthwise_conv(x[:, 4:], weight, state)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(state, full_state)


def test_gdn_chunk_state_matches_full_sequence() -> None:
    generator = torch.Generator().manual_seed(11)
    q = torch.randn(2, 7, 4, 8, generator=generator)
    k = torch.randn(2, 7, 4, 8, generator=generator)
    v = torch.randn(2, 7, 4, 6, generator=generator)
    decay = -torch.rand(2, 7, 4, generator=generator)
    beta = torch.rand(2, 7, 4, generator=generator)
    full, full_state = gated_delta_rule(q, k, v, decay, beta)
    first, state = gated_delta_rule(q[:, :3], k[:, :3], v[:, :3], decay[:, :3], beta[:, :3])
    second, state = gated_delta_rule(
        q[:, 3:], k[:, 3:], v[:, 3:], decay[:, 3:], beta[:, 3:], state
    )
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(state, full_state)


def test_text_rope_preserves_vector_norm() -> None:
    x = torch.randn(2, 5, 3, 16)
    rotated = apply_text_rope(x, torch.arange(5), theta=10_000_000, rotary_dim=8)
    torch.testing.assert_close(rotated.float().norm(dim=-1), x.float().norm(dim=-1))
    torch.testing.assert_close(rotated[..., 8:], x[..., 8:])


def test_reference_paged_attention_obeys_physical_table() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.zeros(3, 2, 1, 2)
    value = torch.zeros_like(key)
    key[2, :, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    value[2, :, 0] = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    table = torch.tensor([[2]], dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    output = paged_attention(query, key, value, table, lengths)
    assert output.shape == query.shape
    assert output[0, 0, 0] > output[0, 0, 1]
    assert output[0, 1, 1] > output[0, 1, 0]
