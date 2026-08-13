from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydraserve.model.runtime import RuntimeState
from hydraserve.transfer import (
    HybridStateBundle,
    InMemoryTransferBackend,
    RuntimeStateCodec,
    TransferMode,
    TransferPipeline,
)


def test_runtime_state_transfer_round_trip(tiny_model) -> None:
    state = RuntimeState(sequence_length=13)
    for slot, layer_index in enumerate(tiny_model.linear_layer_indices):
        state.recurrent[layer_index] = torch.full(
            (1, *tiny_model.ssm_state_shape[1:]), slot + 1, dtype=torch.float32
        )
        state.convolution[layer_index] = torch.full(
            (1, *tiny_model.conv_state_shape[1:]), slot + 11, dtype=torch.bfloat16
        )
    bundle = RuntimeStateCodec.extract(tiny_model, state)
    assert bundle.recurrent.ssm_state.dtype == np.float32
    assert bundle.recurrent.conv_state.dtype == np.float32

    pipeline = TransferPipeline(InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER))
    descriptor = pipeline.send(21, tiny_model, 13, bundle, first_token_id=7)
    received_descriptor, received = pipeline.receive(21)
    restored = RuntimeStateCodec.install(
        tiny_model, received_descriptor, received, device="cpu"
    )
    assert restored.sequence_length == descriptor.prompt_length
    for layer_index in tiny_model.linear_layer_indices:
        torch.testing.assert_close(
            restored.recurrent[layer_index], state.recurrent[layer_index]
        )
        torch.testing.assert_close(
            restored.convolution[layer_index], state.convolution[layer_index].float()
        )
