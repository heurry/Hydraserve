from __future__ import annotations

import json

import numpy as np
import pytest

from hydraserve.cache import Int4Tensor, LinearState
from hydraserve.transfer import (
    HybridStateBundle,
    InMemoryTransferBackend,
    RegionDescriptor,
    RegionType,
    SharedMemoryTransferBackend,
    StateTransferDescriptor,
    TransferMode,
    TransferPipeline,
)


def _state(model) -> LinearState:
    return LinearState(
        np.ones(model.ssm_state_shape, dtype=np.float32),
        np.full(model.conv_state_shape, 2, dtype=np.float32),
    )


def test_descriptor_is_json_round_trip_safe(tiny_model) -> None:
    region = RegionDescriptor(
        RegionType.LINEAR_SSM,
        tiny_model.linear_layer_indices,
        tiny_model.ssm_state_shape,
        "float32",
        False,
        0,
        1,
    )
    conv = RegionDescriptor(
        RegionType.LINEAR_CONV,
        tiny_model.linear_layer_indices,
        tiny_model.conv_state_shape,
        "float32",
        False,
        0,
        1,
    )
    descriptor = StateTransferDescriptor(2, tiny_model.name, 10, 42, TransferMode.PARTIAL_TRANSFER, (region, conv))
    restored = StateTransferDescriptor.from_dict(json.loads(json.dumps(descriptor.to_dict())))
    assert restored == descriptor


def test_recurrent_state_cannot_be_quantized(tiny_model) -> None:
    with pytest.raises(ValueError, match="unquantized float32"):
        RegionDescriptor(
            RegionType.LINEAR_SSM,
            tiny_model.linear_layer_indices,
            tiny_model.ssm_state_shape,
            "int4",
            True,
            0,
            1,
        )


def test_partial_transfer_excludes_kv(tiny_model) -> None:
    backend = InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    pipeline = TransferPipeline(backend)
    descriptor = pipeline.send(
        4, tiny_model, 12, HybridStateBundle(_state(tiny_model)), first_token_id=88
    )
    received_descriptor, received = pipeline.receive(4)
    assert descriptor == received_descriptor
    assert received.kv_cache is None
    assert received.recurrent.ssm_state.dtype == np.float32


def test_descriptor_records_n_minus_one_state_boundary(tiny_model) -> None:
    pipeline = TransferPipeline(
        InMemoryTransferBackend(TransferMode.PARTIAL_TRANSFER)
    )
    descriptor = pipeline.send(
        41,
        tiny_model,
        12,
        HybridStateBundle(_state(tiny_model)),
        first_token_id=8,
        state_token_count=11,
    )
    received, _ = pipeline.receive(41)
    assert descriptor.state_token_count == received.state_token_count == 11


def test_quantized_transfer_packs_kv(tiny_model) -> None:
    backend = InMemoryTransferBackend(TransferMode.QUANTIZED_TRANSFER, bandwidth_gbps=14)
    pipeline = TransferPipeline(backend)
    kv = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
    descriptor = pipeline.send(5, tiny_model, 16, HybridStateBundle(_state(tiny_model), kv))
    _, received = pipeline.receive(5)
    assert descriptor.regions[-1].dtype == "int4"
    assert isinstance(received.kv_cache, Int4Tensor)
    assert backend.supports_layer_pipeline()


def test_posix_shared_memory_partial_transfer(tiny_model) -> None:
    with SharedMemoryTransferBackend(namespace="hydraserve-pytest") as backend:
        pipeline = TransferPipeline(backend)
        pipeline.send(9, tiny_model, 12, HybridStateBundle(_state(tiny_model)))
        descriptor, received = pipeline.receive(9, timeout=1)
        assert descriptor.mode is TransferMode.PARTIAL_TRANSFER
        np.testing.assert_array_equal(received.recurrent.conv_state, 2)
        assert received.kv_cache is None
