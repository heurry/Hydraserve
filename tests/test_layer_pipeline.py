from __future__ import annotations

import numpy as np
import pytest

from hydraserve.transfer import (
    InMemoryTransferBackend,
    LayerTransferPipeline,
    RegionDescriptor,
    RegionType,
    StateTransferDescriptor,
    TransferMode,
)


def _regions(tiny_model):
    return tuple(
        RegionDescriptor(
            RegionType.LINEAR_SSM,
            (layer,),
            tiny_model.ssm_state_shape[1:],
            "float32",
            False,
            0,
            1,
        )
        for layer in tiny_model.linear_layer_indices
    )


def test_manifest_first_layer_pipeline(tiny_model) -> None:
    backend = InMemoryTransferBackend(
        TransferMode.FULL_TRANSFER, bandwidth_gbps=112
    )
    pipeline = LayerTransferPipeline(backend)
    regions = _regions(tiny_model)
    descriptor = StateTransferDescriptor(
        31, tiny_model.name, 9, 7, TransferMode.FULL_TRANSFER, regions
    )
    pipeline.begin_send(descriptor)
    for slot, region in enumerate(regions):
        pipeline.send_region(
            descriptor.request_id,
            region,
            np.full(region.shape, slot, dtype=np.float32),
        )
    received_descriptor = pipeline.begin_receive(31)
    assert received_descriptor == descriptor
    for slot, region in enumerate(received_descriptor.regions):
        received = pipeline.receive_region(31, region)
        np.testing.assert_array_equal(received, slot)


def test_layer_pipeline_rejects_low_bandwidth_backend() -> None:
    backend = InMemoryTransferBackend(
        TransferMode.PARTIAL_TRANSFER, bandwidth_gbps=4.58
    )
    with pytest.raises(ValueError, match="does not support"):
        LayerTransferPipeline(backend)
