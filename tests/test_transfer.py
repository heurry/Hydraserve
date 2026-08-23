from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pytest

from hydraserve.cache import Int4Tensor, Int8Tensor, LinearState, dequantize_int8
from hydraserve.transfer import (
    CudaP2PTransferBackend,
    HybridStateBundle,
    InMemoryTransferBackend,
    RegionDescriptor,
    RegionType,
    SharedMemoryTransferBackend,
    SharedMemoryRingTransferBackend,
    StateTransferDescriptor,
    TransferMode,
    TransferPipeline,
    StateType,
)


def _state(model) -> LinearState:
    return LinearState(
        np.ones(model.ssm_state_shape, dtype=np.float32),
        np.full(model.conv_state_shape, 2, dtype=np.float32),
    )


def _ring_process_producer(namespace, prefix, value, rounds, barrier) -> None:
    backend = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    try:
        for sequence in range(rounds):
            barrier.wait()
            backend.send(
                f"{prefix}-{sequence}",
                np.full((64, 64), value + sequence, dtype=np.float32),
                1,
            )
            barrier.wait()
    finally:
        backend.close()


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


def test_state_type_and_tp_topology_round_trip(tiny_model) -> None:
    region = RegionDescriptor(
        StateType.SLIDING_WINDOW_KV,
        (1,),
        (2, 4, 8),
        "float16",
        False,
        0,
        1,
        src_tp_rank=1,
        dst_tp_rank=0,
        tp_world_size=2,
    )
    assert RegionDescriptor.from_dict(region.to_dict()) == region


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


def test_int8_transfer_quantizes_and_shared_memory_round_trips(tiny_model) -> None:
    backend = SharedMemoryTransferBackend(
        namespace="hydraserve-int8-pytest", mode=TransferMode.INT8_TRANSFER
    )
    try:
        pipeline = TransferPipeline(backend)
        kv = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
        descriptor = pipeline.send(
            6, tiny_model, 16, HybridStateBundle(_state(tiny_model), kv)
        )
        _, received = pipeline.receive(6, timeout=1)
        assert descriptor.regions[-1].dtype == "int8"
        assert isinstance(received.kv_cache, Int8Tensor)
        np.testing.assert_allclose(dequantize_int8(received.kv_cache), kv, atol=0.01)
        assert received.kv_cache.nbytes < kv.nbytes / 2
    finally:
        backend.close()


def test_chunked_kv_manifest_and_final_bundle(tiny_model) -> None:
    backend = InMemoryTransferBackend(TransferMode.FULL_TRANSFER)
    pipeline = TransferPipeline(
        backend, src_tp_rank=1, dst_tp_rank=0, tp_world_size=2
    )
    ranges = ((0, 4), (4, 7))
    pipeline.begin_chunked_send(51, tiny_model, 7, ranges)
    first = np.zeros(
        (tiny_model.num_full_attention_layers, 2, 4, tiny_model.num_kv_heads, tiny_model.head_dim),
        dtype=np.uint16,
    )
    second = np.ones(
        (tiny_model.num_full_attention_layers, 2, 3, tiny_model.num_kv_heads, tiny_model.head_dim),
        dtype=np.uint16,
    )
    pipeline.send_kv_chunk(51, 0, 4, first)
    pipeline.send_kv_chunk(51, 4, 7, second)
    descriptor = pipeline.send(
        51,
        tiny_model,
        7,
        HybridStateBundle(_state(tiny_model)),
        streamed_kv_ranges=ranges,
    )

    assert pipeline.begin_chunked_receive(51) == ranges
    np.testing.assert_array_equal(pipeline.receive_kv_chunk(51, 0, 4), first)
    np.testing.assert_array_equal(pipeline.receive_kv_chunk(51, 4, 7), second)
    received, bundle = pipeline.receive(51)
    assert received == descriptor
    assert received.streamed_kv
    assert bundle.kv_cache is None
    assert all(region.tp_world_size == 2 for region in received.regions)


def test_posix_shared_memory_partial_transfer(tiny_model) -> None:
    with SharedMemoryTransferBackend(
        namespace="hydraserve-pytest", mode=TransferMode.PARTIAL_TRANSFER
    ) as backend:
        pipeline = TransferPipeline(backend)
        pipeline.send(9, tiny_model, 12, HybridStateBundle(_state(tiny_model)))
        descriptor, received = pipeline.receive(9, timeout=1)
        assert descriptor.mode is TransferMode.PARTIAL_TRANSFER
        np.testing.assert_array_equal(received.recurrent.conv_state, 2)
        assert received.kv_cache is None


def test_shared_memory_typed_codec_preserves_nested_arrays_and_tuples() -> None:
    payload = {
        "states": (
            np.arange(12, dtype=np.float32).reshape(3, 4),
            np.arange(5, dtype=np.int32),
        ),
        "metadata": {"layers": [1, 3], "enabled": True},
    }
    with SharedMemoryTransferBackend(namespace="hydraserve-typed-pytest") as backend:
        backend.send("typed", payload, 1)
        restored = backend.receive("typed", 1, timeout=1)
    assert isinstance(restored["states"], tuple)
    np.testing.assert_array_equal(restored["states"][0], payload["states"][0])
    np.testing.assert_array_equal(restored["states"][1], payload["states"][1])
    assert restored["metadata"] == payload["metadata"]


def test_shared_memory_typed_codec_rejects_unsupported_objects() -> None:
    backend = SharedMemoryTransferBackend(namespace="hydraserve-typed-reject")
    with pytest.raises(TypeError, match="unsupported"):
        backend.send("bad", {"value": object()}, 1)


def test_persistent_shared_memory_ring_reuses_slots() -> None:
    namespace = "hydraserve-ring-pytest"
    sender = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    try:
        for sequence in range(4):
            payload = {
                "sequence": sequence,
                "value": np.full((32, 16), sequence, dtype=np.float32),
            }
            sender.send(f"chunk-{sequence}", payload, 1)
            restored = receiver.receive(f"chunk-{sequence}", 1, timeout=1)
            assert restored["sequence"] == sequence
            np.testing.assert_array_equal(restored["value"], payload["value"])
    finally:
        receiver.close()
        sender.close()


def test_persistent_ring_supports_two_concurrent_producers() -> None:
    namespace = "hydraserve-ring-mpsc-pytest"
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    rounds = 16
    context = mp.get_context("spawn")
    barrier = context.Barrier(3)
    first = context.Process(
        target=_ring_process_producer,
        args=(namespace, "a", 100, rounds, barrier),
    )
    second = context.Process(
        target=_ring_process_producer,
        args=(namespace, "b", 200, rounds, barrier),
    )
    first.start()
    second.start()
    try:
        for sequence in range(rounds):
            barrier.wait()
            restored_a = receiver.receive(f"a-{sequence}", 1, timeout=2)
            restored_b = receiver.receive(f"b-{sequence}", 1, timeout=2)
            np.testing.assert_array_equal(
                restored_a,
                np.full((64, 64), 100 + sequence, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                restored_b,
                np.full((64, 64), 200 + sequence, dtype=np.float32),
            )
            barrier.wait()
    finally:
        first.join(2)
        second.join(2)
        if first.is_alive():
            first.terminate()
            first.join(2)
        if second.is_alive():
            second.terminate()
            second.join(2)
        receiver.close()
    assert first.exitcode == 0
    assert second.exitcode == 0


def test_cuda_p2p_default_receive_waits_on_current_stream(monkeypatch) -> None:
    from contextlib import nullcontext
    from threading import Condition

    torch = pytest.importorskip("torch")

    class Stream:
        def __init__(self):
            self.waited = None

        def wait_event(self, event):
            self.waited = event

    backend = object.__new__(CudaP2PTransferBackend)
    backend.dst_gpu = 1
    backend._condition = Condition()
    event = object()
    payload = object()
    backend._messages = {"ready": (payload, event)}
    stream = Stream()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    received = backend.receive("ready", 1)

    assert received is payload
    assert stream.waited is event
