from __future__ import annotations

import json
import multiprocessing as mp
from threading import Event, Thread

import numpy as np
import pytest
import torch

import hydraserve.transfer.backend as transfer_backend_module
from hydraserve.cache import (
    Int4Tensor,
    Int8Tensor,
    KVBlockManager,
    LinearState,
    PagedInt8KVTensor,
    PagedKVCache,
    dequantize_int8,
    dequantize_paged_int8_kv,
)
from hydraserve.transfer.runtime_codec import RuntimeStateCodec
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
    TransferCancelledError,
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


def test_in_memory_receive_can_be_cancelled_without_waiting_for_timeout() -> None:
    backend = InMemoryTransferBackend()
    cancelled = Event()
    errors = []

    def receive() -> None:
        try:
            backend.receive("never-published", 1, timeout=2, cancel_event=cancelled)
        except Exception as exc:  # pragma: no branch - asserted below
            errors.append(exc)

    thread = Thread(target=receive)
    thread.start()
    cancelled.set()
    thread.join(1)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], TransferCancelledError)


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


def test_raw_paged_int8_kv_transfer_round_trips_without_bf16_payload(tiny_model) -> None:
    backend = InMemoryTransferBackend(TransferMode.INT8_TRANSFER)
    pipeline = TransferPipeline(backend)
    key = np.arange(
        tiny_model.num_full_attention_layers
        * 4
        * tiny_model.num_kv_heads
        * tiny_model.head_dim,
        dtype=np.int8,
    ).reshape(
        tiny_model.num_full_attention_layers,
        4,
        tiny_model.num_kv_heads,
        tiny_model.head_dim,
    )
    value = -key
    key_scales = np.full(
        (tiny_model.num_full_attention_layers, 4, tiny_model.num_kv_heads),
        0.25,
        dtype=np.float32,
    )
    value_scales = np.full_like(key_scales, 0.5)
    raw = PagedInt8KVTensor(
        key,
        value,
        key_scales,
        value_scales,
        (
            tiny_model.num_full_attention_layers,
            2,
            4,
            tiny_model.num_kv_heads,
            tiny_model.head_dim,
        ),
        "bfloat16",
    )

    descriptor = pipeline.send(
        61,
        tiny_model,
        4,
        HybridStateBundle(_state(tiny_model), raw),
        first_token_id=9,
    )
    _, received = pipeline.receive(61)

    assert descriptor.regions[-1].dtype == "int8"
    assert isinstance(received.kv_cache, PagedInt8KVTensor)
    np.testing.assert_array_equal(received.kv_cache.key, key)
    np.testing.assert_array_equal(received.kv_cache.value, value)
    np.testing.assert_array_equal(received.kv_cache.key_scales, key_scales)
    np.testing.assert_allclose(
        dequantize_paged_int8_kv(received.kv_cache)[:, 0],
        key.astype(np.float32) * key_scales[..., None],
    )


def test_runtime_codec_installs_raw_int8_cache_without_requantizing(tiny_model) -> None:
    src = PagedKVCache(
        tiny_model,
        KVBlockManager(4, 4),
        device="cpu",
        dtype=torch.bfloat16,
        kv_quant="int8",
    )
    dst = PagedKVCache(
        tiny_model,
        KVBlockManager(4, 4),
        device="cpu",
        dtype=torch.bfloat16,
        kv_quant="int8",
    )
    src.allocate(71, 4)
    dst.allocate(71, 4)
    positions = torch.arange(4)
    for layer_index in tiny_model.full_attention_layer_indices:
        key = torch.linspace(-1, 1, 4 * tiny_model.num_kv_heads * tiny_model.head_dim)
        key = key.reshape(4, tiny_model.num_kv_heads, tiny_model.head_dim)
        value = key * 0.5
        src.write(71, layer_index, positions, key, value)

    payload = RuntimeStateCodec.extract_kv_range(
        tiny_model, src, 71, 0, 4, mode=TransferMode.INT8_TRANSFER
    )
    assert isinstance(payload, PagedInt8KVTensor)
    RuntimeStateCodec.install_kv_range(tiny_model, dst, 71, payload, start=0)
    restored = RuntimeStateCodec.extract_kv_range(
        tiny_model, dst, 71, 0, 4, mode=TransferMode.INT8_TRANSFER
    )

    assert isinstance(restored, PagedInt8KVTensor)
    np.testing.assert_array_equal(restored.key, payload.key)
    np.testing.assert_array_equal(restored.value, payload.value)
    np.testing.assert_array_equal(restored.key_scales, payload.key_scales)
    np.testing.assert_array_equal(restored.value_scales, payload.value_scales)


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


def test_shared_memory_receive_retries_zero_size_creation_race(monkeypatch) -> None:
    backend = SharedMemoryTransferBackend(namespace="hydraserve-empty-race-pytest")
    backend.send("chunk", np.arange(16, dtype=np.float32), 1)
    original = transfer_backend_module.shared_memory.SharedMemory
    raced = False

    def open_after_ftruncate(*args, **kwargs):
        nonlocal raced
        if not kwargs.get("create", False) and not raced:
            raced = True
            raise ValueError("cannot mmap an empty file")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        transfer_backend_module.shared_memory,
        "SharedMemory",
        open_after_ftruncate,
    )
    try:
        restored = backend.receive("chunk", 1, timeout=1)
    finally:
        backend.close()
    assert raced
    np.testing.assert_array_equal(restored, np.arange(16, dtype=np.float32))


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


def test_ring_demultiplexes_other_requests_to_break_head_of_line_blocking() -> None:
    namespace = "hydraserve-ring-demux-pytest"
    sender = SharedMemoryRingTransferBackend(
        namespace=namespace,
        slots=2,
        slot_bytes=1 << 20,
        send_timeout_s=2,
    )
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    producer_error = []

    def send_a_after_b_fills_ring() -> None:
        try:
            sender.send("a", {"request": "a"}, 1)
        except Exception as exc:  # pragma: no cover - asserted below
            producer_error.append(exc)

    try:
        sender.send("b-0", {"sequence": 0}, 1)
        sender.send("b-1", {"sequence": 1}, 1)
        producer = Thread(target=send_a_after_b_fills_ring)
        producer.start()

        # Waiting for A must drain B's two READY slots into the receiver-side
        # mailbox, allowing A's blocked producer to make progress.
        assert receiver.receive("a", 1, timeout=1) == {"request": "a"}
        producer.join(timeout=1)
        assert not producer.is_alive()
        assert not producer_error
        assert receiver.receive("b-0", 1, timeout=1) == {"sequence": 0}
        assert receiver.receive("b-1", 1, timeout=1) == {"sequence": 1}
    finally:
        receiver.close()
        sender.close()


def test_ring_segments_payloads_larger_than_one_slot() -> None:
    namespace = "hydraserve-ring-segment-pytest"
    sender = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=4096, send_timeout_s=2
    )
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=4096
    )
    payload = {
        "values": np.arange(5000, dtype=np.float32),
        "request": 19,
    }
    errors = []

    def send() -> None:
        try:
            sender.send("oversized", payload, 1)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    producer = Thread(target=send)
    producer.start()
    try:
        restored = receiver.receive("oversized", 1, timeout=2)
        producer.join(1)
        assert not producer.is_alive()
        assert not errors
        assert restored["request"] == 19
        np.testing.assert_array_equal(restored["values"], payload["values"])
    finally:
        receiver.close()
        sender.close()


def test_ring_releases_slot_when_decode_raises() -> None:
    namespace = "hydraserve-ring-corrupt-release-pytest"
    sender = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=1, slot_bytes=1 << 20, send_timeout_s=1
    )
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=1, slot_bytes=1 << 20
    )
    original_decode = receiver._decode

    def fail_decode(_metadata, _payload):
        raise RuntimeError("synthetic decode failure")

    try:
        receiver._decode = fail_decode
        sender.send("bad", np.arange(8), 1)
        with pytest.raises(RuntimeError, match="synthetic decode failure"):
            receiver.receive("bad", 1, timeout=1)

        # A one-slot ring proves the failed receive returned the slot to FREE.
        receiver._decode = original_decode
        sender.send("good", np.arange(8), 1)
        np.testing.assert_array_equal(
            receiver.receive("good", 1, timeout=1), np.arange(8)
        )
    finally:
        receiver.close()
        sender.close()


def test_ring_dispatcher_drains_late_segments_for_cancelled_key() -> None:
    namespace = "hydraserve-ring-cancel-drain-pytest"
    sender = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20, send_timeout_s=1
    )
    receiver = SharedMemoryRingTransferBackend(
        namespace=namespace, slots=2, slot_bytes=1 << 20
    )
    errors = []

    def send_late_chunks() -> None:
        try:
            for sequence in range(10):
                sender.send("cancelled-chunk", {"sequence": sequence}, 1)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    try:
        receiver.discard("cancelled-chunk", 1)
        producer = Thread(target=send_late_chunks)
        producer.start()
        producer.join(2)
        assert not producer.is_alive()
        assert not errors
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
