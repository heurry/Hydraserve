from __future__ import annotations

from threading import Event

import pytest

from hydraserve.engine import (
    AdmissionDecision,
    BackendCapacity,
    ContinuousGenerationLoop,
    OverloadedError,
    PartialDecodeError,
    RuntimeGenerationBackend,
    ServingRequest,
)


class FakeBackend:
    def __init__(self) -> None:
        self.live: set[int] = set()
        self.decode_batches: list[tuple[int, ...]] = []

    def prefill(self, request) -> int:
        self.live.add(request.request_id)
        return request.token_ids[-1] + 1

    def decode(self, requests) -> tuple[int, ...]:
        self.decode_batches.append(tuple(request.request_id for request in requests))
        return tuple(request.generated_token_ids[-1] + 1 for request in requests)

    def release(self, request_id: int) -> None:
        self.live.remove(request_id)


def _collect(handle):
    return list(handle)


def test_streams_prefill_seed_then_continuous_decode() -> None:
    first_decode = Event()
    allow_decode = Event()

    class AdmissionBackend(FakeBackend):
        def decode(self, requests):
            if not first_decode.is_set():
                first_decode.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

    backend = AdmissionBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=8)
    first = loop.submit([1, 2], max_new_tokens=4)
    assert first_decode.wait(2)
    second = loop.submit([10], max_new_tokens=2)
    allow_decode.set()
    first_events = _collect(first)
    second_events = _collect(second)
    loop.close()
    assert [event.token_id for event in first_events[:-1]] == [3, 4, 5, 6]
    assert [event.token_id for event in second_events[:-1]] == [11, 12]
    assert first_events[-1].finish_reason == "length"
    assert second_events[-1].finish_reason == "length"
    assert any(len(batch) == 2 for batch in backend.decode_batches)
    assert backend.live == set()


def test_eos_stops_without_extra_decode() -> None:
    backend = FakeBackend()
    loop = ContinuousGenerationLoop(backend, eos_token_id=3)
    handle = loop.submit([1, 2], max_new_tokens=10)
    events = _collect(handle)
    loop.close()
    assert [event.token_id for event in events[:-1]] == [3]
    assert events[-1].finish_reason == "stop"
    assert backend.decode_batches == []


def test_cancel_active_request_releases_backend() -> None:
    entered_decode = Event()
    continue_decode = Event()

    class BlockingBackend(FakeBackend):
        def decode(self, requests):
            entered_decode.set()
            assert continue_decode.wait(2)
            return super().decode(requests)

    backend = BlockingBackend()
    loop = ContinuousGenerationLoop(backend)
    handle = loop.submit([5], max_new_tokens=100)
    first = handle.get(timeout=2)
    assert first.token_id == 6
    assert entered_decode.wait(2)
    handle.cancel()
    continue_decode.set()
    events = list(handle)
    loop.close()
    assert events[-1].finished
    assert events[-1].finish_reason == "cancelled"
    assert backend.live == set()


def test_decode_failure_finishes_entire_affected_batch() -> None:
    class FailingBackend(FakeBackend):
        def decode(self, requests):
            raise RuntimeError("decode exploded")

    backend = FailingBackend()
    loop = ContinuousGenerationLoop(backend)
    first = loop.submit([1], max_new_tokens=2)
    second = loop.submit([2], max_new_tokens=2)
    first_events = _collect(first)
    second_events = _collect(second)
    loop.close()
    assert first_events[-1].finish_reason == "error"
    assert second_events[-1].finish_reason == "error"
    assert "decode exploded" in first_events[-1].error
    assert backend.live == set()


def test_partial_decode_failure_preserves_healthy_requests() -> None:
    first_decode = Event()
    allow_decode = Event()

    class PartiallyFailingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.failed_once = False
            self.blocked_once = False

        def decode(self, requests):
            if not self.blocked_once:
                self.blocked_once = True
                first_decode.set()
                assert allow_decode.wait(2)
            if not self.failed_once and len(requests) == 2:
                self.failed_once = True
                healthy, failed = requests
                raise PartialDecodeError(
                    {healthy.request_id: healthy.generated_token_ids[-1] + 1},
                    {failed.request_id: RuntimeError("bound worker failed")},
                )
            return super().decode(requests)

    backend = PartiallyFailingBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    healthy = loop.submit([1], max_new_tokens=3)
    assert first_decode.wait(2)
    failed = loop.submit([10], max_new_tokens=3)
    allow_decode.set()
    healthy_events = _collect(healthy)
    failed_events = _collect(failed)
    loop.close()
    assert [event.token_id for event in healthy_events[:-1]] == [2, 3, 4]
    assert healthy_events[-1].finish_reason == "length"
    assert failed_events[-1].finish_reason == "error"
    assert "bound worker failed" in failed_events[-1].error
    assert backend.live == set()


def test_prefill_failure_does_not_kill_loop() -> None:
    class SelectiveBackend(FakeBackend):
        def prefill(self, request):
            if request.token_ids == (0,):
                raise RuntimeError("bad prompt")
            return super().prefill(request)

        def release(self, request_id):
            if request_id in self.live:
                super().release(request_id)

    backend = SelectiveBackend()
    loop = ContinuousGenerationLoop(backend)
    failed = loop.submit([0], max_new_tokens=2)
    healthy = loop.submit([4], max_new_tokens=1)
    assert _collect(failed)[-1].finish_reason == "error"
    assert [event.token_id for event in _collect(healthy)[:-1]] == [5]
    loop.close()


def test_disaggregated_prefill_overlaps_active_decode() -> None:
    first_decode_entered = Event()
    allow_first_decode = Event()
    second_prefill_entered = Event()
    allow_second_prefill = Event()

    class AsyncBackend(FakeBackend):
        supports_async_prefill = True

        def __init__(self):
            super().__init__()
            self.decode_calls = 0

        def prefill(self, request):
            if request.token_ids == (20,):
                second_prefill_entered.set()
                assert allow_second_prefill.wait(2)
            return super().prefill(request)

        def decode(self, requests):
            self.decode_calls += 1
            if self.decode_calls == 1:
                first_decode_entered.set()
                assert allow_first_decode.wait(2)
            elif self.decode_calls == 2:
                assert second_prefill_entered.wait(2)
            result = super().decode(requests)
            if second_prefill_entered.is_set():
                allow_second_prefill.set()
            return result

    backend = AsyncBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    first = loop.submit([1], max_new_tokens=4)
    assert first.get(timeout=2).token_id == 2
    assert first_decode_entered.wait(2)
    second = loop.submit([20], max_new_tokens=2)
    allow_first_decode.set()
    first_events = list(first)
    second_events = list(second)
    loop.close()
    assert second_prefill_entered.is_set()
    assert [event.token_id for event in first_events[:-1]] == [3, 4, 5]
    assert [event.token_id for event in second_events[:-1]] == [21, 22]


def test_retryable_admission_waits_for_capacity_instead_of_failing() -> None:
    class CapacityBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.reserved: set[int] = set()

        def admit(self, request):
            if request.request_id in self.reserved:
                return AdmissionDecision.accept()
            if self.reserved:
                return AdmissionDecision.defer("worker is full")
            self.reserved.add(request.request_id)
            return AdmissionDecision.accept()

        def release(self, request_id):
            super().release(request_id)
            self.reserved.remove(request_id)

    backend = CapacityBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    first = loop.submit([1], max_new_tokens=2)
    second = loop.submit([10], max_new_tokens=1)
    assert [event.token_id for event in _collect(first)[:-1]] == [2, 3]
    assert [event.token_id for event in _collect(second)[:-1]] == [11]
    loop.close()
    assert loop.pending_count == 0
    assert backend.reserved == set()


def test_permanent_admission_rejection_is_request_scoped() -> None:
    class RejectLargeBackend(FakeBackend):
        def admit(self, request):
            if len(request.token_ids) > 2:
                return AdmissionDecision.reject("request exceeds KV capacity")
            return AdmissionDecision.accept()

    backend = RejectLargeBackend()
    loop = ContinuousGenerationLoop(backend)
    rejected = loop.submit([1, 2, 3], max_new_tokens=1)
    healthy = loop.submit([4], max_new_tokens=1)
    assert _collect(rejected)[-1].error == "request exceeds KV capacity"
    assert [event.token_id for event in _collect(healthy)[:-1]] == [5]
    loop.close()


def test_bounded_admission_queue_applies_backpressure() -> None:
    entered_prefill = Event()
    allow_prefill = Event()

    class BlockingPrefillBackend(FakeBackend):
        def prefill(self, request):
            if request.token_ids == (1,):
                entered_prefill.set()
                assert allow_prefill.wait(2)
            return super().prefill(request)

    backend = BlockingPrefillBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1, max_queue_size=1)
    first = loop.submit([1], max_new_tokens=1)
    assert entered_prefill.wait(2)
    second = loop.submit([2], max_new_tokens=1)
    with pytest.raises(OverloadedError, match="request limit"):
        loop.submit([3], max_new_tokens=1)
    allow_prefill.set()
    assert [event.token_id for event in _collect(first)[:-1]] == [2]
    assert [event.token_id for event in _collect(second)[:-1]] == [3]
    loop.close()


def test_runtime_admission_reserves_kv_and_recurrent_state_together() -> None:
    from hydraserve.cache import KVBlockManager

    class FakePagedCache:
        def __init__(self):
            self.block_manager = KVBlockManager(8, block_size=4)

        def allocate(self, request_id, num_tokens, *, reserve_tokens=None, token_ids=None):
            return self.block_manager.allocate(
                request_id, num_tokens, reserve_tokens=reserve_tokens
            )

        def free(self, request_id):
            self.block_manager.free(request_id)

    cache = FakePagedCache()
    backend = RuntimeGenerationBackend(object(), cache, max_state_slots=1)
    first = ServingRequest(1, (1, 2, 3), 6)
    second = ServingRequest(2, (4,), 1)
    assert backend.admit(first).admitted
    allocation = cache.block_manager.get(1)
    assert allocation.num_tokens == 3
    assert allocation.reserved_tokens == 8
    assert backend.capacity() == BackendCapacity(8, 6, 1, 0)
    assert backend.capacity().decode_load == 1.0
    deferred = backend.admit(second)
    assert not deferred.admitted and deferred.retryable
    assert "state" in deferred.reason
    backend.release(1)
    assert backend.admit(second).admitted
    backend.release(2)
    assert cache.block_manager.num_free_blocks == cache.block_manager.num_blocks


def test_runtime_decode_bisects_failure_and_rolls_back_failed_request() -> None:
    torch = pytest.importorskip("torch")
    from dataclasses import dataclass
    from hydraserve.cache import KVBlockManager

    @dataclass
    class State:
        sequence_length: int

    class FakePagedCache:
        def __init__(self):
            self.block_manager = KVBlockManager(8, block_size=4)

        def free(self, request_id):
            self.block_manager.free(request_id)

    class SelectiveRuntime:
        device = torch.device("cpu")

        def decode_batch(self, input_ids, states, paged_cache, request_ids):
            if 2 in request_ids:
                for state in states:
                    state.sequence_length += 100
                raise RuntimeError("request 2 kernel failure")
            for state in states:
                state.sequence_length += 1
            logits = torch.zeros(len(request_ids), 1, 16)
            for row, token in enumerate(input_ids[:, 0]):
                logits[row, 0, int(token) + 1] = 1
            return logits, states

    cache = FakePagedCache()
    backend = RuntimeGenerationBackend(SelectiveRuntime(), cache, max_state_slots=2)
    requests = (
        ServingRequest(1, (1,), 3, generated_token_ids=[4]),
        ServingRequest(2, (2,), 3, generated_token_ids=[7]),
    )
    for request in requests:
        cache.block_manager.allocate(request.request_id, 1, reserve_tokens=3)
        backend.state_slots.allocate(request.request_id)
        backend.states[request.request_id] = State(1)

    with pytest.raises(PartialDecodeError) as raised:
        backend.decode(requests)
    assert raised.value.token_ids == {1: 5}
    assert set(raised.value.errors) == {2}
    assert backend.states[1].sequence_length == 2
    assert backend.states[2].sequence_length == 1
    assert cache.block_manager.get(1).num_tokens == 2
    assert cache.block_manager.get(2).num_tokens == 1
    backend.release(1)
    backend.release(2)


def test_backend_capacity_uses_most_constrained_resource() -> None:
    capacity = BackendCapacity(100, 80, 10, 2)
    assert capacity.decode_load == pytest.approx(0.8)
    assert capacity.has_request_slot
