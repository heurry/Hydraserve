from __future__ import annotations

from threading import Event
from time import monotonic, sleep

import pytest

from hydraserve.engine import (
    AdmissionDecision,
    BackendCapacity,
    ContinuousGenerationLoop,
    OverloadedError,
    PartialDecodeError,
    SamplingParams,
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
        @staticmethod
        def decode_batch_sizes(requests):
            return {request.request_id: 1 for request in requests}

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
    assert all(
        event.decode_batch_size == 1
        for event in (*first_events, *second_events)
        if event.decode_batch_size is not None
    )
    assert backend.live == set()


def test_unified_step_token_budget_limits_prefill_and_decode_width() -> None:
    backend = FakeBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=8, max_step_tokens=1)
    first = loop.submit([1], max_new_tokens=3)
    second = loop.submit([10], max_new_tokens=3)
    list(first)
    list(second)
    loop.close()
    assert backend.decode_batches
    assert all(len(batch) <= 1 for batch in backend.decode_batches)


def test_eos_stops_without_extra_decode() -> None:
    backend = FakeBackend()
    loop = ContinuousGenerationLoop(backend, eos_token_id=3)
    handle = loop.submit([1, 2], max_new_tokens=10)
    events = _collect(handle)
    loop.close()
    assert [event.token_id for event in events[:-1]] == [3]
    assert events[-1].finish_reason == "stop"
    assert backend.decode_batches == []


def test_multi_token_stop_sequence_finishes_at_exact_suffix() -> None:
    backend = FakeBackend()
    loop = ContinuousGenerationLoop(backend)
    handle = loop.submit(
        [63],
        max_new_tokens=10,
        sampling_params=SamplingParams(stop_token_sequences=((65, 66),)),
    )
    events = _collect(handle)
    loop.close()
    assert [event.token_id for event in events[:-1]] == [64, 65, 66]
    assert events[-1].finish_reason == "stop"


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


def test_slow_release_does_not_block_other_request_decode() -> None:
    first_release_entered = Event()
    allow_first_release = Event()

    class SlowReleaseBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.blocked_once = False

        def release(self, request_id):
            if not self.blocked_once:
                self.blocked_once = True
                first_release_entered.set()
                assert allow_first_release.wait(2)
            super().release(request_id)

    backend = SlowReleaseBackend()
    loop = ContinuousGenerationLoop(backend)
    first = loop.submit([1], max_new_tokens=1)
    assert first.get(timeout=2).token_id == 2
    assert first_release_entered.wait(2)

    # Cleanup for the first request is still blocked, but it must not hold the
    # generation thread or prevent a different request from reaching decode.
    second = loop.submit([10], max_new_tokens=2)
    assert second.get(timeout=1).token_id == 11
    allow_first_release.set()
    assert list(first)[-1].finish_reason == "length"
    assert list(second)[-1].finish_reason == "length"
    loop.close()
    assert loop.release_total == 2
    assert loop.release_failures_total == 0
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
                healthy = min(requests, key=lambda request: request.request_id)
                failed = max(requests, key=lambda request: request.request_id)
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
            if request.token_ids == tuple(range(20)):
                second_prefill_entered.set()
                assert allow_second_prefill.wait(2)
            return super().prefill(request)

        @staticmethod
        def prefill_admission_tokens(request):
            return min(len(request.token_ids), 4)

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
    loop = ContinuousGenerationLoop(backend, max_batch_size=2, max_step_tokens=8)
    first = loop.submit([1], max_new_tokens=4)
    assert first.get(timeout=2).token_id == 2
    assert first_decode_entered.wait(2)
    second = loop.submit(tuple(range(20)), max_new_tokens=2)
    allow_first_decode.set()
    first_events = list(first)
    second_events = list(second)
    loop.close()
    assert second_prefill_entered.is_set()
    assert [event.token_id for event in first_events[:-1]] == [3, 4, 5]
    assert [event.token_id for event in second_events[:-1]] == [20, 21]


def test_single_async_executor_caps_admission_before_backend_reserve() -> None:
    first_prefill_entered = Event()
    allow_first_prefill = Event()

    class SinglePoolBackend(FakeBackend):
        supports_async_prefill = True
        prefill_parallelism = 1

        def __init__(self):
            super().__init__()
            self.admitted = []

        def admit(self, request):
            self.admitted.append(request.request_id)
            return AdmissionDecision.accept()

        def prefill(self, request):
            if not first_prefill_entered.is_set():
                first_prefill_entered.set()
                assert allow_first_prefill.wait(2)
            return super().prefill(request)

    backend = SinglePoolBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=2)
    first = loop.submit([1], max_new_tokens=1)
    assert first_prefill_entered.wait(2)
    second = loop.submit([2], max_new_tokens=1)

    deadline = monotonic() + 1
    while loop.prefill_slot_deferrals_total == 0 and monotonic() < deadline:
        sleep(0.005)
    assert loop.prefill_slot_deferrals_total > 0
    assert second.request_id not in backend.admitted

    allow_first_prefill.set()
    assert list(first)[-1].finish_reason == "length"
    assert list(second)[-1].finish_reason == "length"
    loop.close()


def test_route_aware_prefill_executors_prevent_long_short_host_hol() -> None:
    long_prefill_entered = Event()
    allow_long_prefill = Event()

    class SplitExecutorBackend(FakeBackend):
        supports_async_prefill = True
        prefill_executor_limits = {"prefill": 1, "decode": 1}

        def __init__(self):
            super().__init__()
            self.admitted = []

        def admit(self, request):
            self.admitted.append(request.request_id)
            request.route = (
                "pd_disaggregated"
                if request.token_ids in {(20,), (21,)}
                else "collocated"
            )
            return AdmissionDecision.accept()

        @staticmethod
        def prefill_executor_group_hint(request):
            return "prefill" if request.token_ids[0] >= 20 else "decode"

        @staticmethod
        def prefill_executor_group(request):
            return "prefill" if request.route == "pd_disaggregated" else "decode"

        def prefill(self, request):
            if request.route == "pd_disaggregated":
                long_prefill_entered.set()
                assert allow_long_prefill.wait(2)
            return super().prefill(request)

    backend = SplitExecutorBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=4)
    long_handle = loop.submit([20], max_new_tokens=1)
    assert long_prefill_entered.wait(2)

    queued_long = loop.submit([21], max_new_tokens=1)
    short_handle = loop.submit([1], max_new_tokens=1)
    short_first = short_handle.get(timeout=1)
    assert short_first.token_id == 2
    assert list(short_handle)[-1].finish_reason == "length"
    assert queued_long.request_id not in backend.admitted
    assert loop.prefill_slot_deferrals_total > 0

    allow_long_prefill.set()
    assert list(long_handle)[-1].finish_reason == "length"
    assert list(queued_long)[-1].finish_reason == "length"
    loop.close()


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


def test_deferred_large_request_does_not_block_admissible_request() -> None:
    class SelectiveAdmissionBackend(FakeBackend):
        def admit(self, request):
            if request.token_ids == (9,):
                return AdmissionDecision.defer("large request is waiting for capacity")
            return AdmissionDecision.accept()

    backend = SelectiveAdmissionBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    deferred = loop.submit([9], max_new_tokens=1)
    healthy = loop.submit([2], max_new_tokens=1)
    assert [event.token_id for event in _collect(healthy)[:-1]] == [3]
    deferred.cancel()
    assert _collect(deferred)[-1].finish_reason == "cancelled"
    loop.close()
    assert loop.pending_count == 0


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

        def allocate(
            self, request_id, num_tokens, *, reserve_tokens=None, token_ids=None
        ):
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


def test_runtime_backend_recovery_replays_only_model_consumed_tokens() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from hydraserve.cache import KVBlockManager

    class FakePagedCache:
        def __init__(self):
            self.block_manager = KVBlockManager(8, block_size=4)

        def allocate(
            self, request_id, num_tokens, *, reserve_tokens=None, token_ids=None
        ):
            return self.block_manager.allocate(
                request_id, num_tokens, reserve_tokens=reserve_tokens
            )

        def free(self, request_id):
            self.block_manager.free(request_id)

        def publish_prefix(self, request_id, token_ids):
            return None

    class RecordingRuntime:
        device = torch.device("cpu")
        config = None

        def __init__(self):
            self.prefill_inputs = []

        def prefill(self, input_ids, **kwargs):
            self.prefill_inputs.append(tuple(input_ids[0].tolist()))
            logits = torch.zeros(1, input_ids.shape[1], 32)
            return logits, SimpleNamespace(sequence_length=input_ids.shape[1])

    runtime = RecordingRuntime()
    cache = FakePagedCache()
    backend = RuntimeGenerationBackend(runtime, cache, max_state_slots=1)
    request = ServingRequest(
        7,
        (1, 2, 3),
        6,
        generated_token_ids=[9, 10, 11],
    )
    assert backend.admit(request).admitted
    backend.states[request.request_id] = SimpleNamespace(sequence_length=3)

    backend.preempt(request.request_id)
    assert backend.capacity().state_free_slots == 1
    decision = backend.recover(request)

    assert decision.admitted
    assert runtime.prefill_inputs == [(1, 2, 3, 9, 10)]
    allocation = cache.block_manager.get(request.request_id)
    assert allocation.num_tokens == 5
    assert allocation.reserved_tokens == 8
    assert backend.states[request.request_id].sequence_length == 5
    backend.release(request.request_id)
    assert cache.block_manager.num_free_blocks == cache.block_manager.num_blocks


def test_backend_capacity_uses_most_constrained_resource() -> None:
    capacity = BackendCapacity(100, 80, 10, 2)
    assert capacity.decode_load == pytest.approx(0.8)
    assert capacity.has_request_slot


def test_prefill_queue_prediction_uses_running_future_remaining_work() -> None:
    from time import monotonic

    class RunningFuture:
        def done(self):
            return False

        def running(self):
            return True

    request = ServingRequest(99, (1,), 1)
    request.route = "collocated"
    request.route_collocated_cost_ms = 100.0
    request.prefill_started_at = monotonic() - 0.04
    predicted = ContinuousGenerationLoop._prefill_queue_ahead_ms(
        {99: (request, RunningFuture())}
    )
    assert 40.0 <= predicted <= 70.0


def test_request_deadline_expires_while_waiting_for_admission() -> None:
    class NeverAdmit(FakeBackend):
        def admit(self, request):
            return AdmissionDecision.defer("full")

    loop = ContinuousGenerationLoop(NeverAdmit(), idle_wait_s=0.001)
    handle = loop.submit([1], max_new_tokens=1, timeout_ms=10)
    terminal = handle.get(timeout=1)
    loop.close()
    assert terminal.finished
    assert terminal.finish_reason == "error"
    assert "deadline expired before admission" in terminal.error


def test_deadline_does_not_emit_decode_token_completed_after_expiry() -> None:
    from time import sleep

    class SlowDecode(FakeBackend):
        def decode(self, requests):
            sleep(0.03)
            return super().decode(requests)

    loop = ContinuousGenerationLoop(SlowDecode(), max_batch_size=1)
    handle = loop.submit([1], max_new_tokens=2, timeout_ms=15)
    events = list(handle)
    loop.close()
    assert [event.token_id for event in events if event.token_id is not None] == [2]
    assert "deadline expired during decode" in events[-1].error


def test_active_limit_can_exceed_decode_batch_and_is_observable() -> None:
    first_prefill_entered = Event()
    release_first_prefill = Event()
    decode_entered = Event()
    release_decode = Event()

    class BlockingDecode(FakeBackend):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def prefill(self, request):
            if request.request_id == 0:
                first_prefill_entered.set()
                assert release_first_prefill.wait(2)
            return super().prefill(request)

        def decode(self, requests):
            self.calls += 1
            if self.calls == 2:
                decode_entered.set()
                assert release_decode.wait(2)
            return super().decode(requests)

    loop = ContinuousGenerationLoop(
        BlockingDecode(), max_batch_size=2, max_active_requests=4
    )
    handles = [loop.submit([1], max_new_tokens=3)]
    assert first_prefill_entered.wait(1)
    handles.extend(loop.submit([index + 1], max_new_tokens=3) for index in range(1, 4))
    release_first_prefill.set()
    assert decode_entered.wait(1)
    assert loop.active_count == 4
    release_decode.set()
    assert all(list(handle)[-1].finish_reason == "length" for handle in handles)
    loop.close()
    assert loop.active_count == 0


def test_higher_priority_request_preempts_and_exactly_recovers_active_request() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class PreemptibleBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.decode_calls = 0
            self.preempted = []
            self.replayed = []

        def decode(self, requests):
            self.decode_calls += 1
            if self.decode_calls == 1:
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            self.preempted.append(request_id)
            self.release(request_id)

        def recover(self, request):
            replay = request.token_ids + tuple(request.generated_token_ids[:-1])
            self.replayed.append((request.request_id, replay))
            self.live.add(request.request_id)
            return AdmissionDecision.accept()

    backend = PreemptibleBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1, max_active_requests=1)
    background = loop.submit([1], max_new_tokens=4, priority=0)
    assert background.get(timeout=2).token_id == 2
    assert decode_entered.wait(2)
    urgent = loop.submit([10], max_new_tokens=1, priority=7)
    allow_decode.set()

    assert [event.token_id for event in urgent][:-1] == [11]
    background_events = list(background)
    loop.close()

    assert [event.token_id for event in background_events[:-1]] == [3, 4, 5]
    assert backend.preempted == [background.request_id]
    assert backend.replayed == [(background.request_id, (1, 2))]
    assert background.request.preemption_count == 1
    assert background.request.recovery_count == 1
    assert loop.preemptions_total == 1
    assert loop.recoveries_total == 1
    assert loop.recovery_failures_total == 0
    assert loop.preempted_count == 0
    assert backend.live == set()


def test_same_priority_deadline_can_preempt_request_without_deadline() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class PreemptibleBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.blocked = False
            self.preempted = []

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            self.preempted.append(request_id)
            self.release(request_id)

        def recover(self, request):
            self.live.add(request.request_id)
            return AdmissionDecision.accept()

    backend = PreemptibleBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    background = loop.submit([1], max_new_tokens=3)
    background.get(timeout=2)
    assert decode_entered.wait(2)
    urgent = loop.submit([20], max_new_tokens=1, timeout_ms=1000)
    allow_decode.set()
    assert list(urgent)[-1].finish_reason == "length"
    assert list(background)[-1].finish_reason == "length"
    loop.close()
    assert backend.preempted == [background.request_id]


def test_recovery_failure_is_request_scoped_and_releases_resources() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class FailedRecoveryBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.blocked = False

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            self.release(request_id)

        def recover(self, request):
            self.live.add(request.request_id)
            raise RuntimeError("recompute kernel failed")

        def release(self, request_id):
            self.live.discard(request_id)

    backend = FailedRecoveryBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    background = loop.submit([1], max_new_tokens=3)
    background.get(timeout=2)
    assert decode_entered.wait(2)
    urgent = loop.submit([10], max_new_tokens=1, priority=7)
    allow_decode.set()
    assert list(urgent)[-1].finish_reason == "length"
    terminal = list(background)[-1]
    loop.close()
    assert terminal.finish_reason == "error"
    assert "recompute kernel failed" in terminal.error
    assert loop.recovery_failures_total == 1
    assert backend.live == set()


def test_async_prefill_path_preempts_and_recovers_without_duplicate_output() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class AsyncPreemptibleBackend(FakeBackend):
        supports_async_prefill = True

        def __init__(self):
            super().__init__()
            self.blocked = False
            self.preempted = []
            self.replayed = []

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            self.preempted.append(request_id)
            self.release(request_id)

        def recover(self, request):
            replay = request.token_ids + tuple(request.generated_token_ids[:-1])
            self.replayed.append((request.request_id, replay))
            self.live.add(request.request_id)
            return AdmissionDecision.accept()

    backend = AsyncPreemptibleBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    background = loop.submit([1], max_new_tokens=4)
    first = background.get(timeout=2)
    assert first.token_id == 2
    assert decode_entered.wait(2)
    urgent = loop.submit([20], max_new_tokens=1, priority=7)
    allow_decode.set()

    assert list(urgent)[-1].finish_reason == "length"
    background_events = [first, *list(background)]
    loop.close()
    assert [event.token_id for event in background_events[:-1]] == [2, 3, 4, 5]
    assert backend.preempted == [background.request_id]
    assert backend.replayed == [(background.request_id, (1, 2))]
    assert loop.preemptions_total == 1
    assert loop.recoveries_total == 1
    assert loop.preempted_count == 0
    assert backend.live == set()


def test_preemption_failure_fails_only_victim_and_still_admits_urgent_work() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class BrokenPreemptionBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.blocked = False

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            raise RuntimeError("release RPC timed out")

        def recover(self, request):
            raise AssertionError("failed preemption must not enter recovery")

    backend = BrokenPreemptionBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    background = loop.submit([1], max_new_tokens=3)
    background.get(timeout=2)
    assert decode_entered.wait(2)
    urgent = loop.submit([10], max_new_tokens=1, priority=7)
    allow_decode.set()

    assert list(urgent)[-1].finish_reason == "length"
    terminal = list(background)[-1]
    loop.close()
    assert terminal.finish_reason == "error"
    assert "preemption failed: release RPC timed out" in terminal.error
    assert loop.preemption_failures_total == 1
    assert loop.preemptions_total == 0
    assert loop.recoveries_total == 0
    assert backend.live == set()


def test_async_recovery_retries_transient_capacity_deferral() -> None:
    decode_entered = Event()
    allow_decode = Event()

    class DeferredRecoveryBackend(FakeBackend):
        supports_async_prefill = True

        def __init__(self):
            super().__init__()
            self.blocked = False
            self.recovery_attempts = 0

        def decode(self, requests):
            if not self.blocked:
                self.blocked = True
                decode_entered.set()
                assert allow_decode.wait(2)
            return super().decode(requests)

        def preempt(self, request_id):
            self.release(request_id)

        def recover(self, request):
            self.recovery_attempts += 1
            if self.recovery_attempts == 1:
                return AdmissionDecision.defer("worker is still draining")
            self.live.add(request.request_id)
            return AdmissionDecision.accept()

    backend = DeferredRecoveryBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    background = loop.submit([1], max_new_tokens=3)
    first = background.get(timeout=2)
    assert decode_entered.wait(2)
    urgent = loop.submit([10], max_new_tokens=1, priority=7)
    allow_decode.set()

    assert list(urgent)[-1].finish_reason == "length"
    assert [first, *list(background)][-1].finish_reason == "length"
    loop.close()
    assert backend.recovery_attempts == 2
    assert loop.recoveries_total == 1
    assert loop.recovery_failures_total == 0
    assert loop.preempted_count == 0


def test_recoverable_worker_loss_replays_without_failing_client_request() -> None:
    class RecoveringWorkerBackend(FakeBackend):
        supports_async_prefill = True

        def __init__(self):
            super().__init__()
            self.failed = False
            self.recovery_attempts = 0
            self.abandoned = []

        def decode(self, requests):
            if not self.failed:
                self.failed = True
                raise PartialDecodeError(
                    {},
                    {
                        request.request_id: RuntimeError("decode worker state lost")
                        for request in requests
                    },
                )
            return super().decode(requests)

        def is_recoverable_decode_error(self, request_id, error):
            return "worker state lost" in str(error)

        def abandon(self, request_id):
            self.abandoned.append(request_id)
            self.live.discard(request_id)

        def recover(self, request):
            self.recovery_attempts += 1
            if self.recovery_attempts == 1:
                return AdmissionDecision.defer("replacement worker is starting")
            self.live.add(request.request_id)
            return AdmissionDecision.accept()

    backend = RecoveringWorkerBackend()
    loop = ContinuousGenerationLoop(backend, max_batch_size=1)
    handle = loop.submit([1], max_new_tokens=4)
    events = list(handle)
    loop.close()

    assert [event.token_id for event in events[:-1]] == [2, 3, 4, 5]
    assert events[-1].finish_reason == "length"
    assert backend.abandoned == [handle.request_id]
    assert backend.recovery_attempts == 2
    assert loop.fault_suspensions_total == 1
    assert loop.preemptions_total == 0
    assert loop.recoveries_total == 1
    assert backend.live == set()
