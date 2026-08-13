from __future__ import annotations

from threading import Event

from hydraserve.engine import ContinuousGenerationLoop


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
