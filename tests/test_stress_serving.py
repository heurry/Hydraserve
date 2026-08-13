from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from hydraserve.engine import ContinuousGenerationLoop


class StressBackend:
    def __init__(self):
        self.live = set()
        self.lock = Lock()
        self.max_decode_batch = 0

    def prefill(self, request):
        with self.lock:
            self.live.add(request.request_id)
        return request.token_ids[-1] + 1

    def decode(self, requests):
        with self.lock:
            self.max_decode_batch = max(self.max_decode_batch, len(requests))
        return tuple(request.generated_token_ids[-1] + 1 for request in requests)

    def release(self, request_id):
        with self.lock:
            self.live.discard(request_id)


def test_concurrent_submit_cancel_and_continuous_batch_soak() -> None:
    backend = StressBackend()
    loop = ContinuousGenerationLoop(
        backend,
        max_batch_size=8,
        max_active_requests=32,
        max_queue_size=256,
        max_queue_tokens=100_000,
    )

    def submit(index):
        handle = loop.submit([index % 17 + 1], max_new_tokens=12)
        if index % 11 == 0:
            handle.cancel()
        return handle

    with ThreadPoolExecutor(max_workers=16) as executor:
        handles = list(executor.map(submit, range(128)))
        event_streams = list(executor.map(list, handles))
    loop.close()

    assert all(events and events[-1].finished for events in event_streams)
    assert all(events[-1].finish_reason in {"length", "cancelled"} for events in event_streams)
    assert loop.pending_count == 0
    assert loop.pending_tokens == 0
    assert backend.live == set()
    assert backend.max_decode_batch == 8


def test_repeated_loop_start_close_releases_every_request() -> None:
    for cycle in range(20):
        backend = StressBackend()
        loop = ContinuousGenerationLoop(backend, max_batch_size=4)
        handles = [loop.submit([cycle + 1], max_new_tokens=3) for _ in range(8)]
        assert all(list(handle)[-1].finish_reason == "length" for handle in handles)
        loop.close()
        assert backend.live == set()
