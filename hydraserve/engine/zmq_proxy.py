"""GIL-independent ZeroMQ ROUTER/DEALER broker primitives for DP serving."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorkerLoad:
    pending: int = 0
    served: int = 0
    ready_wave: int = 0


class ZMQWaveRouter:
    """Load-aware worker selection with optional decode-wave synchronization."""

    def __init__(self, worker_ids, *, synchronize_waves: bool = True) -> None:
        identities = tuple(bytes(worker) for worker in worker_ids)
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("worker identities must be non-empty and unique")
        self.workers = {identity: WorkerLoad() for identity in identities}
        self.synchronize_waves = synchronize_waves
        self.wave = 0

    def mark_ready(self, worker_id: bytes, wave: int) -> None:
        self.workers[worker_id].ready_wave = max(
            self.workers[worker_id].ready_wave, int(wave)
        )

    def choose(self) -> bytes | None:
        candidates = [
            (load.pending, load.served, worker_id)
            for worker_id, load in self.workers.items()
            if not self.synchronize_waves or load.ready_wave >= self.wave
        ]
        if not candidates:
            return None
        _, _, worker_id = min(candidates)
        load = self.workers[worker_id]
        load.pending += 1
        load.served += 1
        if self.synchronize_waves and all(
            item.served > self.wave for item in self.workers.values()
        ):
            self.wave += 1
        return worker_id

    def complete(self, worker_id: bytes) -> None:
        load = self.workers[worker_id]
        load.pending = max(0, load.pending - 1)


def run_zmq_broker(frontend: str, backend: str) -> None:
    """Run a zero-copy ROUTER/DEALER proxy; pyzmq is loaded only on use."""
    try:
        import zmq
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install HydraServe with the 'serve' extra for ZMQ") from exc
    context = zmq.Context.instance()
    clients = context.socket(zmq.ROUTER)
    workers = context.socket(zmq.DEALER)
    clients.bind(frontend)
    workers.bind(backend)
    try:
        zmq.proxy(clients, workers)
    finally:
        clients.close(linger=0)
        workers.close(linger=0)
