from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Mapping, Protocol


class CapacitySnapshot(Protocol):
    kv_total_blocks: int
    kv_free_blocks: int
    state_total_slots: int
    state_free_slots: int

    @property
    def decode_load(self) -> float: ...


@dataclass(frozen=True, slots=True)
class WorkerTopology:
    bandwidth_gbps: float = 1.0
    hops: int = 1

    def __post_init__(self) -> None:
        if self.bandwidth_gbps <= 0 or self.hops <= 0:
            raise ValueError("worker topology values must be positive")


@dataclass(frozen=True, slots=True)
class DecodeWorkerSnapshot:
    worker_id: int
    device: str
    capacity: CapacitySnapshot
    topology: WorkerTopology = WorkerTopology()
    active_requests: int = 0
    healthy: bool = True

    def __post_init__(self) -> None:
        if self.worker_id < 0 or not self.device or self.active_requests < 0:
            raise ValueError("invalid decode worker snapshot")


@dataclass(frozen=True, slots=True)
class WorkerSelection:
    worker_id: int
    score: float
    decode_load: float
    prefix_match_tokens: int


@dataclass(frozen=True, slots=True)
class WorkerScoringConfig:
    load_weight: float = 1.0
    topology_weight: float = 0.15
    affinity_weight: float = 0.75

    def __post_init__(self) -> None:
        if min(self.load_weight, self.topology_weight, self.affinity_weight) < 0:
            raise ValueError("worker scoring weights cannot be negative")


class DecodeWorkerRegistry:
    """Thread-safe worker inventory, scoring, and immutable request binding."""

    def __init__(
        self,
        workers: tuple[DecodeWorkerSnapshot, ...],
        *,
        scoring: WorkerScoringConfig | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one decode worker is required")
        if len({worker.worker_id for worker in workers}) != len(workers):
            raise ValueError("decode worker ids must be unique")
        self.scoring = scoring or WorkerScoringConfig()
        self._workers = {worker.worker_id: worker for worker in workers}
        self._bindings: dict[int, int] = {}
        self._lock = RLock()

    def update_capacity(self, worker_id: int, capacity: CapacitySnapshot) -> None:
        with self._lock:
            worker = self._worker(worker_id)
            self._workers[worker_id] = replace(worker, capacity=capacity)

    def set_health(self, worker_id: int, healthy: bool) -> None:
        with self._lock:
            worker = self._worker(worker_id)
            self._workers[worker_id] = replace(worker, healthy=healthy)

    def candidates(
        self,
        *,
        required_blocks: int,
        prompt_tokens: int,
        prefix_matches: Mapping[int, int] | None = None,
    ) -> tuple[WorkerSelection, ...]:
        if required_blocks <= 0 or prompt_tokens <= 0:
            raise ValueError("request resource demand must be positive")
        prefix_matches = prefix_matches or {}
        with self._lock:
            selections = []
            for worker in self._workers.values():
                capacity = worker.capacity
                if (
                    not worker.healthy
                    or capacity.state_free_slots <= 0
                    or capacity.kv_free_blocks < required_blocks
                ):
                    continue
                matched = max(0, int(prefix_matches.get(worker.worker_id, 0)))
                affinity = min(1.0, matched / prompt_tokens)
                topology_cost = (
                    worker.topology.hops / worker.topology.bandwidth_gbps
                )
                score = (
                    self.scoring.load_weight * capacity.decode_load
                    + self.scoring.topology_weight * topology_cost
                    - self.scoring.affinity_weight * affinity
                )
                selections.append(
                    WorkerSelection(
                        worker_id=worker.worker_id,
                        score=score,
                        decode_load=capacity.decode_load,
                        prefix_match_tokens=matched,
                    )
                )
            selections.sort(key=lambda item: (item.score, item.worker_id))
            return tuple(selections)

    def bind(self, request_id: int, worker_id: int) -> None:
        with self._lock:
            self._worker(worker_id)
            current = self._bindings.get(request_id)
            if current is not None and current != worker_id:
                raise RuntimeError(
                    f"request {request_id} is already bound to decode worker {current}"
                )
            if current is None:
                self._bindings[request_id] = worker_id
                worker = self._workers[worker_id]
                self._workers[worker_id] = replace(
                    worker, active_requests=worker.active_requests + 1
                )

    def worker_for(self, request_id: int) -> int:
        with self._lock:
            try:
                return self._bindings[request_id]
            except KeyError as exc:
                raise KeyError(f"request {request_id} has no decode worker binding") from exc

    def release(self, request_id: int) -> int | None:
        with self._lock:
            worker_id = self._bindings.pop(request_id, None)
            if worker_id is None:
                return None
            worker = self._workers[worker_id]
            self._workers[worker_id] = replace(
                worker, active_requests=max(0, worker.active_requests - 1)
            )
            return worker_id

    def release_worker(self, worker_id: int) -> tuple[int, ...]:
        """Atomically invalidate every request bound to one failed worker."""
        with self._lock:
            worker = self._worker(worker_id)
            request_ids = tuple(
                request_id
                for request_id, bound_worker in self._bindings.items()
                if bound_worker == worker_id
            )
            for request_id in request_ids:
                self._bindings.pop(request_id)
            self._workers[worker_id] = replace(
                worker,
                active_requests=max(0, worker.active_requests - len(request_ids)),
            )
            return request_ids

    def snapshots(self) -> tuple[DecodeWorkerSnapshot, ...]:
        with self._lock:
            return tuple(self._workers[key] for key in sorted(self._workers))

    def _worker(self, worker_id: int) -> DecodeWorkerSnapshot:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise KeyError(f"unknown decode worker {worker_id}") from exc
