from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from threading import RLock

from hydraserve.router.adaptive_router import AdaptiveRouter, Route


class RequestState(str, Enum):
    WAITING = "waiting"
    PREFILL_RUNNING = "prefill_running"
    TRANSFER_PENDING = "transfer_pending"
    READY = "ready"
    RUNNING = "running"
    PREEMPTED = "preempted"
    RECOVERING = "recovering"
    FINISHED = "finished"
    FAILED = "failed"


_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.WAITING: {RequestState.PREFILL_RUNNING, RequestState.FAILED},
    RequestState.PREFILL_RUNNING: {
        RequestState.TRANSFER_PENDING,
        RequestState.READY,
        RequestState.FAILED,
    },
    RequestState.TRANSFER_PENDING: {RequestState.READY, RequestState.FAILED},
    RequestState.READY: {RequestState.RUNNING, RequestState.FAILED},
    RequestState.RUNNING: {
        RequestState.PREEMPTED,
        RequestState.FINISHED,
        RequestState.FAILED,
    },
    RequestState.PREEMPTED: {RequestState.RECOVERING, RequestState.FAILED},
    RequestState.RECOVERING: {
        RequestState.READY,
        RequestState.PREEMPTED,
        RequestState.FAILED,
    },
    RequestState.FINISHED: set(),
    RequestState.FAILED: set(),
}


@dataclass(slots=True)
class Request:
    request_id: int
    token_ids: tuple[int, ...]
    max_new_tokens: int
    route: Route
    state: RequestState = RequestState.WAITING
    generated_token_ids: list[int] = field(default_factory=list)

    def transition(self, target: RequestState) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid request transition {self.state.value} -> {target.value}")
        self.state = target


class CentralScheduler:
    def __init__(self, router: AdaptiveRouter | None = None) -> None:
        self.router = router or AdaptiveRouter()
        self._ids = count()
        self._requests: dict[int, Request] = {}
        self._lock = RLock()

    def submit(
        self,
        token_ids: list[int] | tuple[int, ...],
        max_new_tokens: int,
        decode_load: float = 0.0,
        decode_has_slot: bool = True,
    ) -> Request:
        if not token_ids or max_new_tokens <= 0:
            raise ValueError("request needs a prompt and positive max_new_tokens")
        with self._lock:
            request_id = next(self._ids)
            request = Request(
                request_id,
                tuple(token_ids),
                max_new_tokens,
                self.router.route(len(token_ids), decode_load, decode_has_slot),
            )
            self._requests[request_id] = request
            return request

    def get(self, request_id: int) -> Request:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request {request_id}") from exc
