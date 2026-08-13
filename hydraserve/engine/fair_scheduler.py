from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FairSchedulingConfig:
    max_priority: int = 7
    priority_bias: float = 1.0
    aging_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.max_priority < 0:
            raise ValueError("max priority cannot be negative")
        if self.priority_bias < 0 or self.aging_weight <= 0:
            raise ValueError("invalid fairness weights")


@dataclass(slots=True)
class _Account:
    service_tokens: int = 0
    last_served_round: int = 0


class FairDecodeScheduler:
    """Weighted fair decode selection with aging-based starvation prevention."""

    def __init__(self, config: FairSchedulingConfig | None = None) -> None:
        self.config = config or FairSchedulingConfig()
        self._round = 0
        self._accounts: dict[int, _Account] = {}

    def select(self, requests, limit: int):
        candidates = tuple(requests)
        if limit <= 0:
            raise ValueError("decode selection limit must be positive")
        if not candidates:
            return ()
        if len({request.request_id for request in candidates}) != len(candidates):
            raise ValueError("decode candidates must have unique request ids")
        self._round += 1
        live = {request.request_id for request in candidates}
        for request_id in tuple(self._accounts):
            if request_id not in live:
                self._accounts.pop(request_id)

        def score(request):
            if not 0 <= request.priority <= self.config.max_priority:
                raise ValueError(
                    f"request priority must be in [0, {self.config.max_priority}]"
                )
            account = self._accounts.setdefault(
                request.request_id, _Account(last_served_round=self._round - 1)
            )
            weight = request.priority + 1
            waiting_rounds = self._round - account.last_served_round
            return (
                account.service_tokens / weight
                - self.config.priority_bias * request.priority
                - self.config.aging_weight * waiting_rounds,
                request.request_id,
            )

        selected = tuple(sorted(candidates, key=score)[:limit])
        for request in selected:
            account = self._accounts[request.request_id]
            account.service_tokens += 1
            account.last_served_round = self._round
        return selected

    def forget(self, request_id: int) -> None:
        self._accounts.pop(request_id, None)

