from __future__ import annotations

from typing import Any, Callable

from hydraserve.engine.continuous_batching import ContinuousBatchScheduler, DecodeBatch


Sampler = Callable[[Any], tuple[int, ...]]


class ContinuousBatchExecutor:
    """Connect iteration scheduling to one batched model decode invocation."""

    def __init__(
        self,
        scheduler: ContinuousBatchScheduler,
        runtime,
        paged_cache,
        *,
        sampler: Sampler | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.runtime = runtime
        self.paged_cache = paged_cache
        self.sampler = sampler or self._greedy
        self.states: dict[int, Any] = {}

    def register_state(self, request_id: int, state: Any) -> None:
        if request_id in self.states:
            raise ValueError(f"duplicate runtime state for request {request_id}")
        self.states[request_id] = state

    def step(self) -> tuple[DecodeBatch, tuple[int, ...]]:
        import torch

        batch = self.scheduler.next_decode_batch()
        if not batch.request_ids:
            return batch, ()
        try:
            states = [self.states[request_id] for request_id in batch.request_ids]
            input_ids = torch.tensor(
                batch.token_ids,
                device=self.runtime.device,
                dtype=torch.long,
            ).unsqueeze(1)
            logits, _ = self.runtime.decode_batch(
                input_ids,
                states,
                self.paged_cache,
                batch.request_ids,
            )
            sampled = self.sampler(logits[:, -1])
            self.scheduler.commit_decode_tokens(batch, sampled)
            return batch, sampled
        except Exception:
            self.scheduler.fail_decode_batch(batch)
            raise

    @staticmethod
    def _greedy(logits) -> tuple[int, ...]:
        return tuple(int(token) for token in logits.argmax(dim=-1).tolist())
