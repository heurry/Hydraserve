from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    logprobs: int | None = None
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be in (0, 2]")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if self.logprobs is not None and not 0 <= self.logprobs <= 20:
            raise ValueError("logprobs must be in [0, 20]")
        if any(not sequence for sequence in self.stop_token_sequences):
            raise ValueError("stop token sequences cannot be empty")

    def with_seed(self, seed: int) -> "SamplingParams":
        return replace(self, seed=seed)


@dataclass(frozen=True, slots=True)
class TokenSample:
    token_id: int
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, float], ...] = ()


def sample_logits(
    logits,
    histories: Iterable[Iterable[int]],
    params: Iterable[SamplingParams],
    *,
    steps: Iterable[int],
) -> tuple[TokenSample, ...]:
    """Apply per-request penalties/filters and sample independently of batching."""
    if logits.ndim != 2:
        raise ValueError("sampling logits must have shape [batch, vocabulary]")
    histories = tuple(histories)
    params = tuple(params)
    steps = tuple(int(step) for step in steps)
    if not (len(histories) == len(params) == len(steps) == logits.shape[0]):
        raise ValueError("sampling metadata must match the logits batch")
    if os.environ.get("HYDRASERVE_BATCHED_GREEDY", "1") != "0" and all(
        _is_plain_greedy(config) for config in params
    ):
        token_ids = logits.argmax(dim=-1).tolist()
        return tuple(TokenSample(int(token_id)) for token_id in token_ids)

    histories = tuple(
        tuple(int(token) for token in history) for history in histories
    )
    return tuple(
        _sample_row(row.float(), history, config, step)
        for row, history, config, step in zip(
            logits, histories, params, steps, strict=True
        )
    )


def _is_plain_greedy(params: SamplingParams) -> bool:
    return (
        params.temperature == 0.0
        and not _has_penalties(params)
        and params.logprobs is None
    )


def _has_penalties(params: SamplingParams) -> bool:
    return (
        params.repetition_penalty != 1.0
        or params.presence_penalty != 0.0
        or params.frequency_penalty != 0.0
    )


def _sample_row(logits, history, params: SamplingParams, step: int) -> TokenSample:
    import torch

    scores = logits.clone()
    if history and _has_penalties(params):
        counts: dict[int, int] = {}
        for token in history:
            if 0 <= token < scores.numel():
                counts[token] = counts.get(token, 0) + 1
        if counts:
            token_ids = torch.tensor(tuple(counts), device=scores.device, dtype=torch.long)
            values = scores[token_ids]
            penalty = params.repetition_penalty
            scores[token_ids] = torch.where(values < 0, values * penalty, values / penalty)
            frequencies = torch.tensor(
                tuple(counts.values()), device=scores.device, dtype=scores.dtype
            )
            scores[token_ids] -= params.presence_penalty
            scores[token_ids] -= params.frequency_penalty * frequencies

    greedy = params.temperature == 0.0
    if not greedy:
        scores /= params.temperature
        if params.top_k:
            keep = min(params.top_k, scores.numel())
            threshold = torch.topk(scores, keep).values[-1]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        if params.min_p:
            threshold = probabilities.max() * params.min_p
            scores = scores.masked_fill(probabilities < threshold, float("-inf"))
        if params.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            remove = cumulative > params.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores[sorted_indices[remove]] = float("-inf")

    if greedy:
        token_id = int(scores.argmax())
    else:
        probabilities = torch.softmax(scores, dim=-1)
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=scores.device)
            mixed_seed = (params.seed + (step + 1) * 0x1E3779B97F4A7C15) % (2**63)
            generator.manual_seed(mixed_seed)
        token_id = int(torch.multinomial(probabilities, 1, generator=generator))

    if params.logprobs is None:
        return TokenSample(token_id)
    log_probabilities = torch.log_softmax(scores, dim=-1)
    top_count = min(params.logprobs, scores.numel())
    top = ()
    if top_count:
        values, indices = torch.topk(log_probabilities, top_count)
        top = tuple(
            (int(index), float(value))
            for index, value in zip(indices.tolist(), values.tolist(), strict=True)
        )
    return TokenSample(token_id, float(log_probabilities[token_id]), top)
