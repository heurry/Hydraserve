from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    COLLOCATED = "collocated"
    PD_DISAGGREGATED = "pd_disaggregated"


class RouteReason(str, Enum):
    SHORT_PROMPT = "short_prompt"
    MEDIUM_PROMPT = "medium_prompt"
    LONG_PROMPT_PD = "long_prompt_pd"
    FORCED_LONG_PROMPT = "forced_long_prompt"
    DECODE_SATURATED = "decode_saturated"
    NO_DECODE_SLOT = "no_decode_slot"
    PREFILL_UNAVAILABLE = "prefill_unavailable"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    reason: RouteReason
    prompt_tokens: int
    decode_load: float
    decode_has_slot: bool


@dataclass(frozen=True, slots=True)
class RouterConfig:
    short_prompt_tokens: int = 2_048
    long_prompt_tokens: int = 8_192
    force_pd_tokens: int = 32_768
    decode_load_limit: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.short_prompt_tokens < self.long_prompt_tokens <= self.force_pd_tokens:
            raise ValueError("prompt thresholds must be strictly ordered")
        if not 0 < self.decode_load_limit <= 1:
            raise ValueError("decode_load_limit must be in (0, 1]")


class AdaptiveRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, prompt_tokens: int, decode_load: float, decode_has_slot: bool = True) -> Route:
        return self.decide(prompt_tokens, decode_load, decode_has_slot).route

    def decide(
        self,
        prompt_tokens: int,
        decode_load: float,
        decode_has_slot: bool = True,
    ) -> RouteDecision:
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        if prompt_tokens >= self.config.force_pd_tokens:
            route, reason = Route.PD_DISAGGREGATED, RouteReason.FORCED_LONG_PROMPT
        elif prompt_tokens < self.config.short_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.SHORT_PROMPT
        elif not decode_has_slot:
            route, reason = Route.COLLOCATED, RouteReason.NO_DECODE_SLOT
        elif (
            prompt_tokens >= self.config.long_prompt_tokens
            and decode_load < self.config.decode_load_limit
        ):
            route, reason = Route.PD_DISAGGREGATED, RouteReason.LONG_PROMPT_PD
        elif prompt_tokens >= self.config.long_prompt_tokens:
            route, reason = Route.COLLOCATED, RouteReason.DECODE_SATURATED
        else:
            route, reason = Route.COLLOCATED, RouteReason.MEDIUM_PROMPT
        return RouteDecision(
            route=route,
            reason=reason,
            prompt_tokens=prompt_tokens,
            decode_load=decode_load,
            decode_has_slot=decode_has_slot,
        )
