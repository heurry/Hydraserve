from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    COLLOCATED = "collocated"
    PD_DISAGGREGATED = "pd_disaggregated"


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
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if not 0 <= decode_load <= 1:
            raise ValueError("decode_load must be in [0, 1]")
        if prompt_tokens >= self.config.force_pd_tokens:
            return Route.PD_DISAGGREGATED
        if prompt_tokens < self.config.short_prompt_tokens:
            return Route.COLLOCATED
        if (
            prompt_tokens >= self.config.long_prompt_tokens
            and decode_has_slot
            and decode_load < self.config.decode_load_limit
        ):
            return Route.PD_DISAGGREGATED
        return Route.COLLOCATED
