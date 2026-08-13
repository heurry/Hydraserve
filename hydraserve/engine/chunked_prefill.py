from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PrefillChunk:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class ChunkedPrefillScheduler:
    def __init__(self, chunk_size: int = 4096) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def split(self, num_tokens: int, *, n_minus_one: bool = False) -> tuple[PrefillChunk, ...]:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        effective_tokens = num_tokens - 1 if n_minus_one else num_tokens
        if effective_tokens <= 0:
            return ()
        return tuple(
            PrefillChunk(start, min(start + self.chunk_size, effective_tokens))
            for start in range(0, effective_tokens, self.chunk_size)
        )
