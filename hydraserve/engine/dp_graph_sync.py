"""Cross-data-parallel shape synchronization for CUDA Graph decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DPPaddingPlan:
    local_tokens: int
    synchronized_tokens: int

    @property
    def padding_tokens(self) -> int:
        return self.synchronized_tokens - self.local_tokens

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return (True,) * self.local_tokens + (False,) * self.padding_tokens


def synchronize_dp_token_count(
    local_tokens: int, *, process_group=None, device=None
) -> DPPaddingPlan:
    """All-reduce the maximum token count so every DP rank captures one shape."""
    if local_tokens < 0:
        raise ValueError("local token count cannot be negative")
    import torch

    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return DPPaddingPlan(local_tokens, local_tokens)
    target = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    count = torch.tensor(local_tokens, device=target, dtype=torch.int64)
    distributed.all_reduce(count, op=distributed.ReduceOp.MAX, group=process_group)
    return DPPaddingPlan(local_tokens, int(count.cpu()))


def pad_dp_batch(input_ids, target_tokens: int, *, pad_token_id: int = 0):
    """Pad only the batch dimension and return a mask for discarding dummies."""
    import torch

    if input_ids.ndim != 2 or target_tokens < input_ids.shape[0]:
        raise ValueError("invalid DP batch padding target")
    padding = target_tokens - input_ids.shape[0]
    if not padding:
        return input_ids, torch.ones(
            input_ids.shape[0], device=input_ids.device, dtype=torch.bool
        )
    padded = torch.full(
        (padding, input_ids.shape[1]),
        pad_token_id,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    mask = torch.arange(target_tokens, device=input_ids.device) < input_ids.shape[0]
    return torch.cat((input_ids, padded), dim=0), mask
