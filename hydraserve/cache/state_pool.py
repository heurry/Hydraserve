from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True, slots=True)
class StateSlotCapacity:
    total_slots: int
    free_slots: int

    @property
    def allocated_slots(self) -> int:
        return self.total_slots - self.free_slots


class RequestStateSlotManager:
    """Atomic ownership for fixed-size per-request GPU recurrent states."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._free = list(range(capacity))
        self._request_to_slot: dict[int, int] = {}
        self._lock = RLock()

    def allocate(self, request_id: int) -> int:
        with self._lock:
            if request_id in self._request_to_slot:
                return self._request_to_slot[request_id]
            if not self._free:
                raise MemoryError("recurrent-state slots are exhausted")
            slot = self._free.pop(0)
            self._request_to_slot[request_id] = slot
            return slot

    def get(self, request_id: int) -> int:
        with self._lock:
            try:
                return self._request_to_slot[request_id]
            except KeyError as exc:
                raise KeyError(f"request {request_id} has no recurrent-state slot") from exc

    def free(self, request_id: int) -> None:
        with self._lock:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is None:
                return
            self._free.append(slot)
            self._free.sort()

    def capacity(self) -> StateSlotCapacity:
        with self._lock:
            return StateSlotCapacity(self._capacity, len(self._free))


@dataclass(slots=True)
class LinearState:
    ssm_state: np.ndarray
    conv_state: np.ndarray

    def __post_init__(self) -> None:
        if self.ssm_state.dtype != np.float32 or self.conv_state.dtype != np.float32:
            raise TypeError("linear recurrent state must remain float32")

    def copy(self) -> "LinearState":
        return LinearState(self.ssm_state.copy(), self.conv_state.copy())


class LinearStatePool:
    """Fixed-slot pool for the non-quantizable recurrent state."""

    def __init__(
        self,
        capacity: int,
        ssm_shape: tuple[int, ...],
        conv_shape: tuple[int, ...],
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.ssm_shape = ssm_shape
        self.conv_shape = conv_shape
        self._free = list(range(capacity))
        self._request_to_slot: dict[int, int] = {}
        self._slots: dict[int, LinearState] = {}
        self._lock = RLock()

    @property
    def num_free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    def allocate(self, request_id: int) -> int:
        with self._lock:
            if request_id in self._request_to_slot:
                raise ValueError(f"request {request_id} already owns a state slot")
            if not self._free:
                raise MemoryError("linear state pool is exhausted")
            slot = self._free.pop(0)
            self._request_to_slot[request_id] = slot
            self._slots[slot] = LinearState(
                np.zeros(self.ssm_shape, dtype=np.float32),
                np.zeros(self.conv_shape, dtype=np.float32),
            )
            return slot

    def set(self, request_id: int, state: LinearState, *, copy: bool = True) -> None:
        with self._lock:
            slot = self._slot_for(request_id)
            if state.ssm_state.shape != self.ssm_shape:
                raise ValueError(f"unexpected SSM shape {state.ssm_state.shape}")
            if state.conv_state.shape != self.conv_shape:
                raise ValueError(f"unexpected conv shape {state.conv_state.shape}")
            self._slots[slot] = state.copy() if copy else state

    def get(self, request_id: int) -> LinearState:
        with self._lock:
            return self._slots[self._slot_for(request_id)]

    def free(self, request_id: int) -> None:
        with self._lock:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is None:
                return
            self._slots.pop(slot, None)
            self._free.append(slot)
            self._free.sort()

    def _slot_for(self, request_id: int) -> int:
        try:
            return self._request_to_slot[request_id]
        except KeyError as exc:
            raise KeyError(f"request {request_id} has no recurrent-state slot") from exc


class GpuLinearStatePool:
    """Contiguous layer-major FP32 storage for runtime GDN states."""

    def __init__(
        self,
        capacity: int,
        model,
        *,
        device,
        cuda_memory_fraction: float = 0.5,
        cuda_reserve_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        import torch

        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 < cuda_memory_fraction <= 1 or cuda_reserve_bytes < 0:
            raise ValueError("invalid CUDA state-pool memory policy")
        self.layer_indices = tuple(model.linear_layer_indices)
        if model.ssm_state_shape[0] != len(self.layer_indices):
            raise ValueError("SSM state shape does not match linear layers")
        if model.conv_state_shape[0] != len(self.layer_indices):
            raise ValueError("conv state shape does not match linear layers")
        self.layer_to_index = {
            layer_index: index for index, layer_index in enumerate(self.layer_indices)
        }
        requested_capacity = capacity
        target = torch.device(device)
        slot_elements = int(np.prod(model.ssm_state_shape, dtype=np.int64)) + int(
            np.prod(model.conv_state_shape, dtype=np.int64)
        )
        self.bytes_per_slot = slot_elements * 4
        if target.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(target)
            budget = max(
                0,
                int(free_bytes * cuda_memory_fraction) - cuda_reserve_bytes,
            )
            capacity = min(capacity, budget // self.bytes_per_slot)
            if capacity <= 0:
                raise MemoryError(
                    "insufficient CUDA memory for one guaranteed recurrent-state slot"
                )
        self.requested_capacity = requested_capacity
        self.capacity = capacity
        self.slots = RequestStateSlotManager(capacity)
        self.ssm_storage = torch.empty(
            (len(self.layer_indices), capacity, *model.ssm_state_shape[1:]),
            device=target,
            dtype=torch.float32,
        )
        self.conv_storage = torch.empty(
            (len(self.layer_indices), capacity, *model.conv_state_shape[1:]),
            device=target,
            dtype=torch.float32,
        )
        self._states: dict[int, object] = {}
        self._lock = RLock()

    def allocate(self, request_id: int, *, sequence_length: int = 0):
        from hydraserve.model.runtime import RuntimeState

        with self._lock:
            existing = self._states.get(request_id)
            if existing is not None:
                return existing
            slot = self.slots.allocate(request_id)
            self.ssm_storage[:, slot].zero_()
            self.conv_storage[:, slot].zero_()
            state = RuntimeState(sequence_length=sequence_length)
            for layer_index, layer_slot in self.layer_to_index.items():
                state.recurrent[layer_index] = self.ssm_storage[
                    layer_slot, slot : slot + 1
                ]
                state.convolution[layer_index] = self.conv_storage[
                    layer_slot, slot : slot + 1
                ]
            self._states[request_id] = state
            return state

    def install(self, request_id: int, state):
        with self._lock:
            pooled = self.allocate(
                request_id, sequence_length=int(state.sequence_length)
            )
            for layer_index in self.layer_indices:
                try:
                    recurrent = state.recurrent[layer_index]
                    convolution = state.convolution[layer_index]
                except KeyError as exc:
                    raise ValueError(
                        f"runtime state is missing linear layer {layer_index}"
                    ) from exc
                target_recurrent = pooled.recurrent[layer_index]
                target_convolution = pooled.convolution[layer_index]
                if recurrent.shape != target_recurrent.shape:
                    raise ValueError(
                        f"unexpected recurrent shape at layer {layer_index}"
                    )
                if convolution.shape != target_convolution.shape:
                    raise ValueError(
                        f"unexpected convolution shape at layer {layer_index}"
                    )
                target_recurrent.copy_(recurrent.to(target_recurrent))
                target_convolution.copy_(convolution.to(target_convolution))
            pooled.sequence_length = int(state.sequence_length)
            return pooled

    def get(self, request_id: int):
        with self._lock:
            try:
                return self._states[request_id]
            except KeyError as exc:
                raise KeyError(
                    f"request {request_id} has no pooled recurrent state"
                ) from exc

    def free(self, request_id: int) -> None:
        with self._lock:
            self._states.pop(request_id, None)
            self.slots.free(request_id)

    def capacity_snapshot(self) -> StateSlotCapacity:
        return self.slots.capacity()
