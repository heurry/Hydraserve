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
