from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from weakref import ref

import numpy as np


def cuda_state_memory_budget(
    free_bytes: int, memory_fraction: float, reserve_bytes: int
) -> int:
    """Bytes allowed while satisfying both fraction and hard-reserve limits."""
    if free_bytes < 0 or not 0 < memory_fraction <= 1 or reserve_bytes < 0:
        raise ValueError("invalid CUDA state-pool memory policy")
    return min(
        int(free_bytes * memory_fraction),
        max(0, free_bytes - reserve_bytes),
    )


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
        workspace_capacity: int | None = None,
        cuda_memory_fraction: float = 0.5,
        cuda_reserve_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        import torch

        if capacity <= 0 or (workspace_capacity is not None and workspace_capacity <= 0):
            raise ValueError("capacity and workspace capacity must be positive")
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
        requested_workspace_capacity = min(
            capacity, capacity if workspace_capacity is None else workspace_capacity
        )
        target = torch.device(device)
        conv_slot_elements = int(np.prod(model.conv_state_shape, dtype=np.int64))
        slot_elements = int(np.prod(model.ssm_state_shape, dtype=np.int64)) + conv_slot_elements
        self.bytes_per_slot = slot_elements * 4
        # A batch workspace holds a transactional copy of all recurrent states
        # plus a second convolution buffer for the next state.
        self.bytes_per_workspace_slot = (slot_elements + conv_slot_elements) * 4
        effective_workspace_capacity = requested_workspace_capacity
        if target.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(target)
            budget = cuda_state_memory_budget(
                free_bytes, cuda_memory_fraction, cuda_reserve_bytes
            )
            capacity = min(capacity, budget // self.bytes_per_slot)
            effective_workspace_capacity = min(
                requested_workspace_capacity, capacity
            )
            required = (
                capacity * self.bytes_per_slot
                + effective_workspace_capacity * self.bytes_per_workspace_slot
            )
            if required > budget:
                capacity_with_full_workspace = (
                    budget
                    - effective_workspace_capacity * self.bytes_per_workspace_slot
                ) // self.bytes_per_slot
                if capacity_with_full_workspace >= effective_workspace_capacity:
                    capacity = min(capacity, capacity_with_full_workspace)
                else:
                    capacity = min(
                        capacity,
                        budget
                        // (self.bytes_per_slot + self.bytes_per_workspace_slot),
                    )
                    effective_workspace_capacity = min(
                        effective_workspace_capacity, capacity
                    )
            if capacity <= 0:
                raise MemoryError(
                    "insufficient CUDA memory for one guaranteed recurrent-state slot"
                )
        self.requested_capacity = requested_capacity
        self.requested_workspace_capacity = requested_workspace_capacity
        self.capacity = capacity
        self.workspace_capacity = min(effective_workspace_capacity, capacity)
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
        self.ssm_workspace = torch.empty(
            (len(self.layer_indices), self.workspace_capacity, *model.ssm_state_shape[1:]),
            device=target,
            dtype=torch.float32,
        )
        self.conv_workspace = torch.empty(
            (len(self.layer_indices), self.workspace_capacity, *model.conv_state_shape[1:]),
            device=target,
            dtype=torch.float32,
        )
        self.next_conv_workspace = torch.empty_like(self.conv_workspace)
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
            state._state_pool_ref = ref(self)
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

    @contextmanager
    def batch(self, request_ids, states):
        """Gather and transactionally commit one heterogeneous decode batch."""
        import torch

        request_ids = tuple(int(request_id) for request_id in request_ids)
        states = tuple(states)
        if not request_ids or len(request_ids) != len(states):
            raise ValueError("state batch must be non-empty and aligned")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("state batch request ids must be unique")
        if len(request_ids) > self.workspace_capacity:
            raise MemoryError("decode batch exceeds recurrent-state workspace capacity")
        with self._lock:
            slots = []
            for request_id, state in zip(request_ids, states, strict=True):
                if self._states.get(request_id) is not state:
                    raise ValueError("decode state does not match its pool owner")
                slots.append(self.slots.get(request_id))
            slot_ids = torch.tensor(
                slots, device=self.ssm_storage.device, dtype=torch.long
            )
            batch_size = len(request_ids)
            ssm = self.ssm_workspace[:, :batch_size]
            convolution = self.conv_workspace[:, :batch_size]
            next_convolution = self.next_conv_workspace[:, :batch_size]
            torch.index_select(self.ssm_storage, 1, slot_ids, out=ssm)
            torch.index_select(self.conv_storage, 1, slot_ids, out=convolution)
            state_batch = GpuLinearStateBatch(
                self,
                slot_ids,
                ssm,
                convolution,
                next_convolution,
            )
            yield state_batch

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

    def stats(self) -> dict[str, int]:
        return {
            "state_requested_slots": self.requested_capacity,
            "state_total_slots": self.capacity,
            "state_workspace_slots": self.workspace_capacity,
            "state_bytes_per_slot": self.bytes_per_slot,
            "state_storage_bytes": self.capacity * self.bytes_per_slot,
            "state_workspace_bytes": (
                self.workspace_capacity * self.bytes_per_workspace_slot
            ),
        }


class GpuLinearStateBatch:
    """Exclusive reusable state workspace held for one decode transaction."""

    def __init__(
        self,
        pool: GpuLinearStatePool,
        slot_ids,
        recurrent,
        convolution,
        next_convolution,
    ) -> None:
        self.pool = pool
        self.slot_ids = slot_ids
        self.recurrent = recurrent
        self.convolution = convolution
        self.next_convolution = next_convolution

    def layer(self, layer_index: int):
        try:
            layer_slot = self.pool.layer_to_index[layer_index]
        except KeyError as exc:
            raise ValueError(f"layer {layer_index} is not a pooled linear layer") from exc
        return (
            self.recurrent[layer_slot],
            self.convolution[layer_slot],
            self.next_convolution[layer_slot],
        )

    def set_layer_result(self, layer_index: int, recurrent, convolution) -> None:
        target_recurrent, _, target_convolution = self.layer(layer_index)
        if recurrent.shape != target_recurrent.shape:
            raise ValueError("unexpected batched recurrent result shape")
        if convolution.shape != target_convolution.shape:
            raise ValueError("unexpected batched convolution result shape")
        if recurrent.data_ptr() != target_recurrent.data_ptr():
            target_recurrent.copy_(recurrent)
        if convolution.data_ptr() != target_convolution.data_ptr():
            target_convolution.copy_(convolution)

    def commit(self) -> None:
        """Publish every layer with two batched device scatters."""
        self.pool.ssm_storage.index_copy_(1, self.slot_ids, self.recurrent)
        self.pool.conv_storage.index_copy_(1, self.slot_ids, self.next_convolution)
