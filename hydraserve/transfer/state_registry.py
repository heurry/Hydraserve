"""Extensible dispatch table for hybrid-model state families."""

from __future__ import annotations

from hydraserve.transfer.descriptor import StateType


class StateHandlerRegistry:
    def __init__(self) -> None:
        self._handlers = {}

    def register(self, state_type: StateType, handler) -> None:
        state_type = StateType(state_type)
        if not callable(handler):
            raise TypeError("state handler must be callable")
        if state_type in self._handlers:
            raise ValueError(f"state handler already registered: {state_type.value}")
        self._handlers[state_type] = handler

    def dispatch(self, state_type: StateType, *args, **kwargs):
        state_type = StateType(state_type)
        try:
            handler = self._handlers[state_type]
        except KeyError as exc:
            raise NotImplementedError(
                f"no transfer handler for state type {state_type.value}"
            ) from exc
        return handler(*args, **kwargs)

    @property
    def supported_types(self) -> tuple[StateType, ...]:
        return tuple(self._handlers)
