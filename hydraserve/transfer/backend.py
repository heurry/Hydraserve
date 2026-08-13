from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from hashlib import sha256
from multiprocessing import shared_memory
import pickle
import struct
from threading import Condition
from time import monotonic, sleep
from typing import Any

from hydraserve.transfer.descriptor import TransferMode


class TransferBackend(ABC):
    """Transport boundary independent of the tensor runtime."""

    @property
    @abstractmethod
    def transfer_mode(self) -> TransferMode: ...

    @abstractmethod
    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None: ...

    @abstractmethod
    def receive(
        self, key: str, src: int, stream: Any = None, timeout: float | None = None
    ) -> Any: ...

    @abstractmethod
    def get_bandwidth(self) -> float: ...

    @abstractmethod
    def get_latency(self) -> float: ...

    @abstractmethod
    def supports_layer_pipeline(self) -> bool: ...


class InMemoryTransferBackend(TransferBackend):
    """Deterministic backend for protocol tests and single-process simulation."""

    def __init__(
        self,
        mode: TransferMode = TransferMode.PARTIAL_TRANSFER,
        bandwidth_gbps: float = 4.58,
        latency_ms: float = 0.05,
    ) -> None:
        if bandwidth_gbps <= 0 or latency_ms < 0:
            raise ValueError("invalid backend performance values")
        self._mode = mode
        self._bandwidth = bandwidth_gbps
        self._latency = latency_ms
        self._messages: dict[tuple[int, str], Any] = {}
        self._condition = Condition()

    @property
    def transfer_mode(self) -> TransferMode:
        return self._mode

    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None:
        if dst < 0 or not key:
            raise ValueError("invalid transfer destination or key")
        with self._condition:
            message_key = (dst, key)
            if message_key in self._messages:
                raise RuntimeError(f"unconsumed transfer already exists: {key}")
            self._messages[message_key] = deepcopy(payload)
            self._condition.notify_all()

    def receive(
        self, key: str, src: int, stream: Any = None, timeout: float | None = None
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        # The simulation addresses mailboxes by receiving endpoint. ``src`` is
        # that endpoint here; real transports translate this into peer ranks.
        message_key = (src, key)
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while message_key not in self._messages:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out waiting for transfer {key}")
                self._condition.wait(remaining)
            return self._messages.pop(message_key)

    def get_bandwidth(self) -> float:
        return self._bandwidth

    def get_latency(self) -> float:
        return self._latency

    def supports_layer_pipeline(self) -> bool:
        return self._bandwidth >= 10.0


class SharedMemoryTransferBackend(TransferBackend):
    """One-shot POSIX shared-memory mailboxes for the low-bandwidth fallback.

    Payloads are pickled because protocol metadata and NumPy arrays travel as one
    atomic message.  This backend is only for mutually trusted local workers.
    Production transports should serialize typed tensor regions directly.
    """

    _HEADER = struct.Struct("!8sQ")
    _MAGIC = b"HYDRA001"

    def __init__(
        self,
        namespace: str = "hydraserve",
        bandwidth_gbps: float = 4.58,
        latency_ms: float = 0.1,
        poll_interval_s: float = 0.001,
    ) -> None:
        if not namespace or bandwidth_gbps <= 0 or latency_ms < 0 or poll_interval_s <= 0:
            raise ValueError("invalid shared-memory backend configuration")
        self.namespace = namespace
        self._bandwidth = bandwidth_gbps
        self._latency = latency_ms
        self._poll_interval = poll_interval_s
        self._owned_names: set[str] = set()

    @property
    def transfer_mode(self) -> TransferMode:
        return TransferMode.PARTIAL_TRANSFER

    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None:
        if dst < 0 or not key:
            raise ValueError("invalid transfer destination or key")
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        name = self._mailbox_name(key, dst)
        try:
            memory = shared_memory.SharedMemory(
                name=name, create=True, size=self._HEADER.size + len(blob)
            )
        except FileExistsError as exc:
            raise RuntimeError(f"unconsumed shared-memory transfer exists: {key}") from exc
        try:
            memory.buf[: self._HEADER.size] = self._HEADER.pack(self._MAGIC, len(blob))
            memory.buf[self._HEADER.size :] = blob
            self._owned_names.add(name)
        finally:
            memory.close()

    def receive(
        self, key: str, src: int, stream: Any = None, timeout: float | None = None
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        name = self._mailbox_name(key, src)
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            try:
                memory = shared_memory.SharedMemory(name=name, create=False)
                break
            except FileNotFoundError:
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for transfer {key}")
                sleep(self._poll_interval)
        try:
            magic, size = self._HEADER.unpack(bytes(memory.buf[: self._HEADER.size]))
            if magic != self._MAGIC or size > len(memory.buf) - self._HEADER.size:
                raise RuntimeError(f"corrupt shared-memory transfer: {key}")
            payload = pickle.loads(bytes(memory.buf[self._HEADER.size : self._HEADER.size + size]))
        finally:
            memory.close()
            memory.unlink()
            self._owned_names.discard(name)
        return payload

    def get_bandwidth(self) -> float:
        return self._bandwidth

    def get_latency(self) -> float:
        return self._latency

    def supports_layer_pipeline(self) -> bool:
        return False

    def close(self) -> None:
        """Best-effort cleanup for messages not consumed by a receiver."""
        for name in tuple(self._owned_names):
            try:
                memory = shared_memory.SharedMemory(name=name, create=False)
            except FileNotFoundError:
                self._owned_names.discard(name)
                continue
            try:
                memory.unlink()
            finally:
                memory.close()
                self._owned_names.discard(name)

    def _mailbox_name(self, key: str, endpoint: int) -> str:
        digest = sha256(f"{self.namespace}:{endpoint}:{key}".encode()).hexdigest()[:24]
        return f"hydra_{digest}"

    def __enter__(self) -> "SharedMemoryTransferBackend":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
