from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from hashlib import sha256
import json
from multiprocessing import shared_memory
import numpy as np
import struct
from threading import Condition
from time import monotonic, sleep
from typing import Any

from hydraserve.transfer.descriptor import TransferMode
from hydraserve.cache.kv_quantizer import Int4Tensor, Int8Tensor, PagedInt8KVTensor


class TransferCancelledError(RuntimeError):
    """Raised when a pending receive is explicitly cancelled."""


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise TransferCancelledError("transfer receive was cancelled")


class TransferBackend(ABC):
    """Transport boundary independent of the tensor runtime."""

    @property
    @abstractmethod
    def transfer_mode(self) -> TransferMode: ...

    @abstractmethod
    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None: ...

    @abstractmethod
    def receive(
        self,
        key: str,
        src: int,
        stream: Any = None,
        timeout: float | None = None,
        cancel_event=None,
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
        self,
        key: str,
        src: int,
        stream: Any = None,
        timeout: float | None = None,
        cancel_event=None,
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        # The simulation addresses mailboxes by receiving endpoint. ``src`` is
        # that endpoint here; real transports translate this into peer ranks.
        message_key = (src, key)
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while message_key not in self._messages:
                _raise_if_cancelled(cancel_event)
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"timed out waiting for transfer {key}")
                self._condition.wait(
                    0.05 if remaining is None else min(0.05, remaining)
                )
            _raise_if_cancelled(cancel_event)
            return self._messages.pop(message_key)

    def get_bandwidth(self) -> float:
        return self._bandwidth

    def get_latency(self) -> float:
        return self._latency

    def supports_layer_pipeline(self) -> bool:
        return self._bandwidth >= 10.0

    def discard(self, key: str, endpoint: int) -> None:
        with self._condition:
            self._messages.pop((endpoint, key), None)
            self._condition.notify_all()


class SharedMemoryTransferBackend(TransferBackend):
    """One-shot POSIX shared-memory mailboxes for the low-bandwidth fallback.

    NumPy regions are copied directly into a typed wire layout. The header is
    published last, so a receiver can never deserialize a partially written
    message merely because the POSIX object already exists.
    """

    _HEADER = struct.Struct("!8sQQ")
    _MAGIC = b"HYDRA002"
    _PENDING = b"\0" * 8

    def __init__(
        self,
        namespace: str = "hydraserve",
        bandwidth_gbps: float = 4.58,
        latency_ms: float = 0.1,
        poll_interval_s: float = 0.001,
        mode: TransferMode = TransferMode.FULL_TRANSFER,
    ) -> None:
        if not namespace or bandwidth_gbps <= 0 or latency_ms < 0 or poll_interval_s <= 0:
            raise ValueError("invalid shared-memory backend configuration")
        if mode is TransferMode.INTRA_GPU:
            raise ValueError("shared-memory backend cannot serve intra-GPU transfers")
        self.namespace = namespace
        self._bandwidth = bandwidth_gbps
        self._latency = latency_ms
        self._poll_interval = poll_interval_s
        self._mode = mode
        self._owned_names: set[str] = set()

    @property
    def transfer_mode(self) -> TransferMode:
        # Recomputing full-attention KV on the decode side (PARTIAL) is never
        # cheaper than shipping it, since the recompute is a full O(n^2) prefill
        # while the transfer is O(n). Default to FULL; QUANTIZED also available.
        return self._mode

    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None:
        if dst < 0 or not key:
            raise ValueError("invalid transfer destination or key")
        metadata, arrays = self._encode(payload)
        metadata_blob = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        payload_size = sum(array.nbytes for array in arrays)
        name = self._mailbox_name(key, dst)
        try:
            memory = shared_memory.SharedMemory(
                name=name,
                create=True,
                size=self._HEADER.size + len(metadata_blob) + payload_size,
            )
        except FileExistsError as exc:
            raise RuntimeError(f"unconsumed shared-memory transfer exists: {key}") from exc
        try:
            memory.buf[: self._HEADER.size] = self._HEADER.pack(
                self._PENDING, 0, 0
            )
            cursor = self._HEADER.size
            memory.buf[cursor : cursor + len(metadata_blob)] = metadata_blob
            cursor += len(metadata_blob)
            for array in arrays:
                data = memoryview(array).cast("B")
                try:
                    memory.buf[cursor : cursor + array.nbytes] = data
                finally:
                    data.release()
                cursor += array.nbytes
            memory.buf[: self._HEADER.size] = self._HEADER.pack(
                self._MAGIC, len(metadata_blob), payload_size
            )
            self._owned_names.add(name)
        finally:
            memory.close()

    def receive(
        self,
        key: str,
        src: int,
        stream: Any = None,
        timeout: float | None = None,
        cancel_event=None,
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        name = self._mailbox_name(key, src)
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            _raise_if_cancelled(cancel_event)
            try:
                memory = shared_memory.SharedMemory(name=name, create=False)
                break
            except FileNotFoundError:
                pass
            except ValueError as exc:
                # POSIX shm_open publishes the name before CPython's creator
                # has completed ftruncate().  A racing receiver can therefore
                # observe a real object whose size is temporarily zero and
                # SharedMemory reports "cannot mmap an empty file".  Treat only
                # that transient creation state like a missing mailbox.
                if "cannot mmap an empty file" not in str(exc).lower():
                    raise
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for transfer {key}")
            sleep(self._poll_interval)
        try:
            while True:
                _raise_if_cancelled(cancel_event)
                magic, metadata_size, payload_size = self._HEADER.unpack(
                    bytes(memory.buf[: self._HEADER.size])
                )
                if magic == self._MAGIC:
                    break
                if magic != self._PENDING:
                    raise RuntimeError(f"corrupt shared-memory transfer: {key}")
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for transfer {key} publication")
                sleep(self._poll_interval)
            available = len(memory.buf) - self._HEADER.size
            if metadata_size + payload_size != available:
                raise RuntimeError(f"corrupt shared-memory transfer: {key}")
            metadata_start = self._HEADER.size
            payload_start = metadata_start + metadata_size
            metadata = json.loads(
                bytes(memory.buf[metadata_start:payload_start]).decode("utf-8")
            )
            payload_view = memory.buf[payload_start:]
            try:
                payload = self._decode(metadata, payload_view)
            finally:
                payload_view.release()
        finally:
            memory.close()
            memory.unlink()
            self._owned_names.discard(name)
        return payload

    @classmethod
    def _encode(cls, payload: Any) -> tuple[Any, tuple[np.ndarray, ...]]:
        arrays: list[np.ndarray] = []

        def visit(value):
            if isinstance(value, Int4Tensor):
                return {
                    "__hydra_int4__": True,
                    "packed": visit(value.packed),
                    "scales": visit(value.scales),
                    "shape": list(value.shape),
                    "group_size": value.group_size,
                    "original_dtype": value.original_dtype,
                }
            if isinstance(value, Int8Tensor):
                return {
                    "__hydra_int8__": True,
                    "quantized": visit(value.quantized),
                    "scales": visit(value.scales),
                    "shape": list(value.shape),
                    "group_size": value.group_size,
                    "original_dtype": value.original_dtype,
                }
            if isinstance(value, PagedInt8KVTensor):
                return {
                    "__hydra_paged_int8_kv__": True,
                    "key": visit(value.key),
                    "value": visit(value.value),
                    "key_scales": visit(value.key_scales),
                    "value_scales": visit(value.value_scales),
                    "shape": list(value.shape),
                    "original_dtype": value.original_dtype,
                }
            if isinstance(value, np.ndarray):
                contiguous = np.ascontiguousarray(value)
                index = len(arrays)
                arrays.append(contiguous)
                return {
                    "__hydra_ndarray__": index,
                    "shape": list(contiguous.shape),
                    "dtype": contiguous.dtype.str,
                    "nbytes": contiguous.nbytes,
                }
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise TypeError("shared-memory dictionaries require string keys")
                return {key: visit(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return {"__hydra_tuple__": [visit(item) for item in value]}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            raise TypeError(
                f"unsupported shared-memory payload type: {type(value).__name__}"
            )

        return visit(payload), tuple(arrays)

    @classmethod
    def _decode(cls, metadata: Any, payload) -> Any:
        arrays: dict[int, np.ndarray] = {}
        cursor = 0

        def visit(value):
            nonlocal cursor
            if isinstance(value, dict) and value.get("__hydra_int4__") is True:
                return Int4Tensor(
                    visit(value["packed"]),
                    visit(value["scales"]),
                    tuple(int(item) for item in value["shape"]),
                    int(value["group_size"]),
                    str(value["original_dtype"]),
                )
            if isinstance(value, dict) and value.get("__hydra_int8__") is True:
                return Int8Tensor(
                    visit(value["quantized"]),
                    visit(value["scales"]),
                    tuple(int(item) for item in value["shape"]),
                    int(value["group_size"]),
                    str(value["original_dtype"]),
                )
            if (
                isinstance(value, dict)
                and value.get("__hydra_paged_int8_kv__") is True
            ):
                return PagedInt8KVTensor(
                    visit(value["key"]),
                    visit(value["value"]),
                    visit(value["key_scales"]),
                    visit(value["value_scales"]),
                    tuple(int(item) for item in value["shape"]),
                    str(value["original_dtype"]),
                )
            if isinstance(value, dict) and "__hydra_ndarray__" in value:
                index = int(value["__hydra_ndarray__"])
                if index in arrays:
                    raise RuntimeError("duplicate ndarray index in shared-memory payload")
                dtype = np.dtype(value["dtype"])
                shape = tuple(int(item) for item in value["shape"])
                nbytes = int(value["nbytes"])
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if nbytes != expected or cursor + nbytes > len(payload):
                    raise RuntimeError("invalid ndarray region in shared-memory payload")
                region = payload[cursor : cursor + nbytes]
                try:
                    array = np.ndarray(shape, dtype=dtype, buffer=region).copy()
                finally:
                    region.release()
                cursor += nbytes
                arrays[index] = array
                return array
            if isinstance(value, dict) and "__hydra_tuple__" in value:
                return tuple(visit(item) for item in value["__hydra_tuple__"])
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            return value

        decoded = visit(metadata)
        if cursor != len(payload):
            raise RuntimeError("unreferenced bytes in shared-memory payload")
        return decoded

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

    def discard(self, key: str, endpoint: int) -> None:
        name = self._mailbox_name(key, endpoint)
        try:
            memory = shared_memory.SharedMemory(name=name, create=False)
        except (FileNotFoundError, ValueError):
            return
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
