"""Persistent bounded POSIX shared-memory transport with producer backpressure."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from multiprocessing import shared_memory
from pathlib import Path
import struct
from tempfile import gettempdir
from threading import Lock
from time import monotonic, sleep
from typing import Any

import fcntl

from hydraserve.transfer.backend import SharedMemoryTransferBackend
from hydraserve.transfer.descriptor import TransferMode


class SharedMemoryRingTransferBackend(SharedMemoryTransferBackend):
    """Reuse fixed SHM slots instead of creating and unlinking every chunk.

    Slot state is protected by a tiny file lock. Payload publication remains
    header-last, and a full ring naturally applies backpressure to prefill.
    """

    _RING_HEADER = struct.Struct("!8sB7xQQ32s")
    _RING_MAGIC = b"HYDRR003"
    _FREE = 0
    _WRITING = 1
    _READY = 2
    _STATE_OFFSET = 8

    def __init__(
        self,
        namespace: str = "hydraserve",
        *,
        slots: int = 3,
        slot_bytes: int = 64 << 20,
        poll_interval_s: float = 0.0005,
        send_timeout_s: float = 600.0,
        mode: TransferMode = TransferMode.FULL_TRANSFER,
    ) -> None:
        if slots <= 0 or slot_bytes <= 0 or send_timeout_s <= 0:
            raise ValueError("invalid shared-memory ring configuration")
        super().__init__(
            namespace=namespace,
            poll_interval_s=poll_interval_s,
            mode=mode,
        )
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.send_timeout_s = send_timeout_s
        self._send_lock = Lock()
        self._memories: list[shared_memory.SharedMemory] = []
        self._lock_fds: list[int] = []
        self._ring_owned: set[str] = set()
        self._open_ring()

    def _open_ring(self) -> None:
        lock_root = Path(gettempdir()) / "hydraserve-shm-ring"
        lock_root.mkdir(mode=0o700, exist_ok=True)
        namespace_hash = sha256(self.namespace.encode("utf-8")).hexdigest()[:20]
        total_size = self._RING_HEADER.size + self.slot_bytes
        for slot in range(self.slots):
            lock_path = lock_root / f"{namespace_hash}-{slot}.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            self._lock_fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                name = f"{self.namespace}-ring-{slot}"
                try:
                    memory = shared_memory.SharedMemory(
                        name=name, create=True, size=total_size
                    )
                    self._ring_owned.add(name)
                    memory.buf[: self._RING_HEADER.size] = self._pack_free()
                except FileExistsError:
                    memory = shared_memory.SharedMemory(name=name, create=False)
                    if len(memory.buf) != total_size:
                        memory.close()
                        raise RuntimeError(
                            f"shared-memory ring size mismatch for {name}"
                        )
                    magic = bytes(memory.buf[:8])
                    if magic != self._RING_MAGIC:
                        memory.close()
                        raise RuntimeError(f"invalid shared-memory ring header for {name}")
                self._memories.append(memory)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

    @classmethod
    def _pack_free(cls) -> bytes:
        return cls._RING_HEADER.pack(cls._RING_MAGIC, cls._FREE, 0, 0, b"\0" * 32)

    def send(self, key: str, payload: Any, dst: int, stream: Any = None) -> None:
        if dst < 0 or not key:
            raise ValueError("invalid transfer destination or key")
        metadata, arrays = self._encode(payload)
        metadata_blob = json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        payload_size = sum(array.nbytes for array in arrays)
        required = len(metadata_blob) + payload_size
        if required > self.slot_bytes:
            raise ValueError(
                f"transfer needs {required} bytes, ring slot holds {self.slot_bytes}"
            )
        digest = sha256(f"{dst}:{key}".encode("utf-8")).digest()
        deadline = monotonic() + self.send_timeout_s
        with self._send_lock:
            while True:
                for memory in self._memories:
                    if memory.buf[self._STATE_OFFSET] != self._FREE:
                        continue
                    # Each namespace is SPSC. Reserve with a byte store, write
                    # metadata/payload, then publish READY with a final byte
                    # store. Startup still uses flock while creating slots.
                    memory.buf[self._STATE_OFFSET] = self._WRITING
                    memory.buf[: self._RING_HEADER.size] = self._RING_HEADER.pack(
                        self._RING_MAGIC,
                        self._WRITING,
                        len(metadata_blob),
                        payload_size,
                        digest,
                    )
                    cursor = self._RING_HEADER.size
                    memory.buf[cursor : cursor + len(metadata_blob)] = metadata_blob
                    cursor += len(metadata_blob)
                    for array in arrays:
                        data = memoryview(array).cast("B")
                        try:
                            memory.buf[cursor : cursor + array.nbytes] = data
                        finally:
                            data.release()
                        cursor += array.nbytes
                    memory.buf[self._STATE_OFFSET] = self._READY
                    return
                if monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for a free SHM ring slot: {key}")
                sleep(self._poll_interval)

    def receive(
        self, key: str, src: int, stream: Any = None, timeout: float | None = None
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        digest = sha256(f"{src}:{key}".encode("utf-8")).digest()
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            for memory in self._memories:
                if memory.buf[self._STATE_OFFSET] != self._READY:
                    continue
                magic, state, metadata_size, payload_size, actual = (
                    self._RING_HEADER.unpack(
                        bytes(memory.buf[: self._RING_HEADER.size])
                    )
                )
                if magic != self._RING_MAGIC:
                    raise RuntimeError("shared-memory ring header is corrupt")
                if state != self._READY or actual != digest:
                    continue
                if metadata_size + payload_size > self.slot_bytes:
                    raise RuntimeError("shared-memory ring payload is corrupt")
                metadata_start = self._RING_HEADER.size
                payload_start = metadata_start + metadata_size
                metadata = json.loads(
                    bytes(memory.buf[metadata_start:payload_start]).decode("utf-8")
                )
                view = memory.buf[payload_start : payload_start + payload_size]
                try:
                    result = self._decode(metadata, view)
                finally:
                    view.release()
                memory.buf[self._STATE_OFFSET] = self._FREE
                return result
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for SHM ring transfer {key}")
            sleep(self._poll_interval)

    def close(self) -> None:
        for memory in self._memories:
            name = memory.name
            memory.close()
            if name in self._ring_owned:
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
        self._memories.clear()
        for fd in self._lock_fds:
            os.close(fd)
        self._lock_fds.clear()
        self._ring_owned.clear()
