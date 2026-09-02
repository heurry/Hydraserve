"""Persistent bounded POSIX shared-memory transport with producer backpressure."""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import json
import os
from multiprocessing import shared_memory
from pathlib import Path
import struct
from tempfile import gettempdir
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Any

import fcntl
import numpy as np

from hydraserve.transfer.backend import (
    SharedMemoryTransferBackend,
    _raise_if_cancelled,
)
from hydraserve.transfer.descriptor import TransferMode


class SharedMemoryRingTransferBackend(SharedMemoryTransferBackend):
    """Reuse fixed SHM slots with bounded producer backpressure.

    A namespace is single-consumer but may have multiple P-worker producers.
    The per-slot file lock protects only the FREE -> WRITING claim across
    processes; payload publication remains lock-free/header-last after that
    claim, so large copies never hold the lock.
    """

    _RING_HEADER = struct.Struct("!8sB7xQQ32s")
    _RING_MAGIC = b"HYDRR003"
    _FREE = 0
    _WRITING = 1
    _READY = 2
    _READING = 3
    _STATE_OFFSET = 8
    _SEGMENT_MARKER = "__hydraserve_ring_segment_v1__"
    _WIRE_PREFIX = struct.Struct("!Q")

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
        # A D process may prepare several requests concurrently. Only one
        # thread scans/claims slots at a time; messages for other request keys
        # are copied out and parked here so they cannot block the ring head.
        self._receive_scan_lock = Lock()
        self._pending_lock = Lock()
        self._pending: dict[bytes, deque[tuple[bool, Any]]] = defaultdict(deque)
        self._cancelled_digests: set[bytes] = set()
        self._dispatcher_start_lock = Lock()
        self._dispatcher_stop = Event()
        self._dispatcher_thread: Thread | None = None
        self._dispatcher_error: Exception | None = None
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
            self._send_segmented(key, metadata_blob, arrays, dst)
            return
        self._send_encoded_slot(key, metadata_blob, arrays, payload_size, dst)

    def _send_segmented(
        self,
        key: str,
        metadata_blob: bytes,
        arrays: tuple[np.ndarray, ...],
        dst: int,
    ) -> None:
        payload_size = sum(array.nbytes for array in arrays)
        wire = bytearray(self._WIRE_PREFIX.size + len(metadata_blob) + payload_size)
        wire[: self._WIRE_PREFIX.size] = self._WIRE_PREFIX.pack(len(metadata_blob))
        cursor = self._WIRE_PREFIX.size
        wire[cursor : cursor + len(metadata_blob)] = metadata_blob
        cursor += len(metadata_blob)
        for array in arrays:
            data = memoryview(array).cast("B")
            try:
                wire[cursor : cursor + array.nbytes] = data
            finally:
                data.release()
            cursor += array.nbytes

        # Leave room for the typed-codec metadata wrapped around each byte
        # segment. Probe the actual encoding so unusually small test rings
        # either choose a safe size or fail explicitly.
        segment_bytes = max(1, self.slot_bytes - 2048)
        while True:
            probe = {
                self._SEGMENT_MARKER: True,
                "index": 0,
                "count": 1,
                "wire_bytes": len(wire),
                "data": np.empty(segment_bytes, dtype=np.uint8),
            }
            probe_metadata, probe_arrays = self._encode(probe)
            probe_blob = json.dumps(
                probe_metadata, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(probe_blob) + sum(item.nbytes for item in probe_arrays) <= self.slot_bytes:
                break
            if segment_bytes == 1:
                raise ValueError("ring slot is too small for segmented transfer metadata")
            segment_bytes = max(1, segment_bytes // 2)

        count = (len(wire) + segment_bytes - 1) // segment_bytes
        for index in range(count):
            start = index * segment_bytes
            end = min(len(wire), start + segment_bytes)
            segment = {
                self._SEGMENT_MARKER: True,
                "index": index,
                "count": count,
                "wire_bytes": len(wire),
                "data": np.frombuffer(wire, dtype=np.uint8, count=end - start, offset=start),
            }
            part_metadata, part_arrays = self._encode(segment)
            part_blob = json.dumps(
                part_metadata, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            part_size = sum(item.nbytes for item in part_arrays)
            self._send_encoded_slot(key, part_blob, part_arrays, part_size, dst)

    def _send_encoded_slot(
        self,
        key: str,
        metadata_blob: bytes,
        arrays: tuple[np.ndarray, ...],
        payload_size: int,
        dst: int,
    ) -> None:
        required = len(metadata_blob) + payload_size
        if required > self.slot_bytes:
            raise ValueError(
                f"segmented transfer part needs {required} bytes, "
                f"ring slot holds {self.slot_bytes}"
            )
        digest = sha256(f"{dst}:{key}".encode("utf-8")).digest()
        deadline = monotonic() + self.send_timeout_s
        with self._send_lock:
            while True:
                for memory, fd in zip(
                    self._memories, self._lock_fds, strict=True
                ):
                    # Different P worker processes can target the same D ring.
                    # Atomically claim a FREE slot before copying. The lock is
                    # released immediately after the one-byte state change.
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    try:
                        if memory.buf[self._STATE_OFFSET] != self._FREE:
                            continue
                        memory.buf[self._STATE_OFFSET] = self._WRITING
                    finally:
                        fcntl.flock(fd, fcntl.LOCK_UN)
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
                    raise TimeoutError(
                        "timed out waiting for a free SHM ring slot: "
                        f"{key}; {self.snapshot()}"
                    )
                sleep(self._poll_interval)

    def _pop_pending(self, digest: bytes) -> tuple[bool, Any] | None:
        with self._pending_lock:
            queue = self._pending.get(digest)
            if not queue:
                return None
            item = queue.popleft()
            if not queue:
                self._pending.pop(digest, None)
            return item

    def _push_pending(self, digest: bytes, item: tuple[bool, Any]) -> None:
        with self._pending_lock:
            if digest in self._cancelled_digests:
                return
            self._pending[digest].append(item)

    def discard(self, key: str, endpoint: int) -> None:
        """Drop a current or future message for a cancelled request phase."""
        self._ensure_dispatcher()
        digest = sha256(f"{endpoint}:{key}".encode("utf-8")).digest()
        with self._pending_lock:
            self._pending.pop(digest, None)
            self._cancelled_digests.add(digest)
        if self._receive_scan_lock.acquire(blocking=False):
            try:
                self._drain_ready_slots()
            finally:
                self._receive_scan_lock.release()

    @staticmethod
    def _unwrap_pending(item: tuple[bool, Any]) -> Any:
        ok, value = item
        if ok:
            return value
        raise value

    def _drain_ready_slots(self, *, only_cancelled: bool = False) -> None:
        """Copy every currently READY message out of the bounded ring.

        This is receive-side demultiplexing: a waiter for request A is allowed
        to drain request B into process memory, freeing slots for A's producer.
        It removes cross-request head-of-line deadlocks without requiring a
        permanently running dispatcher thread.
        """
        for memory, fd in zip(self._memories, self._lock_fds, strict=True):
            if memory.buf[self._STATE_OFFSET] != self._READY:
                continue
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                if memory.buf[self._STATE_OFFSET] != self._READY:
                    continue
                header = self._RING_HEADER.unpack(
                    bytes(memory.buf[: self._RING_HEADER.size])
                )
                magic, state, metadata_size, payload_size, actual = header
                if magic != self._RING_MAGIC or state != self._READY:
                    memory.buf[: self._RING_HEADER.size] = self._pack_free()
                    raise RuntimeError("shared-memory ring header is corrupt")
                if only_cancelled:
                    with self._pending_lock:
                        if actual not in self._cancelled_digests:
                            continue
                memory.buf[self._STATE_OFFSET] = self._READING
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

            item: tuple[bool, Any]
            try:
                if metadata_size + payload_size > self.slot_bytes:
                    raise RuntimeError("shared-memory ring payload is corrupt")
                metadata_start = self._RING_HEADER.size
                payload_start = metadata_start + metadata_size
                metadata = json.loads(
                    bytes(memory.buf[metadata_start:payload_start]).decode("utf-8")
                )
                view = memory.buf[payload_start : payload_start + payload_size]
                try:
                    item = (True, self._decode(metadata, view))
                finally:
                    view.release()
            except Exception as exc:
                item = (False, exc)
            finally:
                # Even malformed JSON/tensor metadata must not strand a slot
                # in READY/READING forever.
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    memory.buf[: self._RING_HEADER.size] = self._pack_free()
                finally:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            self._push_pending(actual, item)

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher_thread is not None:
            return
        with self._dispatcher_start_lock:
            if self._dispatcher_thread is not None:
                return

            def dispatch() -> None:
                while not self._dispatcher_stop.is_set():
                    try:
                        with self._receive_scan_lock:
                            # Preserve bounded backpressure for live requests;
                            # the resident dispatcher exists only to recycle
                            # late arrivals whose receiver was cancelled.
                            self._drain_ready_slots(only_cancelled=True)
                    except Exception as exc:
                        self._dispatcher_error = exc
                        return
                    self._dispatcher_stop.wait(self._poll_interval)

            self._dispatcher_thread = Thread(
                target=dispatch,
                name=f"hydraserve-shm-dispatch-{self.namespace}",
                daemon=True,
            )
            self._dispatcher_thread.start()

    def receive(
        self,
        key: str,
        src: int,
        stream: Any = None,
        timeout: float | None = None,
        cancel_event=None,
    ) -> Any:
        self._ensure_dispatcher()
        deadline = None if timeout is None else monotonic() + timeout

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            return max(0.0, deadline - monotonic())

        first = self._receive_one(
            key,
            src,
            timeout=remaining_timeout(),
            cancel_event=cancel_event,
        )
        if not isinstance(first, dict) or first.get(self._SEGMENT_MARKER) is not True:
            return first
        count = int(first.get("count", 0))
        wire_bytes = int(first.get("wire_bytes", -1))
        if count <= 0 or wire_bytes < self._WIRE_PREFIX.size:
            raise RuntimeError("invalid segmented SHM ring envelope")
        parts = {int(first["index"]): first["data"]}
        while len(parts) < count:
            part = self._receive_one(
                key,
                src,
                timeout=remaining_timeout(),
                cancel_event=cancel_event,
            )
            if (
                not isinstance(part, dict)
                or part.get(self._SEGMENT_MARKER) is not True
                or int(part.get("count", -1)) != count
                or int(part.get("wire_bytes", -1)) != wire_bytes
            ):
                raise RuntimeError("inconsistent segmented SHM ring envelope")
            index = int(part["index"])
            if index in parts or not 0 <= index < count:
                raise RuntimeError("duplicate or invalid SHM ring segment")
            parts[index] = part["data"]
        wire = b"".join(bytes(parts[index]) for index in range(count))
        if len(wire) != wire_bytes:
            raise RuntimeError("segmented SHM ring transfer length mismatch")
        metadata_size = self._WIRE_PREFIX.unpack(wire[: self._WIRE_PREFIX.size])[0]
        metadata_start = self._WIRE_PREFIX.size
        payload_start = metadata_start + metadata_size
        if payload_start > len(wire):
            raise RuntimeError("segmented SHM ring metadata length is corrupt")
        metadata = json.loads(wire[metadata_start:payload_start].decode("utf-8"))
        view = memoryview(wire)[payload_start:]
        try:
            return self._decode(metadata, view)
        finally:
            view.release()

    def _receive_one(
        self,
        key: str,
        src: int,
        stream: Any = None,
        timeout: float | None = None,
        cancel_event=None,
    ) -> Any:
        if src < 0 or not key:
            raise ValueError("invalid transfer source or key")
        digest = sha256(f"{src}:{key}".encode("utf-8")).digest()
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            _raise_if_cancelled(cancel_event)
            if self._dispatcher_error is not None:
                raise RuntimeError("SHM ring receive dispatcher failed") from self._dispatcher_error
            pending = self._pop_pending(digest)
            if pending is not None:
                return self._unwrap_pending(pending)
            if self._receive_scan_lock.acquire(blocking=False):
                try:
                    self._drain_ready_slots()
                finally:
                    self._receive_scan_lock.release()
                pending = self._pop_pending(digest)
                if pending is not None:
                    return self._unwrap_pending(pending)
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for SHM ring transfer {key}; {self.snapshot()}"
                )
            sleep(self._poll_interval)

    def snapshot(self) -> str:
        counts = {
            "free": 0,
            "writing": 0,
            "ready": 0,
            "reading": 0,
            "unknown": 0,
        }
        names = {
            self._FREE: "free",
            self._WRITING: "writing",
            self._READY: "ready",
            self._READING: "reading",
        }
        for memory in self._memories:
            counts[names.get(int(memory.buf[self._STATE_OFFSET]), "unknown")] += 1
        with self._pending_lock:
            buffered = sum(len(queue) for queue in self._pending.values())
            keys = len(self._pending)
        return f"ring={counts}, buffered_messages={buffered}, buffered_keys={keys}"

    def close(self) -> None:
        self._dispatcher_stop.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=2)
            self._dispatcher_thread = None
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
        with self._pending_lock:
            self._pending.clear()
            self._cancelled_digests.clear()
