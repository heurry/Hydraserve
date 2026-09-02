"""Metadata-only bootstrap service for PD transfer handshakes."""

from __future__ import annotations

from collections import defaultdict
from threading import Condition
from time import monotonic
import json
import socket
import socketserver
from threading import Thread

from hydraserve.transfer.backend import TransferCancelledError


class BootstrapRegistry:
    """One-shot request metadata registry, separate from the tensor data plane."""

    def __init__(self) -> None:
        self._messages = defaultdict(dict)
        self._cancelled = set()
        self._condition = Condition()

    def publish(self, request_id: int, kind: str, metadata: dict) -> None:
        if request_id < 0 or not kind or not isinstance(metadata, dict):
            raise ValueError("invalid bootstrap metadata")
        with self._condition:
            if (request_id, kind) in self._cancelled:
                raise TransferCancelledError("bootstrap handshake was cancelled")
            if kind in self._messages[request_id]:
                raise RuntimeError("bootstrap metadata was already published")
            self._messages[request_id][kind] = dict(metadata)
            self._condition.notify_all()

    def consume(
        self, request_id: int, kind: str, *, timeout: float | None = None
    ) -> dict:
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while kind not in self._messages.get(request_id, {}):
                if (request_id, kind) in self._cancelled:
                    raise TransferCancelledError("bootstrap handshake was cancelled")
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("bootstrap metadata handshake timed out")
                self._condition.wait(remaining)
            metadata = self._messages[request_id].pop(kind)
            if not self._messages[request_id]:
                self._messages.pop(request_id, None)
            return metadata

    def cancel(self, request_id: int, kind: str) -> None:
        if request_id < 0 or not kind:
            raise ValueError("invalid bootstrap cancellation")
        with self._condition:
            self._cancelled.add((request_id, kind))
            request_messages = self._messages.get(request_id)
            if request_messages is not None:
                request_messages.pop(kind, None)
                if not request_messages:
                    self._messages.pop(request_id, None)
            self._condition.notify_all()


class BootstrapClient:
    """Adapter accepted by TransferPipeline; remote implementations share this API."""

    def __init__(self, registry: BootstrapRegistry) -> None:
        self.registry = registry

    def publish(self, request_id: int, kind: str, metadata: dict) -> None:
        self.registry.publish(request_id, kind, metadata)

    def consume(
        self, request_id: int, kind: str, *, timeout: float | None = None
    ) -> dict:
        return self.registry.consume(request_id, kind, timeout=timeout)

    def cancel(self, request_id: int, kind: str) -> None:
        self.registry.cancel(request_id, kind)


class _BootstrapTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = json.loads(self.rfile.readline().decode("utf-8"))
            registry = self.server.registry  # type: ignore[attr-defined]
            if request["op"] == "publish":
                registry.publish(
                    int(request["request_id"]), request["kind"], request["metadata"]
                )
                response = {"ok": True}
            elif request["op"] == "consume":
                response = {
                    "ok": True,
                    "metadata": registry.consume(
                        int(request["request_id"]),
                        request["kind"],
                        timeout=request.get("timeout"),
                    ),
                }
            elif request["op"] == "cancel":
                registry.cancel(int(request["request_id"]), request["kind"])
                response = {"ok": True}
            else:
                response = {"ok": False, "error": "unknown bootstrap operation"}
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        self.wfile.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")


class BootstrapServer:
    """Small metadata control plane; tensor payloads never pass through it."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.registry = BootstrapRegistry()
        self._server = socketserver.ThreadingTCPServer(
            (host, port), _BootstrapTCPHandler
        )
        self._server.daemon_threads = True
        self._server.registry = self.registry
        self._thread: Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address
        return str(host), int(port)

    def start(self) -> "BootstrapServer":
        if self._thread is None:
            self._thread = Thread(
                target=self._server.serve_forever,
                name="hydraserve-bootstrap",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> "BootstrapServer":
        return self.start()

    def __exit__(self, *_args) -> None:
        self.close()


class NetworkBootstrapClient:
    def __init__(self, address: tuple[str, int]) -> None:
        self.address = address

    def _call(self, payload: dict, timeout: float | None = None) -> dict:
        with socket.create_connection(self.address, timeout=timeout) as connection:
            connection.sendall(
                json.dumps(payload, separators=(",", ":")).encode() + b"\n"
            )
            response = b""
            while not response.endswith(b"\n"):
                part = connection.recv(65536)
                if not part:
                    break
                response += part
        decoded = json.loads(response.decode("utf-8"))
        if not decoded.get("ok"):
            if decoded.get("error_type") == "TransferCancelledError":
                raise TransferCancelledError(decoded.get("error", "cancelled"))
            raise RuntimeError(decoded.get("error", "bootstrap request failed"))
        return decoded

    def publish(self, request_id: int, kind: str, metadata: dict) -> None:
        self._call(
            {
                "op": "publish",
                "request_id": request_id,
                "kind": kind,
                "metadata": metadata,
            }
        )

    def consume(
        self, request_id: int, kind: str, *, timeout: float | None = None
    ) -> dict:
        return self._call(
            {
                "op": "consume",
                "request_id": request_id,
                "kind": kind,
                "timeout": timeout,
            },
            timeout=None if timeout is None else timeout + 0.25,
        )["metadata"]

    def cancel(self, request_id: int, kind: str) -> None:
        self._call(
            {
                "op": "cancel",
                "request_id": request_id,
                "kind": kind,
            }
        )
