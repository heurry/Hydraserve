#!/usr/bin/env python3
"""Load-aware reverse proxy for N data-parallel HydraServe workers.

Forwards each request to the backend with the fewest ``scheduler.active_requests``
(as reported by its ``/health`` endpoint), keeping a 4xDP benchmark load balanced
across serve processes. Pure stdlib (``http.server`` + ``urllib``), no dependencies.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import error, request

FORWARDED_HEADERS = ("Content-Type", "Accept", "Authorization", "X-Request-Id")


def _normalize_backend(raw: str) -> str:
    """Accept ``8000``, ``127.0.0.1:8000``, or ``http://host:8000``."""
    value = raw.strip()
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")
    if "://" in value:
        return value.rstrip("/")
    if value.isdigit():
        return f"http://127.0.0.1:{value}"
    return f"http://{value}"


class LoadAwareProxyHandler(BaseHTTPRequestHandler):
    backends: tuple[str, ...] = ()
    health_timeout: float = 1.0
    forward_timeout: float = 600.0
    _pending_lock = threading.Lock()
    _pending: dict[str, int] = {}
    _served: dict[str, int] = {}
    _served_chars: dict[str, int] = {}

    def _pick_backend(self, weight: int = 1) -> str:
        """Pick the backend with the fewest in-flight requests.

        Local in-flight counts avoid the burst thundering-herd: N concurrent
        requests all reading ``/health`` before any forward would all see zero
        load and pile onto the first backend. (The ``weight`` arg is accepted
        but unused: count-based balancing measured a lower tail TPOT than
        KV-occupancy balancing, which starved short requests.)
        """
        with self._pending_lock:
            best = self.backends[0]
            best_pending = self._pending.get(best, 0)
            for backend in self.backends[1:]:
                pending = self._pending.get(backend, 0)
                if pending < best_pending:
                    best, best_pending = backend, pending
            self._pending[best] = best_pending + 1
            self._served[best] = self._served.get(best, 0) + 1
            return best

    def _release_backend(self, backend: str) -> None:
        with self._pending_lock:
            self._pending[backend] = max(0, self._pending.get(backend, 0) - 1)

    def _stats(self) -> None:
        with self._pending_lock:
            stats = {
                backend: {
                    "served": self._served.get(backend, 0),
                    "prompt_chars": self._served_chars.get(backend, 0),
                    "pending": self._pending.get(backend, 0),
                }
                for backend in self.backends
            }
        body = json.dumps(stats).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/stats":
            self._stats()
        else:
            self._forward()

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        prompt_chars = 0
        if body:
            try:
                prompt_chars = len(json.loads(body.decode("utf-8")).get("prompt", ""))
            except Exception:
                prompt_chars = 0
        weight = prompt_chars if prompt_chars > 0 else 1
        backend = self._pick_backend(weight)
        try:
            with self._pending_lock:
                self._served_chars[backend] = (
                    self._served_chars.get(backend, 0) + prompt_chars
                )
            upstream = request.Request(
                f"{backend}{self.path}", data=body, method=self.command
            )
            for header in FORWARDED_HEADERS:
                value = self.headers.get(header)
                if value:
                    upstream.add_header(header, value)
            try:
                with request.urlopen(upstream, timeout=self.forward_timeout) as response:
                    self.send_response(response.status)
                    content_type = response.headers.get("Content-Type")
                    if content_type:
                        self.send_header("Content-Type", content_type)
                    self.end_headers()
                    # Stream line-by-line: the upstream uses a close-delimited
                    # (no Content-Length) response, so a fixed-size read() would
                    # block until the socket closes and batch every SSE event.
                    for raw_line in response:
                        self.wfile.write(raw_line)
                        self.wfile.flush()
            except error.HTTPError as exc:
                self.send_response(exc.code)
                self.end_headers()
                self.wfile.write(exc.read())
            except Exception as exc:  # pragma: no cover - network/unexpected failure
                self.send_response(502)
                self.end_headers()
                self.wfile.write(f"proxy error: {exc}".encode("utf-8"))
        finally:
            self._release_backend(backend)

    do_POST = _forward

    def log_message(self, format, *args):  # noqa: A002 - keep proxy log terse
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Load-aware DP proxy for HydraServe")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--backends", required=True, help="comma-separated backend addresses"
    )
    parser.add_argument("--health-timeout", type=float, default=1.0)
    parser.add_argument("--forward-timeout", type=float, default=600.0)
    args = parser.parse_args()

    LoadAwareProxyHandler.backends = tuple(
        _normalize_backend(part) for part in args.backends.split(",") if part.strip()
    )
    if not LoadAwareProxyHandler.backends:
        parser.error("--backends must list at least one backend")
    LoadAwareProxyHandler.health_timeout = args.health_timeout
    LoadAwareProxyHandler.forward_timeout = args.forward_timeout

    server = ThreadingHTTPServer((args.host, args.port), LoadAwareProxyHandler)
    print(f"load-aware proxy listening on {args.host}:{args.port}", flush=True)
    print(f"backends: {', '.join(LoadAwareProxyHandler.backends)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
