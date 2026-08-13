"""Small dependency-light OpenAI-compatible completions server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from time import time
from typing import Any
from uuid import uuid4

from hydraserve.engine.serving_loop import OverloadedError
from hydraserve.model.tokenizer import IncrementalTextDecoder


class HydraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, generation_loop, tokenizer, model_name: str):
        super().__init__(address, handler)
        self.generation_loop = generation_loop
        self.tokenizer = tokenizer
        self.model_name = model_name


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "HydraServe/0.1"

    @property
    def hydra(self) -> HydraHTTPServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        if self.path == "/health":
            payload: dict[str, Any] = {"status": "ok"}
            capacity = getattr(self.hydra.generation_loop.backend, "capacity", None)
            if capacity is not None:
                snapshot = capacity()
                payload["capacity"] = {
                    "kv_total_blocks": snapshot.kv_total_blocks,
                    "kv_free_blocks": snapshot.kv_free_blocks,
                    "state_total_slots": snapshot.state_total_slots,
                    "state_free_slots": snapshot.state_free_slots,
                    "decode_load": snapshot.decode_load,
                }
            recovery_stats = getattr(
                self.hydra.generation_loop.backend, "recovery_stats", None
            )
            if recovery_stats is not None:
                recovery = recovery_stats()
                payload["decode_workers"] = {
                    "total": recovery.total_workers,
                    "healthy": recovery.healthy_workers,
                    "recovering": list(recovery.recovering_workers),
                    "restart_attempts": recovery.attempts,
                    "restart_successes": recovery.successes,
                    "restart_failures": recovery.failures,
                }
                if recovery.healthy_workers < recovery.total_workers:
                    payload["status"] = "degraded"
            self._json(200, payload)
            return
        if self.path == "/metrics":
            self._metrics()
            return
        if self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.hydra.model_name,
                            "object": "model",
                            "owned_by": "hydraserve",
                        }
                    ],
                },
            )
            return
        self._error(404, "not_found", f"unknown endpoint: {self.path}")

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/v1/completions":
                prompt = payload.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                chat = False
            elif self.path == "/v1/chat/completions":
                messages = payload.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                prompt = self.hydra.tokenizer.render_chat(messages)
                chat = True
            else:
                self._error(404, "not_found", f"unknown endpoint: {self.path}")
                return
            self._validate_model(payload)
            max_tokens = payload.get("max_tokens", 16)
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            temperature = payload.get("temperature", 0)
            if temperature not in (0, 0.0, None):
                raise ValueError("this runtime currently supports greedy temperature=0 only")
            prompt_ids = self.hydra.tokenizer.encode(prompt)
            handle = self.hydra.generation_loop.submit(prompt_ids, max_tokens)
            if payload.get("stream", False):
                self._stream(handle, prompt_ids, chat=chat)
            else:
                self._complete(handle, prompt_ids, chat=chat)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request_error", str(exc))
        except OverloadedError as exc:
            self._error(429, "overloaded_error", str(exc))
        except Exception as exc:
            self._error(500, "server_error", str(exc))

    def _validate_model(self, payload: dict[str, Any]) -> None:
        requested = payload.get("model", self.hydra.model_name)
        if requested != self.hydra.model_name:
            raise ValueError(f"unknown model {requested!r}")

    def _complete(self, handle, prompt_ids, *, chat: bool) -> None:
        decoder = IncrementalTextDecoder(self.hydra.tokenizer)
        finish_reason = None
        for event in handle:
            if event.error:
                raise RuntimeError(event.error)
            if event.token_id is not None:
                decoder.push(event.token_id)
            if event.finished:
                finish_reason = event.finish_reason
        choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": decoder.text}
            object_name = "chat.completion"
        else:
            choice["text"] = decoder.text
            object_name = "text_completion"
        self._json(
            200,
            {
                "id": f"cmpl-{uuid4().hex}",
                "object": object_name,
                "created": int(time()),
                "model": self.hydra.model_name,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(decoder.token_ids),
                    "total_tokens": len(prompt_ids) + len(decoder.token_ids),
                },
            },
        )

    def _stream(self, handle, prompt_ids, *, chat: bool) -> None:
        response_id = f"cmpl-{uuid4().hex}"
        created = int(time())
        decoder = IncrementalTextDecoder(self.hydra.tokenizer)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in handle:
                if event.error:
                    self._sse({"error": {"message": event.error, "type": "server_error"}})
                    break
                if event.token_id is not None:
                    delta = decoder.push(event.token_id)
                    choice: dict[str, Any] = {"index": 0, "finish_reason": None}
                    if chat:
                        choice["delta"] = {"content": delta}
                        object_name = "chat.completion.chunk"
                    else:
                        choice["text"] = delta
                        object_name = "text_completion"
                    self._sse(
                        {
                            "id": response_id,
                            "object": object_name,
                            "created": created,
                            "model": self.hydra.model_name,
                            "choices": [choice],
                        }
                    )
                if event.finished:
                    tail = decoder.finish()
                    if tail:
                        choice = {"index": 0, "finish_reason": None}
                        if chat:
                            choice["delta"] = {"content": tail}
                        else:
                            choice["text"] = tail
                        self._sse(
                            {
                                "id": response_id,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "created": created,
                                "model": self.hydra.model_name,
                                "choices": [choice],
                            }
                        )
                    choice = {"index": 0, "finish_reason": event.finish_reason}
                    choice["delta" if chat else "text"] = {} if chat else ""
                    self._sse(
                        {
                            "id": response_id,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created,
                            "model": self.hydra.model_name,
                            "choices": [choice],
                        }
                    )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            handle.cancel()
        finally:
            self.close_connection = True

    def _read_json(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Content-Length is required")
        length = int(content_length)
        if length <= 0 or length > 16 * 1024 * 1024:
            raise ValueError("invalid request body size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _metrics(self) -> None:
        loop = self.hydra.generation_loop
        backend = loop.backend
        lines = [
            "# TYPE hydraserve_admission_pending_requests gauge",
            f"hydraserve_admission_pending_requests {loop.pending_count}",
            "# TYPE hydraserve_admission_pending_tokens gauge",
            f"hydraserve_admission_pending_tokens {loop.pending_tokens}",
        ]
        capacity = getattr(backend, "capacity", None)
        if capacity is not None:
            snapshot = capacity()
            lines.extend(
                [
                    "# TYPE hydraserve_kv_blocks gauge",
                    f'hydraserve_kv_blocks{{state="total"}} {snapshot.kv_total_blocks}',
                    f'hydraserve_kv_blocks{{state="free"}} {snapshot.kv_free_blocks}',
                    "# TYPE hydraserve_recurrent_state_slots gauge",
                    f'hydraserve_recurrent_state_slots{{state="total"}} {snapshot.state_total_slots}',
                    f'hydraserve_recurrent_state_slots{{state="free"}} {snapshot.state_free_slots}',
                    "# TYPE hydraserve_decode_load gauge",
                    f"hydraserve_decode_load {snapshot.decode_load}",
                ]
            )
        routing_stats = getattr(backend, "routing_stats", None)
        if routing_stats is not None:
            stats = routing_stats()
            lines.extend(
                [
                    "# TYPE hydraserve_routed_requests_total counter",
                    f'hydraserve_routed_requests_total{{route="collocated"}} {stats.collocated}',
                    f'hydraserve_routed_requests_total{{route="pd_disaggregated"}} {stats.pd_disaggregated}',
                    "# TYPE hydraserve_pd_failures_total counter",
                    f"hydraserve_pd_failures_total {stats.pd_failures}",
                    "# TYPE hydraserve_prefill_worker_healthy gauge",
                    f"hydraserve_prefill_worker_healthy {1 if stats.prefill_healthy else 0}",
                ]
            )
        recovery_stats = getattr(backend, "recovery_stats", None)
        if recovery_stats is not None:
            stats = recovery_stats()
            lines.extend(
                [
                    "# TYPE hydraserve_decode_workers gauge",
                    f'hydraserve_decode_workers{{state="total"}} {stats.total_workers}',
                    f'hydraserve_decode_workers{{state="healthy"}} {stats.healthy_workers}',
                    f'hydraserve_decode_workers{{state="recovering"}} {len(stats.recovering_workers)}',
                    "# TYPE hydraserve_worker_restarts_total counter",
                    f'hydraserve_worker_restarts_total{{outcome="attempt"}} {stats.attempts}',
                    f'hydraserve_worker_restarts_total{{outcome="success"}} {stats.successes}',
                    f'hydraserve_worker_restarts_total{{outcome="failure"}} {stats.failures}',
                ]
            )
        body = ("\n".join(lines) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        )
        self.wfile.flush()

    def _error(self, status: int, error_type: str, message: str) -> None:
        self._json(status, {"error": {"message": message, "type": error_type}})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    host: str,
    port: int,
    *,
    generation_loop,
    tokenizer,
    model_name: str,
) -> HydraHTTPServer:
    return HydraHTTPServer(
        (host, port),
        _Handler,
        generation_loop=generation_loop,
        tokenizer=tokenizer,
        model_name=model_name,
    )
