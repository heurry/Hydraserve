"""Small dependency-light OpenAI-compatible completions server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from time import time
from typing import Any
from uuid import uuid4

from hydraserve.engine.serving_loop import OverloadedError
from hydraserve.engine.sampling import SamplingParams
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
            loop = self.hydra.generation_loop
            payload: dict[str, Any] = {
                "status": "ok",
                "scheduler": {
                    "max_batch_size": loop.max_batch_size,
                    "max_active_requests": loop.max_active_requests,
                    "active_requests": loop.active_count,
                    "prefill_pending_requests": loop.prefill_pending_count,
                    "admission_pending_requests": loop.pending_count,
                    "preempted_requests": loop.preempted_count,
                    "preemptions_total": loop.preemptions_total,
                    "preemption_failures_total": loop.preemption_failures_total,
                    "recoveries_total": loop.recoveries_total,
                    "recovery_failures_total": loop.recovery_failures_total,
                    "fault_suspensions_total": loop.fault_suspensions_total,
                },
            }
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
            prefill_recovery_stats = getattr(
                self.hydra.generation_loop.backend, "prefill_recovery_stats", None
            )
            if prefill_recovery_stats is not None:
                recovery = prefill_recovery_stats()
                payload["prefill_worker"] = {
                    "healthy": recovery.healthy,
                    "recovering": recovery.recovering,
                    "restart_attempts": recovery.attempts,
                    "restart_successes": recovery.successes,
                    "restart_failures": recovery.failures,
                }
                if not recovery.healthy:
                    payload["status"] = "degraded"
            cache_stats = getattr(
                self.hydra.generation_loop.backend, "cache_stats", None
            )
            if cache_stats is not None:
                payload["kv_cache"] = cache_stats()
            routing_cost_stats = getattr(
                self.hydra.generation_loop.backend, "routing_cost_stats", None
            )
            if routing_cost_stats is not None:
                costs = routing_cost_stats()
                if costs is not None:
                    payload["routing_cost_model"] = {
                        "collocated_observations": costs.collocated_observations,
                        "pd_observations": costs.pd_observations,
                        "collocated_correction": costs.collocated_correction,
                        "pd_correction": costs.pd_correction,
                        "collocated_drifted_buckets": list(
                            costs.collocated_drifted_buckets
                        ),
                        "pd_drifted_buckets": list(costs.pd_drifted_buckets),
                    }
                    if (
                        costs.collocated_drifted_buckets
                        or costs.pd_drifted_buckets
                    ):
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
            n = payload.get("n", 1)
            best_of = payload.get("best_of", 1)
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (n, best_of)
            ):
                raise ValueError("n and best_of must be integers")
            if n != 1 or best_of != 1:
                raise ValueError("this runtime currently supports exactly one choice")
            unsupported = set(payload) & {
                "tools",
                "tool_choice",
                "functions",
                "parallel_tool_calls",
                "suffix",
                "echo",
                "logit_bias",
            }
            if unsupported:
                raise ValueError(
                    f"unsupported request field(s): {', '.join(sorted(unsupported))}"
                )
            max_tokens = payload.get(
                "max_tokens", payload.get("max_completion_tokens", 16)
            )
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            prompt_ids = self.hydra.tokenizer.encode(prompt)
            context_limit = getattr(self.hydra.tokenizer, "model_max_length", None)
            if context_limit and len(prompt_ids) + max_tokens > context_limit:
                raise ValueError(
                    f"prompt plus max_tokens exceeds context limit {context_limit}"
                )
            sampling = self._sampling_params(payload, chat=chat)
            priority = payload.get("priority", 0)
            if not isinstance(priority, int) or isinstance(priority, bool):
                raise ValueError("priority must be an integer")
            timeout_ms = payload.get("timeout_ms")
            if timeout_ms is not None and (
                not isinstance(timeout_ms, (int, float))
                or isinstance(timeout_ms, bool)
                or timeout_ms <= 0
            ):
                raise ValueError("timeout_ms must be a positive number")
            stream = payload.get("stream", False)
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")
            ignore_eos = payload.get("ignore_eos", False)
            if not isinstance(ignore_eos, bool):
                raise ValueError("ignore_eos must be a boolean")
            stream_options = payload.get("stream_options")
            include_usage = False
            if stream_options is not None:
                if not stream or not isinstance(stream_options, dict):
                    raise ValueError("stream_options requires stream=true and an object")
                unknown = set(stream_options) - {"include_usage"}
                if unknown:
                    raise ValueError("unsupported stream_options field")
                include_usage = stream_options.get("include_usage", False)
                if not isinstance(include_usage, bool):
                    raise ValueError("stream_options.include_usage must be a boolean")
            handle = self.hydra.generation_loop.submit(
                prompt_ids,
                max_tokens,
                ignore_eos=ignore_eos,
                priority=priority,
                sampling_params=sampling,
                timeout_ms=timeout_ms,
            )
            if stream:
                self._stream(
                    handle,
                    prompt_ids,
                    chat=chat,
                    include_usage=include_usage,
                )
            else:
                self._complete(handle, prompt_ids, chat=chat)
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_request_error", str(exc))
        except OverloadedError as exc:
            self._error(429, "overloaded_error", str(exc))
        except RuntimeError as exc:
            if "deadline expired" in str(exc):
                self._error(408, "timeout_error", str(exc))
            else:
                self._error(500, "server_error", str(exc))
        except Exception as exc:
            self._error(500, "server_error", str(exc))

    def _validate_model(self, payload: dict[str, Any]) -> None:
        requested = payload.get("model", self.hydra.model_name)
        if requested != self.hydra.model_name:
            raise ValueError(f"unknown model {requested!r}")

    def _sampling_params(self, payload: dict[str, Any], *, chat: bool) -> SamplingParams:
        stop = payload.get("stop")
        if isinstance(stop, str):
            stop = (stop,)
        elif isinstance(stop, list) and all(isinstance(item, str) for item in stop):
            stop = tuple(stop)
        elif stop is None:
            stop = ()
        else:
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > 4 or any(not item for item in stop):
            raise ValueError("stop supports one to four non-empty strings")
        stop_sequences = tuple(tuple(self.hydra.tokenizer.encode(item)) for item in stop)

        if chat:
            requested = payload.get("logprobs", False)
            if not isinstance(requested, bool):
                raise ValueError("chat logprobs must be a boolean")
            top_logprobs = payload.get("top_logprobs", 0)
            if not isinstance(top_logprobs, int) or isinstance(top_logprobs, bool):
                raise ValueError("top_logprobs must be an integer")
            if top_logprobs and not requested:
                raise ValueError("top_logprobs requires logprobs=true")
            logprobs = top_logprobs if requested else None
        else:
            if "top_logprobs" in payload:
                raise ValueError("top_logprobs is only valid for chat completions")
            logprobs = payload.get("logprobs")
            if logprobs is not None and (
                not isinstance(logprobs, int) or isinstance(logprobs, bool)
            ):
                raise ValueError("completion logprobs must be an integer")

        return SamplingParams(
            temperature=self._number(payload, "temperature", 1.0),
            top_p=self._number(payload, "top_p", 1.0),
            top_k=self._integer(payload, "top_k", 0),
            min_p=self._number(payload, "min_p", 0.0),
            repetition_penalty=self._number(payload, "repetition_penalty", 1.0),
            presence_penalty=self._number(payload, "presence_penalty", 0.0),
            frequency_penalty=self._number(payload, "frequency_penalty", 0.0),
            seed=self._optional_integer(payload, "seed"),
            logprobs=logprobs,
            stop_token_sequences=stop_sequences,
        )

    @staticmethod
    def _integer(payload: dict[str, Any], name: str, default: int) -> int:
        value = payload.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        return value

    @staticmethod
    def _number(payload: dict[str, Any], name: str, default: float) -> float:
        value = payload.get(name, default)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a number")
        return float(value)

    @classmethod
    def _optional_integer(cls, payload: dict[str, Any], name: str) -> int | None:
        value = payload.get(name)
        return None if value is None else cls._integer(payload, name, 0)

    def _complete(self, handle, prompt_ids, *, chat: bool) -> None:
        decoder = IncrementalTextDecoder(self.hydra.tokenizer)
        finish_reason = None
        token_events = []
        for event in handle:
            if event.error:
                raise RuntimeError(event.error)
            if event.token_id is not None:
                token_events.append(event)
            if event.finished:
                finish_reason = event.finish_reason
        visible = self._visible_events(handle, token_events, finish_reason)
        for event in visible:
            decoder.push(event.token_id)
        choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": decoder.text}
            object_name = "chat.completion"
        else:
            choice["text"] = decoder.text
            object_name = "text_completion"
        if handle.request.sampling_params.logprobs is not None:
            choice["logprobs"] = self._format_logprobs(visible, chat=chat)
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
                    "completion_tokens": len(token_events),
                    "total_tokens": len(prompt_ids) + len(token_events),
                },
            },
        )

    def _stream(self, handle, prompt_ids, *, chat: bool, include_usage: bool) -> None:
        response_id = f"cmpl-{uuid4().hex}"
        created = int(time())
        decoder = IncrementalTextDecoder(self.hydra.tokenizer)
        pending = []
        generated_count = 0
        max_hold = max(
            (len(sequence) for sequence in handle.request.sampling_params.stop_token_sequences),
            default=0,
        )
        if (
            not handle.request.ignore_eos
            and self.hydra.generation_loop.eos_token_id is not None
        ):
            max_hold = max(1, max_hold)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in handle:
                if event.error:
                    error_type = (
                        "timeout_error"
                        if "deadline expired" in event.error
                        else "server_error"
                    )
                    self._sse(
                        {"error": {"message": event.error, "type": error_type}}
                    )
                    break
                if event.token_id is not None:
                    generated_count += 1
                    pending.append(event)
                    while len(pending) > max_hold:
                        self._stream_token(
                            pending.pop(0), decoder, response_id, created, chat
                        )
                if event.finished:
                    visible = self._visible_events(handle, pending, event.finish_reason)
                    for pending_event in visible:
                        self._stream_token(
                            pending_event, decoder, response_id, created, chat
                        )
                    pending.clear()
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
            if include_usage:
                self._sse(
                    {
                        "id": response_id,
                        "object": (
                            "chat.completion.chunk" if chat else "text_completion"
                        ),
                        "created": created,
                        "model": self.hydra.model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": generated_count,
                            "total_tokens": len(prompt_ids) + generated_count,
                        },
                    }
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            handle.cancel()
        finally:
            self.close_connection = True

    def _stream_token(self, event, decoder, response_id, created, chat) -> None:
        delta = decoder.push(event.token_id)
        choice: dict[str, Any] = {"index": 0, "finish_reason": None}
        if chat:
            choice["delta"] = {"content": delta}
            object_name = "chat.completion.chunk"
        else:
            choice["text"] = delta
            object_name = "text_completion"
        if event.logprob is not None:
            choice["logprobs"] = self._format_logprobs((event,), chat=chat)
        self._sse(
            {
                "id": response_id,
                "object": object_name,
                "created": created,
                "model": self.hydra.model_name,
                "choices": [choice],
            }
        )

    def _visible_events(self, handle, events, finish_reason):
        visible = list(events)
        if finish_reason != "stop" or not visible:
            return visible
        token_ids = tuple(event.token_id for event in visible)
        matches = [
            len(sequence)
            for sequence in handle.request.sampling_params.stop_token_sequences
            if len(token_ids) >= len(sequence) and token_ids[-len(sequence) :] == sequence
        ]
        eos = self.hydra.generation_loop.eos_token_id
        if not handle.request.ignore_eos and eos is not None and token_ids[-1] == eos:
            matches.append(1)
        trim = max(matches, default=0)
        return visible[:-trim] if trim else visible

    def _format_logprobs(self, events, *, chat: bool):
        if chat:
            return {
                "content": [
                    {
                        **self._chat_logprob(event.token_id, event.logprob),
                        "top_logprobs": [
                            self._chat_logprob(token_id, logprob)
                            for token_id, logprob in event.top_logprobs
                        ],
                    }
                    for event in events
                ]
            }
        tokens = [self.hydra.tokenizer.decode((event.token_id,)) for event in events]
        offsets = []
        cursor = 0
        for token in tokens:
            offsets.append(cursor)
            cursor += len(token)
        return {
            "tokens": tokens,
            "token_logprobs": [event.logprob for event in events],
            "top_logprobs": [
                {
                    self.hydra.tokenizer.decode((token_id,)): logprob
                    for token_id, logprob in event.top_logprobs
                }
                for event in events
            ],
            "text_offset": offsets,
        }

    def _chat_logprob(self, token_id: int, logprob: float | None) -> dict[str, Any]:
        token = self.hydra.tokenizer.decode((token_id,))
        return {
            "token": token,
            "logprob": logprob,
            "bytes": list(token.encode("utf-8")),
        }

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
            "# TYPE hydraserve_scheduler_requests gauge",
            f'hydraserve_scheduler_requests{{state="active"}} {loop.active_count}',
            f'hydraserve_scheduler_requests{{state="prefill_pending"}} {loop.prefill_pending_count}',
            f'hydraserve_scheduler_requests{{state="admission_pending"}} {loop.pending_count}',
            f'hydraserve_scheduler_requests{{state="preempted"}} {loop.preempted_count}',
            "# TYPE hydraserve_scheduler_preemptions_total counter",
            f'hydraserve_scheduler_preemptions_total{{outcome="success"}} {loop.preemptions_total}',
            f'hydraserve_scheduler_preemptions_total{{outcome="failure"}} {loop.preemption_failures_total}',
            "# TYPE hydraserve_scheduler_recoveries_total counter",
            f'hydraserve_scheduler_recoveries_total{{outcome="success"}} {loop.recoveries_total}',
            f'hydraserve_scheduler_recoveries_total{{outcome="failure"}} {loop.recovery_failures_total}',
            "# TYPE hydraserve_scheduler_fault_suspensions_total counter",
            f"hydraserve_scheduler_fault_suspensions_total {loop.fault_suspensions_total}",
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
        cache_stats = getattr(backend, "cache_stats", None)
        if cache_stats is not None:
            stats = cache_stats()
            if stats:
                state_workspace_metrics = []
                if "state_workspace_slots" in stats:
                    state_workspace_metrics = [
                        "# TYPE hydraserve_recurrent_state_workspace_slots gauge",
                        "hydraserve_recurrent_state_workspace_slots "
                        f"{stats['state_workspace_slots']}",
                        "# TYPE hydraserve_recurrent_state_memory_bytes gauge",
                        'hydraserve_recurrent_state_memory_bytes{kind="storage"} '
                        f"{stats.get('state_storage_bytes', 0)}",
                        'hydraserve_recurrent_state_memory_bytes{kind="workspace"} '
                        f"{stats.get('state_workspace_bytes', 0)}",
                    ]
                lines.extend(
                    [
                        *state_workspace_metrics,
                        "# TYPE hydraserve_kv_cache_blocks gauge",
                        *(
                            f'hydraserve_kv_cache_blocks{{state="{state}"}} '
                            f"{stats.get(key, 0)}"
                            for state, key in (
                                ("physical_total", "physical_total_blocks"),
                                ("requested", "requested_physical_blocks"),
                                ("physical_free", "physical_free_blocks"),
                                ("usable_total", "usable_total_blocks"),
                                ("allocatable_free", "allocatable_free_blocks"),
                                ("headroom", "headroom_blocks"),
                                ("allocated", "allocated_blocks"),
                                ("shared", "shared_blocks"),
                                ("high_watermark", "high_watermark_blocks"),
                            )
                        ),
                        "# TYPE hydraserve_kv_cache_memory_bytes gauge",
                        'hydraserve_kv_cache_memory_bytes{kind="allocated"} '
                        f"{stats.get('physical_cache_bytes', 0)}",
                        'hydraserve_kv_cache_memory_bytes{kind="reserved"} '
                        f"{stats.get('memory_reserved_bytes', 0)}",
                        "# TYPE hydraserve_kv_cache_memory_clamped gauge",
                        "hydraserve_kv_cache_memory_clamped "
                        f"{stats.get('memory_clamped', 0)}",
                        "# TYPE hydraserve_kv_allocation_failures_total counter",
                        f"hydraserve_kv_allocation_failures_total {stats.get('allocation_failures', 0)}",
                        "# TYPE hydraserve_kv_internal_fragmentation_tokens gauge",
                        "hydraserve_kv_internal_fragmentation_tokens "
                        f"{stats.get('internal_fragmentation_tokens', 0)}",
                        "# TYPE hydraserve_prefix_cache_blocks gauge",
                        'hydraserve_prefix_cache_blocks{state="cached"} '
                        f"{stats.get('prefix_cached_blocks', 0)}",
                        'hydraserve_prefix_cache_blocks{state="referenced"} '
                        f"{stats.get('prefix_referenced_blocks', 0)}",
                        'hydraserve_prefix_cache_blocks{state="evictable"} '
                        f"{stats.get('prefix_evictable_blocks', 0)}",
                        "# TYPE hydraserve_prefix_cache_events_total counter",
                        'hydraserve_prefix_cache_events_total{event="hit"} '
                        f"{stats.get('prefix_hits', 0)}",
                        'hydraserve_prefix_cache_events_total{event="miss"} '
                        f"{stats.get('prefix_misses', 0)}",
                        'hydraserve_prefix_cache_events_total{event="admission"} '
                        f"{stats.get('prefix_admissions', 0)}",
                        'hydraserve_prefix_cache_events_total{event="rejected_admission"} '
                        f"{stats.get('prefix_rejected_admissions', 0)}",
                        'hydraserve_prefix_cache_events_total{event="eviction"} '
                        f"{stats.get('prefix_evictions', 0)}",
                        "# TYPE hydraserve_prefix_cache_evictions_total counter",
                        'hydraserve_prefix_cache_evictions_total{reason="active_pressure"} '
                        f"{stats.get('prefix_evicted_active_pressure', 0)}",
                        'hydraserve_prefix_cache_evictions_total{reason="cache_capacity"} '
                        f"{stats.get('prefix_evicted_cache_capacity', 0)}",
                        'hydraserve_prefix_cache_evictions_total{reason="manual"} '
                        f"{stats.get('prefix_evicted_manual', 0)}",
                        "# TYPE hydraserve_prefix_cache_rejections_total counter",
                        'hydraserve_prefix_cache_rejections_total{reason="frequency"} '
                        f"{stats.get('prefix_rejected_frequency', 0)}",
                        'hydraserve_prefix_cache_rejections_total{reason="capacity"} '
                        f"{stats.get('prefix_rejected_capacity', 0)}",
                        'hydraserve_prefix_cache_rejections_total{reason="size"} '
                        f"{stats.get('prefix_rejected_size', 0)}",
                        'hydraserve_prefix_cache_rejections_total{reason="length"} '
                        f"{stats.get('prefix_rejected_length', 0)}",
                        "# TYPE hydraserve_prefix_cache_hit_tokens_total counter",
                        f"hydraserve_prefix_cache_hit_tokens_total {stats.get('prefix_hit_tokens', 0)}",
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
        routing_cost_stats = getattr(backend, "routing_cost_stats", None)
        if routing_cost_stats is not None:
            stats = routing_cost_stats()
            if stats is not None:
                lines.extend(
                    [
                        "# TYPE hydraserve_route_cost_observations_total counter",
                        'hydraserve_route_cost_observations_total{route="collocated"} '
                        f"{stats.collocated_observations}",
                        'hydraserve_route_cost_observations_total{route="pd_disaggregated"} '
                        f"{stats.pd_observations}",
                        "# TYPE hydraserve_route_cost_correction_ratio gauge",
                        'hydraserve_route_cost_correction_ratio{route="collocated"} '
                        f"{stats.collocated_correction}",
                        'hydraserve_route_cost_correction_ratio{route="pd_disaggregated"} '
                        f"{stats.pd_correction}",
                        "# TYPE hydraserve_route_cost_profile_drift gauge",
                        'hydraserve_route_cost_profile_drift{route="collocated"} '
                        f"{1 if stats.collocated_drifted_buckets else 0}",
                        'hydraserve_route_cost_profile_drift{route="pd_disaggregated"} '
                        f"{1 if stats.pd_drifted_buckets else 0}",
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
        prefill_recovery_stats = getattr(backend, "prefill_recovery_stats", None)
        if prefill_recovery_stats is not None:
            stats = prefill_recovery_stats()
            lines.extend(
                [
                    "# TYPE hydraserve_prefill_worker_recovering gauge",
                    f"hydraserve_prefill_worker_recovering {1 if stats.recovering else 0}",
                    "# TYPE hydraserve_prefill_worker_restarts_total counter",
                    'hydraserve_prefill_worker_restarts_total{outcome="attempt"} '
                    f"{stats.attempts}",
                    'hydraserve_prefill_worker_restarts_total{outcome="success"} '
                    f"{stats.successes}",
                    'hydraserve_prefill_worker_restarts_total{outcome="failure"} '
                    f"{stats.failures}",
                ]
            )
        validation_stats = getattr(backend, "transfer_validation_stats", None)
        if validation_stats is not None:
            stats = validation_stats()
            lines.extend(
                [
                    "# TYPE hydraserve_pd_replay_mismatches_total counter",
                    f"hydraserve_pd_replay_mismatches_total {stats.replay_mismatches}",
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
        # BaseHTTPRequestHandler uses an unbuffered _SocketWriter by default;
        # write() already sends the event immediately and flush() is a no-op.
        self.wfile.write(
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
        )

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
