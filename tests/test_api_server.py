from __future__ import annotations

import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from hydraserve.api import create_server
from hydraserve.engine import ContinuousGenerationLoop, TokenSample
from hydraserve.engine import OverloadedError


class FakeTokenizer:
    eos_token_id = None

    def encode(self, text):
        return tuple(ord(character) for character in text)

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    def render_chat(self, messages):
        return "\n".join(message["content"] for message in messages)


class FakeBackend:
    def __init__(self):
        self.live = set()

    def prefill(self, request):
        self.live.add(request.request_id)
        return ord("A")

    def decode(self, requests):
        return tuple(request.generated_token_ids[-1] + 1 for request in requests)

    def release(self, request_id):
        self.live.remove(request_id)


def _post(base, path, payload):
    request = Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=3) as response:
        return response.headers, response.read()


def test_completion_chat_and_sse_protocol() -> None:
    loop = ContinuousGenerationLoop(FakeBackend())
    try:
        server = create_server(
            "127.0.0.1", 0, generation_loop=loop, tokenizer=FakeTokenizer(), model_name="tiny"
        )
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        _, body = _post(
            base,
            "/v1/completions",
            {"model": "tiny", "prompt": "x", "max_tokens": 3},
        )
        response = json.loads(body)
        assert response["choices"][0]["text"] == "ABC"
        assert response["usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 3,
            "total_tokens": 4,
        }

        _, body = _post(
            base,
            "/v1/chat/completions",
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        )
        response = json.loads(body)
        assert response["choices"][0]["message"]["content"] == "A"

        headers, body = _post(
            base,
            "/v1/completions",
            {
                "model": "tiny",
                "prompt": "x",
                "max_tokens": 2,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert headers.get_content_type() == "text/event-stream"
        lines = [line for line in body.decode().splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        content_chunks = [chunk for chunk in chunks if chunk["choices"]]
        assert "".join(chunk["choices"][0]["text"] for chunk in content_chunks) == "AB"
        assert content_chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["completion_tokens"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        loop.close()


def test_overload_returns_http_429() -> None:
    class FullLoop:
        def submit(self, token_ids, max_tokens, **kwargs):
            raise OverloadedError("admission queue request limit reached")

    try:
        server = create_server(
            "127.0.0.1",
            0,
            generation_loop=FullLoop(),
            tokenizer=FakeTokenizer(),
            model_name="tiny",
        )
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as caught:
            _post(
                base,
                "/v1/completions",
                {"model": "tiny", "prompt": "x", "max_tokens": 1},
            )
        assert caught.value.code == 429
        payload = json.loads(caught.value.read())
        assert payload["error"]["type"] == "overloaded_error"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_health_and_prometheus_metrics_expose_capacity() -> None:
    from hydraserve.engine import BackendCapacity, WorkerRecoveryStats

    class CapacityBackend(FakeBackend):
        def capacity(self):
            return BackendCapacity(10, 7, 4, 3)

        def recovery_stats(self):
            return WorkerRecoveryStats(2, 1, 3, 1, 2, (1,))

    loop = ContinuousGenerationLoop(CapacityBackend())
    try:
        server = create_server(
            "127.0.0.1",
            0,
            generation_loop=loop,
            tokenizer=FakeTokenizer(),
            model_name="tiny",
        )
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health", timeout=3) as response:
            health = json.loads(response.read())
        assert health["capacity"]["kv_free_blocks"] == 7
        assert health["status"] == "degraded"
        assert health["decode_workers"]["recovering"] == [1]
        with urlopen(base + "/metrics", timeout=3) as response:
            metrics = response.read().decode()
        assert 'hydraserve_kv_blocks{state="free"} 7' in metrics
        assert "hydraserve_admission_pending_requests 0" in metrics
        assert 'hydraserve_decode_workers{state="healthy"} 1' in metrics
        assert 'hydraserve_worker_restarts_total{outcome="success"} 1' in metrics
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        loop.close()


def test_sampling_logprobs_and_stop_strings_follow_openai_shapes() -> None:
    class SamplingBackend(FakeBackend):
        def prefill(self, request):
            self.live.add(request.request_id)
            return TokenSample(ord("A"), -0.1, ((ord("A"), -0.1), (ord("Z"), -2.0)))

        def decode(self, requests):
            return tuple(
                TokenSample(
                    request.generated_token_ids[-1] + 1,
                    -0.2,
                    ((request.generated_token_ids[-1] + 1, -0.2),),
                )
                for request in requests
            )

    loop = ContinuousGenerationLoop(SamplingBackend())
    try:
        server = create_server(
            "127.0.0.1",
            0,
            generation_loop=loop,
            tokenizer=FakeTokenizer(),
            model_name="tiny",
        )
    except PermissionError:
        pytest.skip("sandbox forbids loopback sockets")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        _, body = _post(
            base,
            "/v1/completions",
            {
                "model": "tiny",
                "prompt": "x",
                "max_tokens": 5,
                "temperature": 0,
                "stop": "BC",
                "logprobs": 2,
                "seed": 7,
            },
        )
        response = json.loads(body)
        choice = response["choices"][0]
        assert choice["text"] == "A"
        assert choice["finish_reason"] == "stop"
        assert choice["logprobs"]["tokens"] == ["A"]
        assert choice["logprobs"]["token_logprobs"] == [-0.1]
        assert response["usage"]["completion_tokens"] == 3

        _, body = _post(
            base,
            "/v1/chat/completions",
            {
                "model": "tiny",
                "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 2,
            },
        )
        chat_choice = json.loads(body)["choices"][0]
        content = chat_choice["logprobs"]["content"]
        assert content[0]["token"] == "A"
        assert content[0]["bytes"] == [65]
        assert len(content[0]["top_logprobs"]) == 2

        headers, body = _post(
            base,
            "/v1/completions",
            {
                "model": "tiny",
                "prompt": "x",
                "max_tokens": 5,
                "stream": True,
                "stop": ["BC"],
                "logprobs": 1,
            },
        )
        assert headers.get_content_type() == "text/event-stream"
        lines = [line for line in body.decode().splitlines() if line.startswith("data: ")]
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "A"
        assert chunks[0]["choices"][0]["logprobs"]["tokens"] == ["A"]
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        loop.close()
