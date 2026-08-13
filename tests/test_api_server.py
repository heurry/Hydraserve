from __future__ import annotations

import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from hydraserve.api import create_server
from hydraserve.engine import ContinuousGenerationLoop
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
            {"model": "tiny", "prompt": "x", "max_tokens": 2, "stream": True},
        )
        assert headers.get_content_type() == "text/event-stream"
        lines = [line for line in body.decode().splitlines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(line[6:]) for line in lines[:-1]]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == "AB"
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        loop.close()


def test_overload_returns_http_429() -> None:
    class FullLoop:
        def submit(self, token_ids, max_tokens):
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
    from hydraserve.engine import BackendCapacity

    class CapacityBackend(FakeBackend):
        def capacity(self):
            return BackendCapacity(10, 7, 4, 3)

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
        with urlopen(base + "/metrics", timeout=3) as response:
            metrics = response.read().decode()
        assert 'hydraserve_kv_blocks{state="free"} 7' in metrics
        assert "hydraserve_admission_pending_requests 0" in metrics
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        loop.close()
