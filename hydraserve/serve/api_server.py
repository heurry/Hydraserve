"""
FastAPI-based OpenAI-compatible API server.

Provides:
  - POST /v1/chat/completions — Chat completions endpoint
  - GET /v1/models — List available models
  - GET /health — Health check with system status
  - POST /v1/completions — Legacy completions (basic support)

The server wraps the CentralScheduler and manages the request lifecycle
from HTTP request to streaming/non-streaming response.
"""

import time
import uuid
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from hydraserve.config import HydraServeConfig, ServingMode, RequestState
from hydraserve.serve.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatChoice, ChatMessage, TokenUsage,
    ModelListResponse, ModelInfo, HealthResponse, ErrorResponse,
)


class HydraServeAPI:
    """
    OpenAI-compatible API server for HydraServe.

    Wraps the inference engines behind a standard REST API,
    enabling drop-in replacement for OpenAI clients.
    """

    def __init__(
        self,
        config: HydraServeConfig,
        scheduler,  # CentralScheduler
    ):
        self.config = config
        self.scheduler = scheduler

        self.app = FastAPI(
            title="HydraServe API",
            description="Prefill-Decode Disaggregated Inference Engine for Hybrid Attention LLMs",
            version="0.1.0",
        )

        self._register_routes()

    def _register_routes(self) -> None:
        """Register all API routes."""

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            stats = self.scheduler.get_stats()
            return HealthResponse(
                status="ok",
                mode=self.config.mode.value,
                gpu_count=2,  # Hardcoded for dual 3090
                backend=self.config.transfer.backend,
                active_requests=stats["decode"]["total_requests"],
            )

        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            models = [
                ModelInfo(
                    id="Qwen3.5-4B",
                    created=1710000000,
                    owned_by="hydraserve",
                ),
                ModelInfo(
                    id="Qwen3.5-9B",
                    created=1710000000,
                    owned_by="hydraserve",
                ),
                ModelInfo(
                    id="Qwen3.6-27B",
                    created=1720000000,
                    owned_by="hydraserve",
                ),
            ]
            return ModelListResponse(data=models)

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """
            Chat completions endpoint (OpenAI-compatible).

            Handles both streaming and non-streaming responses.
            """
            # Build prompt from messages
            prompt = self._build_prompt(request.messages)

            # Tokenize (simplified — in production uses the model's tokenizer)
            input_ids = self._tokenize(prompt)

            # Extract sampling params
            sampling_params = {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_tokens": request.max_tokens,
            }

            if request.stop:
                sampling_params["stop"] = request.stop

            request_id = str(uuid.uuid4())

            if request.stream:
                return StreamingResponse(
                    self._stream_response(request_id, input_ids, sampling_params,
                                          request.max_tokens, request.model),
                    media_type="text/event-stream",
                )
            else:
                return await self._non_stream_response(
                    request_id, input_ids, sampling_params,
                    request.max_tokens, request.model
                )

        @self.app.get("/stats")
        async def get_stats():
            """Get detailed system statistics."""
            return self.scheduler.get_stats()

        @self.app.post("/v1/completions")
        async def completions(request: ChatCompletionRequest):
            """Legacy completions endpoint (redirects to chat)."""
            return await chat_completions(request)

    # ─── Internal ───────────────────────────────────────────────

    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """Build a formatted prompt from chat messages."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def _tokenize(self, prompt: str) -> List[int]:
        """Tokenize prompt. Placeholder — real impl uses model tokenizer."""
        # In production: tokenizer.encode(prompt)
        # For now, return placeholder token IDs
        return [1] * min(len(prompt) // 4, 100)  # Rough character-to-token estimate

    async def _non_stream_response(
        self,
        request_id: str,
        input_ids: List[int],
        sampling_params: Dict,
        max_tokens: int,
        model_name: str,
    ) -> ChatCompletionResponse:
        """Handle a non-streaming chat completion request."""
        req_id = hash(request_id) % (2**31)

        # Submit request to scheduler
        self.scheduler.submit_request(input_ids, sampling_params, max_tokens)

        # Poll for completion (in production: use asyncio events)
        generated_tokens = []
        finish_reason = None
        start_time = time.time()
        timeout = 120  # seconds

        while time.time() - start_time < timeout:
            output = self.scheduler.poll_output(req_id)
            if output is None:
                await asyncio.sleep(0.01)
                continue

            generated_tokens = output["tokens"]
            if output["is_finished"]:
                finish_reason = output["finish_reason"]
                break

            await asyncio.sleep(0.01)

        # Decode tokens
        text = self._decode(generated_tokens)

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id[:8]}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason or "length",
                )
            ],
            usage=TokenUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(generated_tokens),
                total_tokens=len(input_ids) + len(generated_tokens),
            ),
        )

    async def _stream_response(
        self,
        request_id: str,
        input_ids: List[int],
        sampling_params: Dict,
        max_tokens: int,
        model_name: str,
    ) -> AsyncGenerator[str, None]:
        """Handle a streaming chat completion request (SSE)."""
        req_id = hash(request_id) % (2**31)
        self.scheduler.submit_request(input_ids, sampling_params, max_tokens)

        created = int(time.time())
        last_token_count = 0
        start_time = time.time()
        timeout = 120

        while time.time() - start_time < timeout:
            output = self.scheduler.poll_output(req_id)
            if output is None:
                await asyncio.sleep(0.01)
                continue

            tokens = output["tokens"]
            new_tokens = tokens[last_token_count:]

            for token_id in new_tokens:
                text = self._decode([token_id])
                chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {__import__('json').dumps(chunk)}\n\n"

            last_token_count = len(tokens)

            if output["is_finished"]:
                final_chunk = {
                    "id": f"chatcmpl-{request_id[:8]}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": output["finish_reason"] or "length",
                    }],
                }
                yield f"data: {__import__('json').dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            await asyncio.sleep(0.01)

    def _decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text. Placeholder."""
        # In production: tokenizer.decode(token_ids)
        return f"[{','.join(str(t) for t in token_ids)}]"


def create_app(config: HydraServeConfig, scheduler) -> FastAPI:
    """Factory function to create the FastAPI app."""
    api = HydraServeAPI(config, scheduler)
    return api.app


def serve(config: HydraServeConfig, scheduler) -> None:
    """Start the API server."""
    app = create_app(config, scheduler)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.api_workers,
        log_level="info",
    )
