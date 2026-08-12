"""
OpenAI-compatible API protocol definitions.

Implements the request/response schemas compatible with
OpenAI's chat completions API for drop-in replacement.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class SamplingParams(BaseModel):
    """Sampling parameters for generation."""
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=100)
    max_tokens: int = Field(default=256, ge=1, le=131072)
    stop: Optional[List[str]] = Field(default=None)
    stop_token_ids: Optional[List[int]] = Field(default=None)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: Optional[int] = Field(default=None)


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="Qwen3.5-9B")
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=131072)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = Field(default=None)
    n: int = Field(default=1, ge=1, le=8)
    user: Optional[str] = Field(default=None)

    # HydraServe-specific extensions
    force_mode: Optional[str] = Field(
        default=None,
        description="Force routing mode: 'collocated' or 'pd_disaggregated'"
    )


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    """A single chat completion choice."""
    index: int = 0
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, str]] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class ModelInfo(BaseModel):
    """Model information response."""
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "hydraserve"


class ModelListResponse(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    mode: str = ""
    gpu_count: int = 0
    backend: str = ""
    active_requests: int = 0


class ErrorResponse(BaseModel):
    """Error response."""
    error: Dict[str, Any] = Field(default_factory=dict)
