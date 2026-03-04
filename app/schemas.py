"""Shared request and response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

BackendName = Literal["ollama", "groq"]


class GenerateSingleRequest(BaseModel):
    """Request payload for one stateless content generation call."""

    theme_name: str = Field(min_length=1)
    tone_funny_pct: int = Field(ge=0, le=100)
    tone_emotion_pct: int = Field(ge=0, le=100)
    prompt_keywords: list[str] = Field(default_factory=list)
    visual_style: str = Field(min_length=1)
    backend: BackendName
    model: str = Field(min_length=1)
    count: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=300, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0, le=2)
    trace_id: str | None = None
    seed: int | None = None

    @field_validator("prompt_keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        """Remove blank keywords before prompt construction."""

        return [item.strip() for item in value if item.strip()]

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Normalize the model name before backend validation."""

        return value.strip()

    @field_validator("trace_id")
    @classmethod
    def strip_trace_id(cls, value: str | None) -> str | None:
        """Normalize optional trace IDs."""

        if value is None:
            return None
        value = value.strip()
        return value or None


class ResponseMeta(BaseModel):
    """Metadata attached to successful generation responses."""

    latency_ms: int
    request_id: str
    trace_id: str | None = None
    busy: bool = False


class GenerateSingleResponse(BaseModel):
    """Successful response for POST /generate/single."""

    ok: Literal[True]
    backend: BackendName
    model: str
    items: list[str]
    meta: ResponseMeta


class OllamaModelCatalog(BaseModel):
    """Ollama-discovered chat and embedding model lists."""

    chat_models: list[str]
    embedding_models: list[str]


class ModelDiscoveryResponse(BaseModel):
    """Successful response for GET /models."""

    ok: Literal[True]
    ollama: OllamaModelCatalog


class HealthResponse(BaseModel):
    """Successful response for GET /health."""

    ok: Literal[True]
    service: str
    version: str
    busy: bool
    ollama_reachable: bool
