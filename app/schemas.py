"""Shared request and response models for the HTTP API."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_validator

BackendName = Literal["ollama", "groq"]
EmojiPolicy = Literal["none", "light", "expressive"]
ToneStyle = Literal["minimal", "poetic", "conversational", "witty", "inspirational"]
OutputFormat = Literal["lines", "numbered"]

DEFAULT_AVOID_PHRASES = [
    "new week",
    "rise and shine",
    "inner strength",
    "you got this",
    "make it happen",
    "shine bright",
    "positive vibes",
]


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
    max_words: int = Field(default=16, ge=1, le=64)
    emoji_policy: EmojiPolicy = "none"
    tone_style: ToneStyle = "conversational"
    audience: str = "general"
    avoid_cliches: bool = True
    avoid_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_AVOID_PHRASES))
    output_format: OutputFormat = "numbered"
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

    @field_validator("audience")
    @classmethod
    def strip_audience(cls, value: str) -> str:
        """Normalize audience text and enforce a non-empty value."""

        value = value.strip()
        return value or "general"

    @field_validator("avoid_phrases")
    @classmethod
    def normalize_avoid_phrases(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate avoid-phrase values."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized

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

    latency_ms: int | None = None
    request_id: str
    trace_id: str | None = None
    busy: bool = False
    applied_settings: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    """Structured error payload returned by the service."""

    error_type: str
    message: str
    backend: BackendName | str | None = None
    model: str | None = None
    http_status: int | None = None
    retry_after_ms: int | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response body returned by the service."""

    ok: Literal[False]
    error: ErrorBody
    meta: ResponseMeta


class GenerateSingleResponse(BaseModel):
    """Successful response for POST /generate/single."""

    ok: Literal[True]
    backend: BackendName
    model: str
    items: list[str]
    meta: ResponseMeta
    errors: ErrorBody | None = None


class CompareModelTarget(BaseModel):
    """One backend/model target in a compare request."""

    backend: BackendName
    model: str = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Normalize the target model name."""

        return value.strip()


class GenerateCompareModelsRequest(BaseModel):
    """Shared prompt plus multiple backend/model targets."""

    theme_name: str = Field(min_length=1)
    tone_funny_pct: int = Field(ge=0, le=100)
    tone_emotion_pct: int = Field(ge=0, le=100)
    prompt_keywords: list[str] = Field(default_factory=list)
    visual_style: str = Field(min_length=1)
    targets: list[CompareModelTarget] = Field(min_length=1)
    count: int = Field(default=3, ge=1, le=10)
    max_tokens: int = Field(default=300, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0, le=2)
    max_words: int = Field(default=16, ge=1, le=64)
    emoji_policy: EmojiPolicy = "none"
    tone_style: ToneStyle = "conversational"
    audience: str = "general"
    avoid_cliches: bool = True
    avoid_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_AVOID_PHRASES))
    output_format: OutputFormat = "numbered"
    trace_id: str | None = None
    seed: int | None = None

    @field_validator("prompt_keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        """Remove blank keywords before prompt construction."""

        return [item.strip() for item in value if item.strip()]

    @field_validator("trace_id")
    @classmethod
    def strip_trace_id(cls, value: str | None) -> str | None:
        """Normalize optional trace IDs."""

        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("audience")
    @classmethod
    def strip_audience(cls, value: str) -> str:
        """Normalize audience text and enforce a non-empty value."""

        value = value.strip()
        return value or "general"

    @field_validator("avoid_phrases")
    @classmethod
    def normalize_avoid_phrases(cls, value: list[str]) -> list[str]:
        """Normalize and deduplicate avoid-phrase values."""

        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = item.strip()
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(candidate)
        return normalized


class CompareModelResult(BaseModel):
    """One model result for compare-models responses."""

    ok: bool
    backend: BackendName
    model: str
    items: list[str] = Field(default_factory=list)
    error: ErrorBody | None = None


class GenerateCompareModelsResponse(BaseModel):
    """Response for compare-models requests."""

    ok: bool
    results: list[CompareModelResult]
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
