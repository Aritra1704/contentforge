"""Environment-backed configuration for the stateless generation service."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def split_csv(value: str) -> list[str]:
    """Return a normalized list from a comma-delimited environment value."""

    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "llm-comparator"
    service_version: str = Field(default="dev", validation_alias="SERVICE_VERSION")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", validation_alias="GROQ_MODEL")

    ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("OLLAMA_URL", "OLLAMA_BASE_URL"),
    )
    ollama_chat_models_raw: str = Field(
        default="mistral:7b,qwen2.5:7b-instruct,llama3.1:8b",
        validation_alias="OLLAMA_CHAT_MODELS",
    )
    ollama_embedding_models_raw: str = Field(
        default="nomic-embed-text:latest",
        validation_alias=AliasChoices("OLLAMA_EMBEDDING_MODELS", "OLLAMA_EMBED_MODEL"),
    )

    max_concurrent_jobs: int = Field(default=1, validation_alias="MAX_CONCURRENT_JOBS", ge=1)
    max_queue: int = Field(default=0, validation_alias="MAX_QUEUE", ge=0)
    busy_retry_after_ms: int = Field(default=2000, validation_alias="BUSY_RETRY_AFTER_MS", ge=1)
    request_timeout_sec: float = Field(default=120.0, validation_alias="REQUEST_TIMEOUT_SEC", gt=0)
    judge_enabled: bool = Field(default=False, validation_alias="JUDGE_ENABLED")
    judge_provider: Literal["openai", "ollama"] = Field(
        default="ollama",
        validation_alias=AliasChoices("JUDGE_PROVIDER", "JUDGE_BACKEND"),
    )
    judge_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="JUDGE_MODEL")
    openai_judge_enabled: bool = Field(default=False, validation_alias="OPENAI_JUDGE_ENABLED")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_judge_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_JUDGE_MODEL")
    judge_mode: Literal["tie_break", "always"] = Field(
        default="tie_break",
        validation_alias="JUDGE_MODE",
    )
    judge_tie_threshold: int = Field(default=7, validation_alias="JUDGE_TIE_THRESHOLD", ge=0)
    quality_memory_enabled: bool = Field(default=False, validation_alias="QUALITY_MEMORY_ENABLED")
    quality_memory_dsn: str = Field(
        default="",
        validation_alias=AliasChoices(
            "QUALITY_MEMORY_DSN",
            "POSTGRES_DSN",
            "DATABASE_URL",
            "DB_URL",
        ),
    )

    @property
    def ollama_chat_models(self) -> list[str]:
        """Configured Ollama chat-capable model names."""

        return split_csv(self.ollama_chat_models_raw)

    @property
    def ollama_embedding_models(self) -> list[str]:
        """Configured Ollama embedding-only model names."""

        return split_csv(self.ollama_embedding_models_raw)

    @property
    def ollama_embedding_prefixes(self) -> tuple[str, ...]:
        """Name prefixes treated as embedding-only models."""

        return tuple(name.split(":", 1)[0] for name in self.ollama_embedding_models)

    @property
    def busy_retry_after_seconds(self) -> int:
        """Retry-After header value derived from the millisecond backoff."""

        return max(1, math.ceil(self.busy_retry_after_ms / 1000))

    @property
    def healthcheck_timeout_sec(self) -> float:
        """Short timeout used for liveness checks against Ollama."""

        return min(5.0, self.request_timeout_sec)


settings = Settings()
