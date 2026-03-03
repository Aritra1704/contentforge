"""Application settings for the standalone LLM comparator."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_MODEL",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_qwen25_model: str = Field(
        default="qwen2.5:7b-instruct",
        validation_alias="OLLAMA_QWEN25_MODEL",
    )
    ollama_llama31_model: str = Field(
        default="llama3.1:8b",
        validation_alias="OLLAMA_LLAMA31_MODEL",
    )
    ollama_mistral_model: str = Field(
        default="mistral:7b",
        validation_alias="OLLAMA_MISTRAL_MODEL",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text:latest",
        validation_alias="OLLAMA_EMBED_MODEL",
    )
    db_url: str = Field(
        default="sqlite+aiosqlite:///./llm_comparator.db",
        validation_alias="DB_URL",
    )


settings = Settings()
