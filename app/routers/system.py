"""Health and model discovery endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.llm import fetch_ollama_catalog, is_ollama_reachable
from app.routers.generate import busy_manager
from app.schemas import HealthResponse, ModelDiscoveryResponse, OllamaModelCatalog

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return local service health, version, and saturation state."""

    snapshot = await busy_manager.snapshot()
    return HealthResponse(
        ok=True,
        service=settings.service_name,
        version=settings.service_version,
        busy=snapshot.busy,
        ollama_reachable=await is_ollama_reachable(),
    )


@router.get("/models", response_model=ModelDiscoveryResponse)
async def models() -> ModelDiscoveryResponse:
    """List chat and embedding models exposed by Ollama."""

    chat_models, embedding_models = await fetch_ollama_catalog()
    return ModelDiscoveryResponse(
        ok=True,
        ollama=OllamaModelCatalog(
            chat_models=chat_models,
            embedding_models=embedding_models,
        ),
    )
