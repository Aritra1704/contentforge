"""Quality memory inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.quality_memory import fetch_quality_history
from app.schemas import QualityHistoryResponse

router = APIRouter(prefix="/quality", tags=["quality"])


@router.get("/history", response_model=QualityHistoryResponse)
async def quality_history(
    limit: int = Query(default=50, ge=1, le=200),
    theme_name: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
) -> QualityHistoryResponse:
    """Return recent quality-memory rows with optional filters."""

    runs = await fetch_quality_history(limit=limit, theme_name=theme_name, keyword=keyword)
    return QualityHistoryResponse(ok=True, runs=runs)
