"""Stateless generation endpoints with in-memory busy handling."""

from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.busy import BusyError, BusyManager
from app.config import settings
from app.llm import UpstreamServiceError, generate_items
from app.schemas import GenerateSingleRequest, GenerateSingleResponse, ResponseMeta

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

busy_manager = BusyManager(
    max_concurrent_jobs=settings.max_concurrent_jobs,
    max_queue=settings.max_queue,
    retry_after_ms=settings.busy_retry_after_ms,
)


def busy_response(retry_after_ms: int) -> JSONResponse:
    """Return the standardized busy payload and headers."""

    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(settings.busy_retry_after_seconds)},
        content={
            "ok": False,
            "error": "busy",
            "retry_after_ms": retry_after_ms,
            "meta": {"busy": True},
        },
    )


@router.post("/single", response_model=GenerateSingleResponse)
async def generate_single(payload: GenerateSingleRequest) -> GenerateSingleResponse | JSONResponse:
    """Generate content items with one backend/model pair."""

    request_id = str(uuid4())
    started_at = perf_counter()

    try:
        async with busy_manager.slot():
            items = await generate_items(payload)
    except BusyError as exc:
        return busy_response(exc.retry_after_ms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UpstreamServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    latency_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "request_id=%s trace_id=%s backend=%s model=%s count=%s latency_ms=%s",
        request_id,
        payload.trace_id or "",
        payload.backend,
        payload.model,
        payload.count,
        latency_ms,
    )
    return GenerateSingleResponse(
        ok=True,
        backend=payload.backend,
        model=payload.model,
        items=items,
        meta=ResponseMeta(
            latency_ms=latency_ms,
            request_id=request_id,
            trace_id=payload.trace_id,
            busy=False,
        ),
    )
