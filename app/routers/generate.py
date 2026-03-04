"""Stateless generation endpoints with in-memory busy handling."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from fastapi import APIRouter, Request

from app.busy import BusyError, BusyManager
from app.config import settings
from app.errors import AppError, BusyServiceError
from app.llm import generate_items
from app.observability import format_log_line, get_request_context, sanitize_details, update_request_context
from app.schemas import (
    CompareModelResult,
    CompareModelTarget,
    ErrorBody,
    GenerateCompareModelsRequest,
    GenerateCompareModelsResponse,
    GenerateSingleRequest,
    GenerateSingleResponse,
    ResponseMeta,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])
legacy_router = APIRouter(prefix="/generation", tags=["generate"])

busy_manager = BusyManager(
    max_concurrent_jobs=settings.max_concurrent_jobs,
    max_queue=settings.max_queue,
    retry_after_ms=settings.busy_retry_after_ms,
)


def error_body_from_exception(error: AppError) -> ErrorBody:
    """Convert an AppError into a structured error response fragment."""

    return ErrorBody(
        error_type=error.error_type,
        message=error.message,
        backend=error.backend,
        model=error.model,
        http_status=error.http_status,
        retry_after_ms=error.retry_after_ms,
        details=sanitize_details(error.details),
    )


def applied_settings(payload: GenerateSingleRequest | GenerateCompareModelsRequest) -> dict[str, object]:
    """Expose the most relevant generation settings in response metadata."""

    return {
        "max_words": payload.max_words,
        "emoji_policy": payload.emoji_policy,
        "tone_style": payload.tone_style,
        "avoid_cliches": payload.avoid_cliches,
    }


def build_single_request(
    shared: GenerateCompareModelsRequest,
    target: CompareModelTarget,
) -> GenerateSingleRequest:
    """Expand one compare target into the single-generation request model."""

    return GenerateSingleRequest(
        theme_name=shared.theme_name,
        tone_funny_pct=shared.tone_funny_pct,
        tone_emotion_pct=shared.tone_emotion_pct,
        prompt_keywords=shared.prompt_keywords,
        visual_style=shared.visual_style,
        backend=target.backend,
        model=target.model,
        count=shared.count,
        max_tokens=shared.max_tokens,
        temperature=shared.temperature,
        max_words=shared.max_words,
        emoji_policy=shared.emoji_policy,
        tone_style=shared.tone_style,
        audience=shared.audience,
        avoid_cliches=shared.avoid_cliches,
        avoid_phrases=shared.avoid_phrases,
        output_format=shared.output_format,
        trace_id=shared.trace_id,
        seed=shared.seed,
    )


async def execute_compare_target(
    request: Request,
    payload: GenerateSingleRequest,
) -> CompareModelResult:
    """Execute one compare-model target and retain structured errors in-band."""

    target_context = get_request_context(request)
    target_context.update({"backend": payload.backend, "model": payload.model})

    try:
        items = await generate_items(payload)
    except AppError as exc:
        logger.error(
            format_log_line(
                "compare_target_failed",
                **target_context,
                error_type=exc.error_type,
                message=exc.message,
            ),
            exc_info=exc,
        )
        return CompareModelResult(
            ok=False,
            backend=payload.backend,
            model=payload.model,
            error=error_body_from_exception(exc),
        )
    except Exception as exc:
        logger.error(
            format_log_line(
                "compare_target_failed",
                **target_context,
                error_type="internal_error",
                message="Internal server error.",
            ),
            exc_info=exc,
        )
        return CompareModelResult(
            ok=False,
            backend=payload.backend,
            model=payload.model,
            error=ErrorBody(
                error_type="internal_error",
                message="Internal server error.",
                backend=payload.backend,
                model=payload.model,
                http_status=500,
            ),
        )

    return CompareModelResult(
        ok=True,
        backend=payload.backend,
        model=payload.model,
        items=items,
    )


@router.post("/single", response_model=GenerateSingleResponse)
async def generate_single(request: Request, payload: GenerateSingleRequest) -> GenerateSingleResponse:
    """Generate content items with one backend/model pair."""

    started_at = perf_counter()
    update_request_context(
        request,
        trace_id=payload.trace_id,
        backend=payload.backend,
        model=payload.model,
    )

    try:
        async with busy_manager.slot():
            items = await generate_items(payload)
    except BusyError as exc:
        raise BusyServiceError(
            backend=payload.backend,
            model=payload.model,
            retry_after_ms=exc.retry_after_ms,
        ) from exc

    latency_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        format_log_line(
            "generation_succeeded",
            **get_request_context(request),
            count=payload.count,
            latency_ms=latency_ms,
        )
    )
    return GenerateSingleResponse(
        ok=True,
        backend=payload.backend,
        model=payload.model,
        items=items,
        meta=ResponseMeta(
            latency_ms=latency_ms,
            request_id=request.state.request_id,
            trace_id=payload.trace_id,
            busy=False,
            applied_settings=applied_settings(payload),
        ),
        errors=None,
    )


@router.post("/compare-models", response_model=GenerateCompareModelsResponse)
@legacy_router.post("/compare-models", response_model=GenerateCompareModelsResponse)
async def compare_models(
    request: Request,
    payload: GenerateCompareModelsRequest,
) -> GenerateCompareModelsResponse:
    """Run the same prompt against multiple backend/model targets."""

    started_at = perf_counter()
    update_request_context(
        request,
        trace_id=payload.trace_id,
        backend="multiple",
        model=f"{len(payload.targets)} targets",
    )
    try:
        async with busy_manager.slot():
            expanded_requests = [build_single_request(payload, target) for target in payload.targets]
            results = await asyncio.gather(
                *(execute_compare_target(request, item) for item in expanded_requests)
            )
    except BusyError as exc:
        raise BusyServiceError(
            backend="multiple",
            model=f"{len(payload.targets)} targets",
            retry_after_ms=exc.retry_after_ms,
        ) from exc

    latency_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        format_log_line(
            "compare_models_completed",
            **get_request_context(request),
            targets=len(payload.targets),
            success_count=sum(1 for item in results if item.ok),
            failure_count=sum(1 for item in results if not item.ok),
            latency_ms=latency_ms,
        )
    )
    return GenerateCompareModelsResponse(
        ok=all(item.ok for item in results),
        results=results,
        meta=ResponseMeta(
            latency_ms=latency_ms,
            request_id=request.state.request_id,
            trace_id=payload.trace_id,
            busy=False,
            applied_settings=applied_settings(payload),
        ),
    )
