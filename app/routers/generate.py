"""Stateless generation endpoints with in-memory busy handling."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from fastapi import APIRouter, Request

from app.busy import BusyError, BusyManager
from app.config import settings
from app.errors import AppError, BusyServiceError
from app.judge import run_llm_judge
from app.llm import generate_items
from app.observability import format_log_line, get_request_context, sanitize_details, update_request_context
from app.quality import is_quality_valid, pick_quality_winner, score_quality
from app.quality_memory import (
    augment_avoid_phrases_with_memory,
    output_text_for_storage,
    store_quality_run,
)
from app.schemas import (
    CompareModelResult,
    CompareModelTarget,
    CompareModelsWinner,
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

    max_words = payload.max_words
    if payload.output_spec is not None and payload.output_spec.length.max_words is not None:
        max_words = payload.output_spec.length.max_words

    return {
        "max_words": max_words,
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
        min_words=shared.min_words,
        emoji_policy=shared.emoji_policy,
        tone_style=shared.tone_style,
        audience=shared.audience,
        avoid_cliches=shared.avoid_cliches,
        avoid_phrases=shared.avoid_phrases,
        output_format=shared.output_format,
        output_spec=shared.output_spec.model_copy(deep=True) if shared.output_spec is not None else None,
        trace_id=shared.trace_id,
        seed=shared.seed,
    )


def top_quality_gap(results: list[CompareModelResult]) -> int | None:
    """Return top-two quality-score gap among valid outputs."""

    valid = [item for item in results if item.ok and is_quality_valid(item.quality)]
    if len(valid) < 2:
        return None
    sorted_scores = sorted((item.quality.total for item in valid if item.quality is not None), reverse=True)
    if len(sorted_scores) < 2:
        return None
    return sorted_scores[0] - sorted_scores[1]


async def execute_compare_target(
    request: Request,
    payload: GenerateSingleRequest,
) -> CompareModelResult:
    """Execute one compare-model target and retain structured errors in-band."""

    started_at = perf_counter()
    target_context = get_request_context(request)
    target_context.update({"backend": payload.backend, "model": payload.model})

    try:
        output = await generate_items(payload)
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
            latency_ms=int((perf_counter() - started_at) * 1000),
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
            latency_ms=int((perf_counter() - started_at) * 1000),
            error=ErrorBody(
                error_type="internal_error",
                message="Internal server error.",
                backend=payload.backend,
                model=payload.model,
                http_status=500,
            ),
        )

    quality, _ = score_quality(payload, output)
    return CompareModelResult(
        ok=True,
        backend=payload.backend,
        model=payload.model,
        latency_ms=int((perf_counter() - started_at) * 1000),
        items=output.items,
        raw_text=output.raw_text,
        structured_output=output.structured_output,
        quality=quality,
    )


@router.post("/single", response_model=GenerateSingleResponse)
async def generate_single(request: Request, payload: GenerateSingleRequest) -> GenerateSingleResponse:
    """Generate content items with one backend/model pair."""

    await augment_avoid_phrases_with_memory(payload)
    started_at = perf_counter()
    update_request_context(
        request,
        trace_id=payload.trace_id,
        backend=payload.backend,
        model=payload.model,
    )

    try:
        async with busy_manager.slot():
            output = await generate_items(payload)
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
    quality, _ = score_quality(payload, output)
    await store_quality_run(
        payload=payload,
        backend=payload.backend,
        model=payload.model,
        output_text=output_text_for_storage(output),
        quality_score=quality,
        judge_json=None,
    )
    return GenerateSingleResponse(
        ok=True,
        backend=payload.backend,
        model=payload.model,
        items=output.items,
        raw_text=output.raw_text,
        structured_output=output.structured_output,
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

    await augment_avoid_phrases_with_memory(payload)
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

    baseline_winner = pick_quality_winner(results)
    final_winner = baseline_winner
    judge_result = None
    judge_json = None
    judge_reason: str | None = None
    winner_source = "baseline"

    if settings.judge_enabled:
        should_run_judge = False
        if settings.judge_mode == "always":
            should_run_judge = True
        else:
            gap = top_quality_gap(results)
            if gap is not None and gap <= settings.judge_tie_threshold:
                should_run_judge = True

        if should_run_judge:
            judge_run = await run_llm_judge(payload, results)
            judge_result = judge_run.decision
            if judge_run.decision is None:
                judge_reason = judge_run.reason
            else:
                judge_winner_result = judge_run.candidate_map.get(judge_run.decision.winner_key)
                if judge_winner_result is None:
                    judge_reason = "Judge winner could not be mapped to compare results; baseline winner used."
                else:
                    winner_total = judge_winner_result.quality.total if judge_winner_result.quality else 0
                    final_winner = CompareModelsWinner(
                        backend=judge_winner_result.backend,
                        model=judge_winner_result.model,
                        total_score=winner_total,
                    )
                    winner_source = judge_run.source or "baseline"
                    winner_entry = judge_run.decision.scores.get(judge_run.decision.winner_key)
                    winner_reason = winner_entry.reasons[0] if winner_entry and winner_entry.reasons else None
                    judge_reason = winner_reason or "Winner selected by LLM judge."
            judge_json = judge_result.model_dump(mode="json") if judge_result is not None else None
        elif settings.judge_mode == "tie_break":
            judge_reason = "Judge skipped: baseline lead exceeded tie threshold."

    judge_result_payload = judge_result.model_dump(mode="json") if judge_result is not None else None
    judge_json_payload = judge_result_payload
    for result in results:
        if not result.ok or result.quality is None:
            continue
        await store_quality_run(
            payload=payload,
            backend=result.backend,
            model=result.model,
            output_text=output_text_for_storage(result),
            quality_score=result.quality,
            judge_json=judge_json_payload,
        )

    return GenerateCompareModelsResponse(
        ok=all(item.ok for item in results),
        results=results,
        winner=final_winner,
        winner_source=winner_source,
        judge_result=judge_result_payload,
        judge_json=judge_json,
        judge_reason=judge_reason,
        meta=ResponseMeta(
            latency_ms=latency_ms,
            request_id=request.state.request_id,
            trace_id=payload.trace_id,
            busy=False,
            applied_settings=applied_settings(payload),
        ),
    )
