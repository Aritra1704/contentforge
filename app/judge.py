"""Optional LLM-based judge for compare-model quality ranking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Literal

from app.config import settings
from app.errors import AppError
from app.llm import call_backend
from app.quality import is_quality_valid
from app.schemas import (
    CompareModelResult,
    GenerateCompareModelsRequest,
    GenerateSingleRequest,
    JudgeResult,
    OutputSpec,
    WinnerSource,
)
from src.judge.openai_judge import (
    build_openai_judge_messages,
    judge_candidates as openai_judge_candidates,
    parse_openai_judge_result,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgeRunResult:
    """Judge run output with candidate mapping and parse status."""

    decision: JudgeResult | None
    candidate_map: dict[str, CompareModelResult]
    source: WinnerSource | None = None
    reason: str | None = None


def judge_candidate_id(index: int) -> str:
    """Return stable judge candidate IDs: modelA/modelB/modelC/modelD."""

    return f"model{chr(ord('A') + index)}"


def render_candidate_output(result: CompareModelResult) -> str:
    """Render one candidate output into judge-visible plain text."""

    if result.structured_output is not None:
        pros_lines = "\n".join(f"- {item}" for item in result.structured_output.pros)
        cons_lines = "\n".join(f"- {item}" for item in result.structured_output.cons)
        return f"Pros:\n{pros_lines}\nCons:\n{cons_lines}".strip()
    if result.raw_text:
        return result.raw_text
    return "\n".join(result.items).strip()


def select_judge_candidates(results: list[CompareModelResult]) -> list[CompareModelResult]:
    """Pick 2-4 successful compare candidates for LLM judging."""

    valid_results = [item for item in results if item.ok and item.quality is not None]
    if not valid_results:
        return []

    valid_results.sort(
        key=lambda item: (
            is_quality_valid(item.quality),
            item.quality.total if item.quality is not None else 0,
        ),
        reverse=True,
    )
    return valid_results[:4]


def build_judge_context(payload: GenerateCompareModelsRequest) -> dict[str, object]:
    """Build judge context from the original generation request."""

    spec = payload.output_spec or OutputSpec()
    request_context = payload.model_dump(mode="json")
    cultural_context = str(payload.cultural_context).strip() or "global"
    return {
        "theme_name": payload.theme_name,
        "keywords": payload.prompt_keywords,
        "audience": payload.audience,
        "cultural_context": cultural_context,
        "tone_settings": {
            "tone_funny_pct": payload.tone_funny_pct,
            "tone_emotion_pct": payload.tone_emotion_pct,
            "tone_style": payload.tone_style,
            "emoji_policy": payload.emoji_policy,
        },
        "request_context": request_context,
        "tone_funny_pct": payload.tone_funny_pct,
        "tone_emotion_pct": payload.tone_emotion_pct,
        "tone_style": payload.tone_style,
        "emoji_policy": payload.emoji_policy,
        "avoid_cliches": payload.avoid_cliches,
        "avoid_phrases": payload.avoid_phrases,
        "output_format": spec.format,
        "output_spec": spec.model_dump(mode="json"),
    }


def build_judge_messages(
    payload: GenerateCompareModelsRequest,
    candidate_map: dict[str, CompareModelResult],
) -> list[dict[str, str]]:
    """Build judge messages with strict JSON-only requirements."""

    context = build_judge_context(payload)
    candidates = {
        candidate_id: render_candidate_output(result)
        for candidate_id, result in candidate_map.items()
    }
    return build_openai_judge_messages(context, candidates)


def make_backend_judge_payload(
    shared: GenerateCompareModelsRequest,
    *,
    provider: Literal["ollama", "groq"],
) -> GenerateSingleRequest:
    """Create one internal request payload for backend judge LLM calls."""

    return GenerateSingleRequest(
        theme_name=shared.theme_name,
        tone_funny_pct=shared.tone_funny_pct,
        tone_emotion_pct=shared.tone_emotion_pct,
        prompt_keywords=shared.prompt_keywords,
        visual_style=shared.visual_style,
        backend=provider,
        model=settings.judge_model,
        count=1,
        max_tokens=900,
        temperature=0.0,
        max_words=5000,
        min_words=None,
        emoji_policy=shared.emoji_policy,
        tone_style=shared.tone_style,
        audience=shared.audience,
        cultural_context=shared.cultural_context,
        avoid_cliches=shared.avoid_cliches,
        avoid_phrases=shared.avoid_phrases,
        output_format="lines",
        output_spec=OutputSpec(format="paragraph"),
        trace_id=shared.trace_id,
        seed=shared.seed,
    )


async def run_backend_judge(
    payload: GenerateCompareModelsRequest,
    candidate_map: dict[str, CompareModelResult],
    *,
    provider: Literal["ollama", "groq"],
) -> JudgeRunResult:
    """Run judge with one non-OpenAI backend provider."""

    candidate_ids = list(candidate_map.keys())
    messages = build_judge_messages(payload, candidate_map)

    try:
        judge_payload = make_backend_judge_payload(payload, provider=provider)
    except Exception:
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason="Judge configuration is invalid; baseline winner used.",
        )

    logger.info(
        "judge_request provider=%s model=%s candidate_count=%s candidates=%s timeout_sec=%s connect_timeout_sec=%s",
        provider,
        settings.judge_model,
        len(candidate_ids),
        ",".join(candidate_ids),
        settings.judge_timeout_sec,
        settings.judge_connect_timeout_sec,
    )
    try:
        response_text = await call_backend(
            judge_payload,
            messages=messages,
            timeout_sec=settings.judge_timeout_sec,
            connect_timeout_sec=settings.judge_connect_timeout_sec,
        )
    except AppError as exc:
        logger.warning("judge_call_failed provider=%s error_type=%s message=%s", provider, exc.error_type, exc.message)
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason=f"Judge failed: {exc.message}",
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("judge_call_failed provider=%s error_type=internal message=%s", provider, str(exc))
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason="Judge failed: internal error.",
        )

    decision = parse_openai_judge_result(response_text, candidate_ids)
    if decision is None:
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason="Judge output could not be parsed as strict JSON.",
        )

    logger.info(
        "judge_response provider=%s winner_key=%s ranking=%s",
        provider,
        decision.winner_key,
        ",".join(decision.ranking),
    )
    return JudgeRunResult(
        decision=decision,
        candidate_map=candidate_map,
        source="judge",
        reason=None,
    )


async def run_openai_judge(
    payload: GenerateCompareModelsRequest,
    candidate_map: dict[str, CompareModelResult],
) -> JudgeRunResult:
    """Run judge with OpenAI provider."""

    if not settings.openai_api_key.strip():
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason="Judge skipped: OPENAI_API_KEY is required for JUDGE_PROVIDER=openai.",
        )

    context = build_judge_context(payload)
    candidates = {
        candidate_id: render_candidate_output(result)
        for candidate_id, result in candidate_map.items()
    }
    try:
        decision = await openai_judge_candidates(context, candidates)
    except Exception as exc:
        logger.warning("judge_call_failed provider=openai message=%s", str(exc))
        return JudgeRunResult(
            decision=None,
            candidate_map=candidate_map,
            source="judge",
            reason=f"Judge failed: {str(exc)}",
        )

    return JudgeRunResult(
        decision=decision,
        candidate_map=candidate_map,
        source="judge",
        reason=None,
    )


async def run_llm_judge(
    payload: GenerateCompareModelsRequest,
    results: list[CompareModelResult],
) -> JudgeRunResult:
    """Run optional LLM judge and return parsed decision plus candidate map."""

    candidates = select_judge_candidates(results)
    if len(candidates) < 2:
        return JudgeRunResult(
            decision=None,
            candidate_map={},
            reason="Judge skipped: requires at least 2 successful outputs.",
        )

    candidate_map = {judge_candidate_id(index): item for index, item in enumerate(candidates)}
    judge_provider = settings.judge_provider
    judge_model = settings.judge_model
    if settings.openai_judge_enabled:
        judge_provider = "openai"
        judge_model = settings.openai_judge_model
    elif settings.judge_provider == "openai":
        judge_provider = "ollama"
        judge_model = settings.judge_model

    started_at = perf_counter()
    logger.info(
        "compare_models_judge_started judge_provider=%s judge_model=%s candidate_count=%s",
        judge_provider,
        judge_model,
        len(candidate_map),
    )

    if settings.openai_judge_enabled:
        result = await run_openai_judge(payload, candidate_map)
    elif settings.judge_provider == "openai":
        logger.info("judge_provider_overridden requested=openai applied=ollama reason=openai_judge_disabled")
        result = await run_backend_judge(payload, candidate_map, provider="ollama")
    elif settings.judge_provider == "groq":
        result = await run_backend_judge(payload, candidate_map, provider="groq")
    else:
        result = await run_backend_judge(payload, candidate_map, provider="ollama")

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "compare_models_judge_completed judge_provider=%s judge_model=%s latency_ms=%s fallback_to_baseline=%s",
        judge_provider,
        judge_model,
        elapsed_ms,
        result.decision is None,
    )
    return result
