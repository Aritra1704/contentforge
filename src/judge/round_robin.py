"""Pairwise round-robin judging for multi-candidate content comparison."""

from __future__ import annotations

import asyncio
from functools import cmp_to_key
import json
import logging
from math import comb
from time import perf_counter
from typing import Any

import httpx

from app.errors import AppError, NetworkError, NotConfiguredError, ProviderError
from app.schemas import (
    GenerateSingleRequest,
    OutputSpec,
    RoundRobinCandidateInput,
    RoundRobinDimensionScore,
    RoundRobinJudgeRequest,
    RoundRobinJudgeResponse,
    RoundRobinLeaderboardEntry,
    RoundRobinPairwiseResult,
    RoundRobinWinner,
)

logger = logging.getLogger(__name__)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MAX_ATTEMPTS = 3

try:  # pragma: no cover - optional dependency path
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency path
    AsyncOpenAI = None  # type: ignore[assignment]


class JudgeTimeoutError(Exception):
    """Raised when the round-robin judge exceeds the configured timeout."""

    def __init__(self, *, provider: str, model: str, timeout_seconds: float) -> None:
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Judge timed out after {timeout_seconds:g} seconds")


class JudgeProviderError(Exception):
    """Raised when the configured judge provider fails and fallback is disabled."""

    def __init__(
        self,
        *,
        error_type: str,
        message: str,
        provider: str,
        model: str,
        status_code: int,
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        super().__init__(message)


def _settings() -> Any:
    """Return current app settings, resilient to test-time module reloads."""

    from app.config import settings

    return settings


def _extract_json_fragment(content: str) -> Any:
    """Proxy JSON extraction through the current app.llm module."""

    from app.llm import extract_json_fragment

    return extract_json_fragment(content)


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort integer coercion with default fallback."""

    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _clamp(value: int, *, low: int, high: int) -> int:
    """Clamp a score value into inclusive bounds."""

    return max(low, min(high, value))


def _format_text_block(value: str) -> str:
    """Normalize candidate text blocks for prompt readability."""

    lines = [line.rstrip() for line in value.strip().splitlines()]
    return "\n".join(lines).strip()


def _openai_timeout() -> httpx.Timeout:
    """Return OpenAI timeout config from environment-backed settings."""

    settings = _settings()
    return httpx.Timeout(settings.judge_timeout_sec, connect=settings.judge_connect_timeout_sec)


def build_pairwise_judge_messages(
    payload: RoundRobinJudgeRequest,
    *,
    left_key: str,
    left: RoundRobinCandidateInput,
    right_key: str,
    right: RoundRobinCandidateInput,
) -> list[dict[str, str]]:
    """Build strict pairwise judge prompts for one candidate pair."""

    context_json = json.dumps(payload.prompt_context.model_dump(mode="json"), ensure_ascii=True)
    output_spec_json = json.dumps(payload.prompt_context.output_spec.model_dump(mode="json"), ensure_ascii=True)

    system_prompt = (
        "You are an expert editorial judge for creative content.\n"
        "Compare two candidate outputs for the same request.\n"
        "Rank by writing quality and usability.\n"
        "Ignore latency, backend identity, and model size.\n"
        "Prefer content that feels more human, complete, original, emotionally authentic, and publishable.\n"
        "Return strict JSON only with no markdown and no extra prose."
    )

    user_prompt = (
        "Evaluate candidates using content quality only.\n"
        "Do not reward speed.\n"
        "Do not reward shorter output unless brevity is explicitly required by output_spec.\n"
        "Penalize bland template-like text and abrupt/incomplete endings.\n"
        "Task fit must include tone, audience, format, and cultural_context alignment when relevant.\n"
        "\nPrompt context (full JSON):\n"
        f"{context_json}\n"
        "\nKey requirements:\n"
        f"- Theme: {payload.prompt_context.theme_name}\n"
        f"- Tone funny pct: {payload.prompt_context.tone_funny_pct}\n"
        f"- Tone emotion pct: {payload.prompt_context.tone_emotion_pct}\n"
        f"- Tone style: {payload.prompt_context.tone_style}\n"
        f"- Audience: {payload.prompt_context.audience}\n"
        f"- Cultural context: {payload.prompt_context.cultural_context}\n"
        f"- Output spec: {output_spec_json}\n"
        f"- Avoid cliches: {payload.prompt_context.avoid_cliches}\n"
        "\nCandidates:\n"
        f"{left_key} (model={left.model}, backend={left.backend}):\n{_format_text_block(left.text)}\n\n"
        f"{right_key} (model={right.model}, backend={right.backend}):\n{_format_text_block(right.text)}\n\n"
        "Return strict JSON with exact schema:\n"
        "{\n"
        f'  "winner_key": "{left_key}|{right_key}",\n'
        '  "reason": "short reason",\n'
        '  "scores": {\n'
        f'    "{left_key}": {{\n'
        '      "prompt_fit": 0,\n'
        '      "human_feel": 0,\n'
        '      "originality": 0,\n'
        '      "emotional_authenticity": 0,\n'
        '      "completeness": 0,\n'
        '      "publishability": 0,\n'
        '      "total_points": 0\n'
        "    },\n"
        f'    "{right_key}": {{\n'
        '      "prompt_fit": 0,\n'
        '      "human_feel": 0,\n'
        '      "originality": 0,\n'
        '      "emotional_authenticity": 0,\n'
        '      "completeness": 0,\n'
        '      "publishability": 0,\n'
        '      "total_points": 0\n'
        "    }\n"
        "  }\n"
        "}\n"
        "Score ranges: prompt_fit/human_feel/originality are 0-20; emotional_authenticity/completeness are 0-15; "
        "publishability is 0-10; total_points is 0-100.\n"
        "Use only the two provided candidate keys."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_pairwise_judge_result(
    content: str,
    *,
    left_key: str,
    right_key: str,
) -> RoundRobinPairwiseResult | None:
    """Parse one pairwise judge JSON output into normalized response payload."""

    fragment = _extract_json_fragment(content)
    if not isinstance(fragment, dict):
        return None

    raw_scores = fragment.get("scores")
    if not isinstance(raw_scores, dict):
        return None

    candidate_keys = [left_key, right_key]
    scores: dict[str, RoundRobinDimensionScore] = {}
    for candidate_key in candidate_keys:
        entry = raw_scores.get(candidate_key, {})
        if not isinstance(entry, dict):
            entry = {}

        prompt_fit = _clamp(_safe_int(entry.get("prompt_fit", 0)), low=0, high=20)
        human_feel = _clamp(_safe_int(entry.get("human_feel", 0)), low=0, high=20)
        originality = _clamp(_safe_int(entry.get("originality", 0)), low=0, high=20)
        emotional_authenticity = _clamp(_safe_int(entry.get("emotional_authenticity", 0)), low=0, high=15)
        completeness = _clamp(_safe_int(entry.get("completeness", 0)), low=0, high=15)
        publishability = _clamp(_safe_int(entry.get("publishability", 0)), low=0, high=10)

        computed_total = (
            prompt_fit
            + human_feel
            + originality
            + emotional_authenticity
            + completeness
            + publishability
        )
        total_points = _clamp(
            _safe_int(entry.get("total_points", computed_total), computed_total),
            low=0,
            high=100,
        )

        scores[candidate_key] = RoundRobinDimensionScore(
            prompt_fit=prompt_fit,
            human_feel=human_feel,
            originality=originality,
            emotional_authenticity=emotional_authenticity,
            completeness=completeness,
            publishability=publishability,
            total_points=total_points,
        )

    winner_key = str(fragment.get("winner_key", "")).strip()
    if winner_key not in candidate_keys:
        winner_key = max(candidate_keys, key=lambda key: scores[key].total_points)

    reason = str(fragment.get("reason", "")).strip()
    if not reason:
        reason = f"{winner_key} ranked higher on pairwise quality dimensions."

    return RoundRobinPairwiseResult(
        candidate_a_key=left_key,
        candidate_b_key=right_key,
        winner_key=winner_key,
        reason=reason,
        scores=scores,
    )


def _is_retryable_status(status_code: int) -> bool:
    """Return whether one HTTP status should trigger retry."""

    if status_code == 429:
        return True
    return status_code >= 500


def _backoff_seconds(attempt_index: int) -> float:
    """Return exponential backoff seconds for one attempt index."""

    return 0.5 * (2**attempt_index)


async def _call_openai_rest(messages: list[dict[str, str]]) -> str:
    """Call OpenAI Chat Completions API via HTTP REST."""

    settings = _settings()
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    request_body: dict[str, Any] = {
        "model": settings.judge_model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(OPENAI_MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=_openai_timeout()) as client:
                response = await client.post(
                    OPENAI_CHAT_COMPLETIONS_URL,
                    headers=headers,
                    json=request_body,
                )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            if attempt < OPENAI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            raise NetworkError(
                "OpenAI judge request failed.",
                backend="openai",
                model=settings.judge_model,
                response_status=504,
            ) from exc

        if response.status_code >= 400:
            if _is_retryable_status(response.status_code) and attempt < OPENAI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            raise ProviderError(
                "OpenAI judge returned an error response.",
                backend="openai",
                model=settings.judge_model,
                http_status=response.status_code,
                response_status=502,
            )

        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError(
                "OpenAI judge returned no choices.",
                backend="openai",
                model=settings.judge_model,
                response_status=502,
            )
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            joined = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            return joined.strip()
        return str(content).strip()

    raise ProviderError(
        "OpenAI judge request exhausted retries.",
        backend="openai",
        model=settings.judge_model,
        response_status=502,
    )


async def _call_openai_sdk(messages: list[dict[str, str]]) -> str:
    """Call OpenAI Chat Completions via official SDK when available."""

    assert AsyncOpenAI is not None
    settings = _settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.judge_timeout_sec)
    request_body: dict[str, Any] = {
        "model": settings.judge_model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(OPENAI_MAX_ATTEMPTS):
        try:
            response = await client.chat.completions.create(**request_body)
            message = response.choices[0].message if response.choices else None
            if message is None:
                raise ProviderError(
                    "OpenAI judge returned no choices.",
                    backend="openai",
                    model=settings.judge_model,
                    response_status=502,
                )
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                joined = "".join(str(getattr(item, "text", "")) for item in content)
                return joined.strip()
            return str(content).strip()
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - SDK classes vary by version
            if attempt < OPENAI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            raise ProviderError(
                "OpenAI judge request failed.",
                backend="openai",
                model=settings.judge_model,
                response_status=502,
            ) from exc

    raise ProviderError(
        "OpenAI judge request exhausted retries.",
        backend="openai",
        model=settings.judge_model,
        response_status=502,
    )


async def _call_openai_pairwise(messages: list[dict[str, str]]) -> str:
    """Dispatch OpenAI call using SDK when installed, else REST fallback."""

    if AsyncOpenAI is not None:
        return await _call_openai_sdk(messages)
    return await _call_openai_rest(messages)


def _make_backend_judge_payload(
    payload: RoundRobinJudgeRequest,
    *,
    provider: str,
) -> GenerateSingleRequest:
    """Build one internal payload for backend pairwise judging calls."""

    context = payload.prompt_context
    settings = _settings()
    backend = provider if provider in {"ollama", "groq"} else "ollama"
    return GenerateSingleRequest(
        theme_name=context.theme_name,
        tone_funny_pct=context.tone_funny_pct,
        tone_emotion_pct=context.tone_emotion_pct,
        prompt_keywords=[],
        visual_style="editorial",
        backend=backend,
        model=settings.judge_model,
        count=1,
        max_tokens=1000,
        temperature=0,
        max_words=5000,
        min_words=None,
        emoji_policy="none",
        tone_style=context.tone_style,
        audience=context.audience,
        cultural_context=context.cultural_context,
        avoid_cliches=context.avoid_cliches,
        avoid_phrases=[],
        output_format="lines",
        output_spec=OutputSpec(format="paragraph"),
        trace_id=None,
        seed=None,
    )


def _build_baseline_payload(
    payload: RoundRobinJudgeRequest,
    candidate: RoundRobinCandidateInput,
) -> GenerateSingleRequest:
    """Build one synthetic request payload for baseline local quality scoring."""

    context = payload.prompt_context
    count = context.output_spec.structure.items or 3
    max_words = context.output_spec.length.max_words or 5000
    backend = candidate.backend if candidate.backend in {"ollama", "groq"} else "ollama"
    return GenerateSingleRequest(
        theme_name=context.theme_name,
        tone_funny_pct=context.tone_funny_pct,
        tone_emotion_pct=context.tone_emotion_pct,
        prompt_keywords=[],
        visual_style="editorial",
        backend=backend,
        model=candidate.model,
        count=count,
        max_tokens=1000,
        temperature=0,
        max_words=max_words,
        min_words=context.output_spec.length.min_words,
        emoji_policy="none",
        tone_style=context.tone_style,
        audience=context.audience,
        cultural_context=context.cultural_context,
        avoid_cliches=context.avoid_cliches,
        avoid_phrases=[],
        output_format="lines",
        output_spec=context.output_spec.model_copy(deep=True),
        trace_id=None,
        seed=None,
    )


def _baseline_scores(
    payload: RoundRobinJudgeRequest,
    keyed_candidates: dict[str, RoundRobinCandidateInput],
) -> dict[str, int]:
    """Compute cheap local baseline scores for tie-break mode and timeout fallback."""

    from app.llm import parse_items
    from app.quality import score_quality

    scores: dict[str, int] = {}
    for key, candidate in keyed_candidates.items():
        baseline_payload = _build_baseline_payload(payload, candidate)
        parsed_output = parse_items(candidate.text, payload=baseline_payload)
        quality, _ = score_quality(baseline_payload, parsed_output)
        scores[key] = quality.total
    return scores


def _rank_by_baseline(
    baseline_scores: dict[str, int],
    *,
    index_by_key: dict[str, int],
) -> list[str]:
    """Sort candidate keys by local baseline score."""

    return sorted(
        baseline_scores.keys(),
        key=lambda key: (
            baseline_scores[key],
            -index_by_key[key],
        ),
        reverse=True,
    )


async def _judge_one_pair(
    payload: RoundRobinJudgeRequest,
    *,
    left_key: str,
    left: RoundRobinCandidateInput,
    right_key: str,
    right: RoundRobinCandidateInput,
) -> RoundRobinPairwiseResult:
    """Run one pairwise judge call and return parsed scoring output."""

    settings = _settings()
    provider = settings.judge_provider
    messages = build_pairwise_judge_messages(
        payload,
        left_key=left_key,
        left=left,
        right_key=right_key,
        right=right,
    )

    logger.info(
        "judge_round_robin_pair_request judge_provider=%s judge_model=%s pair=%s_vs_%s timeout_sec=%s connect_timeout_sec=%s",
        provider,
        settings.judge_model,
        left_key,
        right_key,
        settings.judge_timeout_sec,
        settings.judge_connect_timeout_sec,
    )

    if provider == "openai":
        if not settings.openai_api_key.strip():
            raise NotConfiguredError(
                "OpenAI judge backend is not configured (missing OPENAI_API_KEY).",
                backend="openai",
                model=settings.judge_model,
            )
        response_text = await _call_openai_pairwise(messages)
    else:
        judge_payload = _make_backend_judge_payload(payload, provider=provider)
        from app.llm import call_backend

        response_text = await call_backend(
            judge_payload,
            messages=messages,
            timeout_sec=settings.judge_timeout_sec,
            connect_timeout_sec=settings.judge_connect_timeout_sec,
        )

    parsed = parse_pairwise_judge_result(
        response_text,
        left_key=left_key,
        right_key=right_key,
    )
    if parsed is None:
        raise ProviderError(
            "Judge output could not be parsed as strict JSON.",
            backend=provider,
            model=settings.judge_model,
            response_status=502,
        )

    logger.info(
        "judge_round_robin_pair_response judge_provider=%s judge_model=%s winner=%s pair=%s_vs_%s",
        provider,
        settings.judge_model,
        parsed.winner_key,
        left_key,
        right_key,
    )
    return parsed


def _sort_candidate_keys(
    candidate_keys: list[str],
    *,
    wins: dict[str, int],
    points: dict[str, int],
    head_to_head: dict[tuple[str, str], str],
    index_by_key: dict[str, int],
) -> list[str]:
    """Sort candidate keys by wins, points, and head-to-head tie-break."""

    def _compare(left_key: str, right_key: str) -> int:
        if wins[left_key] != wins[right_key]:
            return -1 if wins[left_key] > wins[right_key] else 1

        if points[left_key] != points[right_key]:
            return -1 if points[left_key] > points[right_key] else 1

        pair = tuple(sorted((left_key, right_key)))
        head_to_head_winner = head_to_head.get(pair)
        if head_to_head_winner == left_key:
            return -1
        if head_to_head_winner == right_key:
            return 1

        if index_by_key[left_key] < index_by_key[right_key]:
            return -1
        if index_by_key[left_key] > index_by_key[right_key]:
            return 1
        return 0

    return sorted(candidate_keys, key=cmp_to_key(_compare))


def _build_response(
    *,
    keyed_candidates: dict[str, RoundRobinCandidateInput],
    ranked_keys: list[str],
    pairwise_results: list[RoundRobinPairwiseResult],
    wins: dict[str, int],
    losses: dict[str, int],
    points: dict[str, int],
    warning: str | None = None,
) -> RoundRobinJudgeResponse:
    """Build the final response model from ranking aggregates."""

    settings = _settings()
    leaderboard = [
        RoundRobinLeaderboardEntry(
            candidate_key=key,
            model=keyed_candidates[key].model,
            backend=keyed_candidates[key].backend,
            wins=wins[key],
            losses=losses[key],
            points=points[key],
        )
        for key in ranked_keys
    ]

    winner = None
    if leaderboard:
        top = leaderboard[0]
        winner = RoundRobinWinner(
            candidate_key=top.candidate_key,
            model=top.model,
            backend=top.backend,
            wins=top.wins,
            points=top.points,
        )

    return RoundRobinJudgeResponse(
        pairwise_results=pairwise_results,
        leaderboard=leaderboard,
        winner=winner,
        timeout_seconds_used=settings.judge_timeout_sec,
        judge_provider=settings.judge_provider,
        judge_model=settings.judge_model,
        warning=warning,
    )


def _fallback_to_baseline(
    *,
    keyed_candidates: dict[str, RoundRobinCandidateInput],
    candidate_keys: list[str],
    baseline_scores: dict[str, int],
    index_by_key: dict[str, int],
    wins: dict[str, int],
    losses: dict[str, int],
    points: dict[str, int],
    warning: str,
    started_at: float,
    pair_count: int,
    completed_pairs: int,
) -> RoundRobinJudgeResponse:
    """Return baseline-backed response after judge timeout/provider failure."""

    settings = _settings()
    for key in candidate_keys:
        points[key] = baseline_scores.get(key, 0)
    fallback_ranked = _rank_by_baseline(baseline_scores, index_by_key=index_by_key)
    response = _build_response(
        keyed_candidates=keyed_candidates,
        ranked_keys=fallback_ranked,
        pairwise_results=[],
        wins=wins,
        losses=losses,
        points=points,
        warning=warning,
    )
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "judge_round_robin_completed mode=%s judge_provider=%s judge_model=%s pair_count=%s completed_pairs=%s elapsed_ms=%s fallback=baseline warning=%s",
        settings.judge_mode,
        settings.judge_provider,
        settings.judge_model,
        pair_count,
        completed_pairs,
        elapsed_ms,
        warning,
    )
    return response


async def run_round_robin_judge(payload: RoundRobinJudgeRequest) -> RoundRobinJudgeResponse:
    """Execute pairwise round-robin judging and return ranking response."""

    settings = _settings()
    started_at = perf_counter()

    keyed_candidates = {
        f"candidate_{index + 1}": item
        for index, item in enumerate(payload.candidates)
    }
    candidate_keys = list(keyed_candidates.keys())
    index_by_key = {key: index for index, key in enumerate(candidate_keys)}

    baseline_scores = _baseline_scores(payload, keyed_candidates)
    baseline_ranked = _rank_by_baseline(baseline_scores, index_by_key=index_by_key)

    if settings.judge_mode == "tie_break" and len(candidate_keys) > 2:
        judged_keys = baseline_ranked[:2]
    else:
        judged_keys = list(candidate_keys)

    pair_count = comb(len(judged_keys), 2) if len(judged_keys) >= 2 else 0
    logger.info(
        "judge_round_robin_started mode=%s judge_provider=%s judge_model=%s candidate_count=%s judged_candidate_count=%s pair_count=%s timeout_sec=%s connect_timeout_sec=%s",
        settings.judge_mode,
        settings.judge_provider,
        settings.judge_model,
        len(candidate_keys),
        len(judged_keys),
        pair_count,
        settings.judge_timeout_sec,
        settings.judge_connect_timeout_sec,
    )

    if settings.judge_mode == "always" and settings.judge_provider == "ollama" and len(judged_keys) > 3:
        logger.warning(
            "judge_round_robin_slow_path provider=ollama candidate_count=%s pair_count=%s message=%s",
            len(judged_keys),
            pair_count,
            "Full round robin may be slow on local Ollama.",
        )

    wins = {key: 0 for key in candidate_keys}
    losses = {key: 0 for key in candidate_keys}
    points = {key: 0 for key in candidate_keys}
    head_to_head: dict[tuple[str, str], str] = {}
    pairwise_results: list[RoundRobinPairwiseResult] = []

    completed_pairs = 0
    for left_index, left_key in enumerate(judged_keys):
        left_candidate = keyed_candidates[left_key]
        for right_key in judged_keys[left_index + 1 :]:
            right_candidate = keyed_candidates[right_key]
            try:
                pair_result = await asyncio.wait_for(
                    _judge_one_pair(
                        payload,
                        left_key=left_key,
                        left=left_candidate,
                        right_key=right_key,
                        right=right_candidate,
                    ),
                    timeout=settings.judge_timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.warning(
                    "judge_round_robin_timeout mode=%s judge_provider=%s judge_model=%s timeout_sec=%s completed_pairs=%s total_pairs=%s elapsed_ms=%s",
                    settings.judge_mode,
                    settings.judge_provider,
                    settings.judge_model,
                    settings.judge_timeout_sec,
                    completed_pairs,
                    pair_count,
                    elapsed_ms,
                )
                if settings.judge_fallback_to_baseline:
                    return _fallback_to_baseline(
                        keyed_candidates=keyed_candidates,
                        candidate_keys=candidate_keys,
                        baseline_scores=baseline_scores,
                        index_by_key=index_by_key,
                        wins=wins,
                        losses=losses,
                        points=points,
                        warning="judge timed out, baseline used",
                        started_at=started_at,
                        pair_count=pair_count,
                        completed_pairs=completed_pairs,
                    )
                raise JudgeTimeoutError(
                    provider=settings.judge_provider,
                    model=settings.judge_model,
                    timeout_seconds=settings.judge_timeout_sec,
                ) from exc
            except AppError as exc:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.warning(
                    "judge_round_robin_provider_error mode=%s judge_provider=%s judge_model=%s error_type=%s completed_pairs=%s total_pairs=%s elapsed_ms=%s message=%s",
                    settings.judge_mode,
                    settings.judge_provider,
                    settings.judge_model,
                    exc.error_type,
                    completed_pairs,
                    pair_count,
                    elapsed_ms,
                    exc.message,
                )
                if settings.judge_fallback_to_baseline:
                    return _fallback_to_baseline(
                        keyed_candidates=keyed_candidates,
                        candidate_keys=candidate_keys,
                        baseline_scores=baseline_scores,
                        index_by_key=index_by_key,
                        wins=wins,
                        losses=losses,
                        points=points,
                        warning=f"judge failed ({exc.error_type}), baseline used",
                        started_at=started_at,
                        pair_count=pair_count,
                        completed_pairs=completed_pairs,
                    )
                raise JudgeProviderError(
                    error_type=exc.error_type,
                    message=exc.message,
                    provider=settings.judge_provider,
                    model=settings.judge_model,
                    status_code=exc.response_status,
                ) from exc
            except Exception as exc:
                elapsed_ms = int((perf_counter() - started_at) * 1000)
                logger.warning(
                    "judge_round_robin_provider_error mode=%s judge_provider=%s judge_model=%s error_type=internal_error completed_pairs=%s total_pairs=%s elapsed_ms=%s message=%s",
                    settings.judge_mode,
                    settings.judge_provider,
                    settings.judge_model,
                    completed_pairs,
                    pair_count,
                    elapsed_ms,
                    str(exc),
                )
                if settings.judge_fallback_to_baseline:
                    return _fallback_to_baseline(
                        keyed_candidates=keyed_candidates,
                        candidate_keys=candidate_keys,
                        baseline_scores=baseline_scores,
                        index_by_key=index_by_key,
                        wins=wins,
                        losses=losses,
                        points=points,
                        warning="judge failed (internal_error), baseline used",
                        started_at=started_at,
                        pair_count=pair_count,
                        completed_pairs=completed_pairs,
                    )
                raise JudgeProviderError(
                    error_type="internal_error",
                    message="Judge failed due to an internal error.",
                    provider=settings.judge_provider,
                    model=settings.judge_model,
                    status_code=502,
                ) from exc

            pairwise_results.append(pair_result)
            completed_pairs += 1

            left_score = pair_result.scores.get(left_key)
            right_score = pair_result.scores.get(right_key)
            if left_score is not None:
                points[left_key] += left_score.total_points
            if right_score is not None:
                points[right_key] += right_score.total_points

            if pair_result.winner_key == left_key:
                wins[left_key] += 1
                losses[right_key] += 1
            else:
                wins[right_key] += 1
                losses[left_key] += 1

            pair_key = tuple(sorted((left_key, right_key)))
            head_to_head[pair_key] = pair_result.winner_key

    if settings.judge_mode == "tie_break" and len(candidate_keys) > 2:
        judged_ranked = _sort_candidate_keys(
            judged_keys,
            wins=wins,
            points=points,
            head_to_head=head_to_head,
            index_by_key=index_by_key,
        )
        remaining_ranked = [key for key in baseline_ranked if key not in set(judged_keys)]
        for key in remaining_ranked:
            points[key] = baseline_scores.get(key, 0)
        ranked_keys = [*judged_ranked, *remaining_ranked]
    else:
        ranked_keys = _sort_candidate_keys(
            judged_keys,
            wins=wins,
            points=points,
            head_to_head=head_to_head,
            index_by_key=index_by_key,
        )

    response = _build_response(
        keyed_candidates=keyed_candidates,
        ranked_keys=ranked_keys,
        pairwise_results=pairwise_results,
        wins=wins,
        losses=losses,
        points=points,
        warning=None,
    )
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "judge_round_robin_completed mode=%s judge_provider=%s judge_model=%s pair_count=%s completed_pairs=%s elapsed_ms=%s fallback=none",
        settings.judge_mode,
        settings.judge_provider,
        settings.judge_model,
        pair_count,
        completed_pairs,
        elapsed_ms,
    )
    return response
