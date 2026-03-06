"""OpenAI-backed quality judge integration."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.llm import extract_json_fragment
from app.schemas import JudgeCandidateScore, JudgeResult

logger = logging.getLogger(__name__)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MAX_ATTEMPTS = 3  # initial call + 2 retries

try:  # pragma: no cover - optional dependency path
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency path
    AsyncOpenAI = None  # type: ignore[assignment]


def _clamp(value: int, *, low: int, high: int) -> int:
    """Clamp a score value into inclusive range."""

    return max(low, min(high, value))


def _normalize_string_list(value: object) -> list[str]:
    """Normalize list-like values into compact string lists."""

    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort integer coercion with default fallback."""

    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def build_openai_judge_messages(
    context: dict[str, Any],
    candidates: dict[str, str],
) -> list[dict[str, str]]:
    """Build strict judge messages for OpenAI Chat Completions."""

    candidate_blocks: list[str] = []
    for key, text in candidates.items():
        candidate_blocks.append(f"{key}:\n{text.strip()}")

    keywords = context.get("keywords") or []
    if isinstance(keywords, list):
        keyword_text = ", ".join(str(item).strip() for item in keywords if str(item).strip()) or "none"
    else:
        keyword_text = str(keywords).strip() or "none"

    avoid_phrases = context.get("avoid_phrases") or []
    if isinstance(avoid_phrases, list):
        avoid_phrase_text = ", ".join(str(item).strip() for item in avoid_phrases if str(item).strip()) or "none"
    else:
        avoid_phrase_text = str(avoid_phrases).strip() or "none"

    output_spec = context.get("output_spec") or {}
    output_spec_text = json.dumps(output_spec, ensure_ascii=True)
    output_format = context.get("output_format", "one_liner")
    tone_settings = context.get("tone_settings") or {
        "tone_funny_pct": context.get("tone_funny_pct", 0),
        "tone_emotion_pct": context.get("tone_emotion_pct", 0),
        "tone_style": context.get("tone_style", "conversational"),
        "emoji_policy": context.get("emoji_policy", "none"),
    }
    tone_settings_text = json.dumps(tone_settings, ensure_ascii=True)
    request_context = context.get("request_context") or {}
    request_context_text = json.dumps(request_context, ensure_ascii=True)
    cultural_context = str(context.get("cultural_context", context.get("culture", "global"))).strip() or "global"

    system_prompt = (
        "You are an expert editorial judge for creative content.\n"
        "Your job is to evaluate outputs for:\n"
        "- human feel\n"
        "- originality\n"
        "- emotional authenticity\n"
        "- completeness\n"
        "- tone match\n"
        "- audience fit\n"
        "- usefulness as publishable content\n"
        "\n"
        "Important:\n"
        "- Do NOT reward speed.\n"
        "- Do NOT reward shorter output unless brevity was requested.\n"
        "- Penalize bland, generic, template-like writing.\n"
        "- Penalize incomplete endings or abrupt paragraphs.\n"
        "- Penalize outputs that feel like default AI greeting text.\n"
        "\n"
        "Output strict JSON only, with no prose before or after JSON."
    )

    user_prompt = (
        "Evaluate candidates for quality against prompt requirements.\n"
        "Rank by writing quality and task fit, not speed or superficial cleanliness.\n"
        "Task fit must include alignment with requested cultural_context when relevant.\n"
        "Which output is more human and less bland?\n"
        "Which output best matches requested format and tone?\n"
        "Which output feels complete and usable?\n"
        "If one output is structurally clean but emotionally bland, and another is slightly imperfect "
        "but much stronger creatively, prefer the stronger creative output if it remains usable.\n"
        "Do NOT rank by latency/speed.\n"
        "Do NOT use word count as a proxy for quality unless the requested format requires length constraints.\n"
        "If avoid_cliches=true, explicitly check compliance with the avoid phrase list.\n"
        "\nPrompt requirements:\n"
        f"- Full request context JSON: {request_context_text}\n"
        f"- Theme: {context.get('theme_name', '')}\n"
        f"- Keywords: {keyword_text}\n"
        f"- Audience: {context.get('audience', 'general')}\n"
        f"- Cultural context: {cultural_context}\n"
        f"- Tone settings: {tone_settings_text}\n"
        f"- Tone funny pct: {context.get('tone_funny_pct', 0)}\n"
        f"- Tone emotion pct: {context.get('tone_emotion_pct', 0)}\n"
        f"- Tone style: {context.get('tone_style', 'conversational')}\n"
        f"- Emoji policy: {context.get('emoji_policy', 'none')}\n"
        f"- Avoid cliches: {bool(context.get('avoid_cliches', True))}\n"
        f"- Avoid phrases: {avoid_phrase_text}\n"
        f"- Output format: {output_format}\n"
        f"- OutputSpec: {output_spec_text}\n"
        "\nCandidates:\n"
        f"{chr(10).join(candidate_blocks)}\n"
        "\nReturn strict JSON with exact shape:\n"
        "{\n"
        '  "winner_key": "<candidate_key>",\n'
        '  "ranking": ["<candidate_key>", "<candidate_key>"],\n'
        '  "scores": {\n'
        '    "<candidate_key>": {\n'
        '      "task_fit": 0,\n'
        '      "originality": 0,\n'
        '      "emotional_authenticity": 0,\n'
        '      "completeness": 0,\n'
        '      "clarity_and_flow": 0,\n'
        '      "policy_cleanliness": 0,\n'
        '      "total": 0,\n'
        '      "reason": "...",\n'
        '      "issues": ["..."]\n'
        "    }\n"
        "  }\n"
        "}\n"
        "Use only provided candidate keys."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_openai_judge_result(content: str, candidate_keys: list[str]) -> JudgeResult | None:
    """Parse strict JSON judge output into JudgeResult."""

    fragment = extract_json_fragment(content)
    if not isinstance(fragment, dict):
        return None

    ranking_value = fragment.get("ranking")
    ranking: list[str] = []
    seen: set[str] = set()
    if isinstance(ranking_value, list):
        for item in ranking_value:
            key = str(item).strip()
            if key in candidate_keys and key not in seen:
                seen.add(key)
                ranking.append(key)
    for candidate_key in candidate_keys:
        if candidate_key not in seen:
            ranking.append(candidate_key)
    if not ranking:
        return None

    winner_key = str(fragment.get("winner_key", "")).strip()
    if winner_key not in candidate_keys:
        winner_key = ranking[0]

    raw_scores = fragment.get("scores")
    if not isinstance(raw_scores, dict):
        return None

    scores: dict[str, JudgeCandidateScore] = {}
    for candidate_key in candidate_keys:
        entry = raw_scores.get(candidate_key, {})
        if not isinstance(entry, dict):
            entry = {}

        # New schema fields.
        task_fit = _clamp(_safe_int(entry.get("task_fit", 0)), low=0, high=25)
        originality = _clamp(_safe_int(entry.get("originality", 0)), low=0, high=20)
        emotional_authenticity = _clamp(_safe_int(entry.get("emotional_authenticity", 0)), low=0, high=20)
        completeness = _clamp(_safe_int(entry.get("completeness", 0)), low=0, high=15)
        clarity_and_flow = _clamp(_safe_int(entry.get("clarity_and_flow", 0)), low=0, high=10)
        policy_cleanliness = _clamp(_safe_int(entry.get("policy_cleanliness", 0)), low=0, high=10)

        # Backward-compatible parsing of legacy fields.
        format_compliance = _clamp(_safe_int(entry.get("format_compliance", 0)), low=0, high=30)
        tone_alignment = _clamp(_safe_int(entry.get("tone_alignment", 0)), low=0, high=20)
        clarity_coherence = _clamp(_safe_int(entry.get("clarity_coherence", 0)), low=0, high=20)
        if task_fit == 0 and (format_compliance > 0 or tone_alignment > 0):
            task_fit = _clamp(int(round((format_compliance + tone_alignment) / 2)), low=0, high=25)
        if emotional_authenticity == 0 and tone_alignment > 0:
            emotional_authenticity = tone_alignment
        if clarity_and_flow == 0 and clarity_coherence > 0:
            clarity_and_flow = _clamp(int(round(clarity_coherence / 2)), low=0, high=10)

        reason = str(entry.get("reason", "")).strip()
        issues = _normalize_string_list(entry.get("issues"))
        reasons = _normalize_string_list(entry.get("reasons"))
        violations = _normalize_string_list(entry.get("violations"))
        if not reason and reasons:
            reason = reasons[0]
        if not issues and violations:
            issues = list(violations)

        computed_total = (
            task_fit
            + originality
            + emotional_authenticity
            + completeness
            + clarity_and_flow
            + policy_cleanliness
        )
        total = _clamp(_safe_int(entry.get("total", computed_total), computed_total), low=0, high=100)

        scores[candidate_key] = JudgeCandidateScore(
            task_fit=task_fit,
            originality=originality,
            emotional_authenticity=emotional_authenticity,
            completeness=completeness,
            clarity_and_flow=clarity_and_flow,
            policy_cleanliness=policy_cleanliness,
            total=total,
            reason=reason,
            issues=issues,
            format_compliance=format_compliance,
            tone_alignment=tone_alignment,
            clarity_coherence=clarity_coherence,
            reasons=reasons,
            violations=violations,
        )

    return JudgeResult(winner_key=winner_key, ranking=ranking, scores=scores)


def _is_retryable_status(status_code: int) -> bool:
    """Return whether one HTTP status should trigger retry."""

    if status_code == 429:
        return True
    return status_code >= 500


def _backoff_seconds(attempt_index: int) -> float:
    """Return exponential backoff seconds for one attempt index."""

    return 0.5 * (2**attempt_index)


def _openai_timeout() -> httpx.Timeout:
    """Return OpenAI judge timeout config from environment-backed settings."""

    return httpx.Timeout(settings.judge_timeout_sec, connect=settings.judge_connect_timeout_sec)


async def _call_openai_rest(messages: list[dict[str, str]]) -> str:
    """Call OpenAI Chat Completions API via HTTP REST."""

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    request_body: dict[str, Any] = {
        "model": settings.openai_judge_model,
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
            raise RuntimeError(f"OpenAI judge request failed: {exc.__class__.__name__}") from exc

        if response.status_code >= 400:
            if _is_retryable_status(response.status_code) and attempt < OPENAI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            raise RuntimeError(f"OpenAI judge HTTP {response.status_code}")

        payload = response.json()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI judge returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            joined = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            return joined.strip()
        return str(content).strip()

    raise RuntimeError("OpenAI judge request exhausted retries.")


async def _call_openai_sdk(messages: list[dict[str, str]]) -> str:
    """Call OpenAI Chat Completions via official SDK when available."""

    assert AsyncOpenAI is not None
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.judge_timeout_sec)
    request_body: dict[str, Any] = {
        "model": settings.openai_judge_model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(OPENAI_MAX_ATTEMPTS):
        try:
            response = await client.chat.completions.create(**request_body)
            message = response.choices[0].message if response.choices else None
            if message is None:
                raise RuntimeError("OpenAI judge returned no choices.")
            content = message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                joined = "".join(str(getattr(item, "text", "")) for item in content)
                return joined.strip()
            return str(content).strip()
        except Exception as exc:  # pragma: no cover - SDK classes vary by version
            if attempt < OPENAI_MAX_ATTEMPTS - 1:
                await asyncio.sleep(_backoff_seconds(attempt))
                continue
            raise RuntimeError(f"OpenAI judge SDK request failed: {exc.__class__.__name__}") from exc

    raise RuntimeError("OpenAI judge request exhausted retries.")


async def judge_candidates(context: dict[str, Any], candidates: dict[str, str]) -> JudgeResult:
    """Run OpenAI judge on 2-4 candidates and return normalized JudgeResult."""

    if not settings.openai_api_key.strip():
        raise ValueError("OPENAI_API_KEY is required when JUDGE_PROVIDER=openai.")

    candidate_keys = list(candidates.keys())
    if len(candidate_keys) < 2:
        raise ValueError("OpenAI judge requires at least two candidates.")
    if len(candidate_keys) > 4:
        raise ValueError("OpenAI judge supports at most four candidates.")

    messages = build_openai_judge_messages(context, candidates)
    logger.info(
        "judge_openai_request model=%s candidate_count=%s candidates=%s timeout_sec=%s connect_timeout_sec=%s",
        settings.openai_judge_model,
        len(candidate_keys),
        ",".join(candidate_keys),
        settings.judge_timeout_sec,
        settings.judge_connect_timeout_sec,
    )

    if AsyncOpenAI is not None:
        response_content = await _call_openai_sdk(messages)
    else:
        response_content = await _call_openai_rest(messages)

    parsed = parse_openai_judge_result(response_content, candidate_keys)
    if parsed is None:
        raise RuntimeError("OpenAI judge returned invalid JSON schema.")

    logger.info(
        "judge_openai_response winner_key=%s ranking=%s",
        parsed.winner_key,
        ",".join(parsed.ranking),
    )
    return parsed
