"""Endpoints for running phrase-generation comparisons across multiple LLM backends."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from json import JSONDecodeError
import json
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import GenerationResult

router = APIRouter(prefix="/generate", tags=["generate"])

BackendName = Literal["groq", "ollama_qwen25", "ollama_llama31", "ollama_mistral"]

DEFAULT_BACKENDS: list[BackendName] = [
    "groq",
    "ollama_qwen25",
    "ollama_llama31",
    "ollama_mistral",
]

BACKEND_MODELS: dict[BackendName, str] = {
    "groq": "llama-3.3-70b-versatile",
    "ollama_qwen25": "qwen2.5:7b-instruct",
    "ollama_llama31": "llama3.1:8b",
    "ollama_mistral": "mistral:7b",
}

SYSTEM_PROMPT = (
    "You are a professional greeting card copywriter specializing in emotionally "
    "resonant content for the Indian market. Write phrases that feel personal, "
    "warm, and shareable on WhatsApp and Instagram."
)


class CompareRequest(BaseModel):
    """Request payload for running one prompt across multiple backends."""

    theme_name: str
    tone_funny_pct: int
    tone_emotion_pct: int
    prompt_keywords: list[str]
    visual_style: str
    event_name: str | None = None
    count: int = Field(default=5, ge=1, le=10)
    backends: list[BackendName] = Field(default_factory=lambda: list(DEFAULT_BACKENDS))


class SingleRequest(BaseModel):
    """Request payload for running one prompt against a single backend."""

    theme_name: str
    tone_funny_pct: int
    tone_emotion_pct: int
    prompt_keywords: list[str]
    visual_style: str
    backend: BackendName
    count: int = Field(default=5, ge=1, le=10)
    event_name: str | None = None


class PhraseResult(BaseModel):
    """Normalized backend result returned to API callers."""

    backend: str
    model_name: str
    success: bool
    generation_time_ms: int
    phrases: list[dict[str, Any]]
    best_phrase: str
    error: str | None = None


class CompareResponse(BaseModel):
    """Comparison summary returned after running multiple backends."""

    run_id: str
    theme_name: str
    results: list[PhraseResult]
    fastest_backend: str | None
    all_succeeded: bool


def build_user_prompt(payload: CompareRequest | SingleRequest) -> str:
    """Construct one normalized user prompt for all backends."""

    if payload.tone_funny_pct >= 60:
        tone_instruction = "Emphasize humor, wit, and playful charm."
    elif payload.tone_emotion_pct >= 60:
        tone_instruction = "Emphasize emotional depth, warmth, and heartfelt sincerity."
    else:
        tone_instruction = "Use a balanced mix of humor and emotional warmth."

    event_fragment = f" Occasion or event: {payload.event_name}." if payload.event_name else ""
    keyword_fragment = ", ".join(payload.prompt_keywords) if payload.prompt_keywords else "none"

    return (
        f"Theme: {payload.theme_name}.{event_fragment}"
        f" Style direction: {payload.visual_style}. Keywords: {keyword_fragment}. "
        f"{tone_instruction} Generate exactly {payload.count} greeting card phrases. "
        "Each phrase must be between 8 and 20 words. "
        'Return valid JSON only in this format: {"phrases": [{"text": "...", "tone": "...", '
        '"word_count": 12}]}.'
    )


def expected_tone(tone_funny_pct: int, tone_emotion_pct: int) -> str:
    """Map tone percentages into the expected tone bucket."""

    if tone_funny_pct >= 60:
        return "funny"
    if tone_emotion_pct >= 60:
        return "emotional"
    return "balanced"


def score_phrase(phrase: dict[str, Any], target_tone: str) -> int:
    """Score one candidate phrase for best-phrase selection."""

    text = str(phrase.get("text", "")).strip()
    word_count = int(phrase.get("word_count") or len(text.split()))
    tone = str(phrase.get("tone", "")).strip().lower()
    score = 0

    if 8 <= word_count <= 20:
        score += 10
    if tone == target_tone:
        score += 20
    if "?" in text:
        score += 5
    if "!" in text:
        score += 3
    if word_count < 6:
        score -= 15
    if word_count > 25:
        score -= 10

    return score


def pick_best_phrase(phrases: Sequence[dict[str, Any]], target_tone: str) -> str:
    """Select the highest-scoring phrase text from the generated set."""

    if not phrases:
        return ""

    best_candidate = max(phrases, key=lambda phrase: score_phrase(phrase, target_tone))
    return str(best_candidate.get("text", "")).strip()


def parse_phrase_payload(content: str, target_tone: str) -> list[dict[str, Any]]:
    """Parse model output into normalized phrase dictionaries with a text fallback."""

    payload: Any
    try:
        payload = json.loads(content)
    except JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(content[start : end + 1])
            except JSONDecodeError:
                payload = None
        else:
            payload = None

    phrases: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_phrases = payload.get("phrases", [])
        if isinstance(raw_phrases, list):
            for item in raw_phrases:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                phrases.append(
                    {
                        "text": text,
                        "tone": str(item.get("tone") or target_tone),
                        "word_count": int(item.get("word_count") or len(text.split())),
                    }
                )

    if phrases:
        return phrases

    fallback_lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip().lstrip("-*0123456789. ").strip()
        if line:
            fallback_lines.append(line)

    if not fallback_lines and content.strip():
        fallback_lines.append(content.strip())

    return [
        {
            "text": line,
            "tone": target_tone,
            "word_count": len(line.split()),
        }
        for line in fallback_lines
    ]


def extract_content_and_tokens(backend: BackendName, payload: dict[str, Any]) -> tuple[str, int | None]:
    """Extract the text completion and token count from a backend-specific response."""

    if backend == "groq":
        message = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        token_count = usage.get("total_tokens")
        return str(message), int(token_count) if token_count is not None else None

    message = payload["message"]["content"]
    prompt_count = payload.get("prompt_eval_count")
    eval_count = payload.get("eval_count")
    token_count: int | None = None
    if isinstance(prompt_count, int) and isinstance(eval_count, int):
        token_count = prompt_count + eval_count
    return str(message), token_count


async def call_backend(
    backend: BackendName,
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[dict[str, Any], str]:
    """Call one configured backend and return the raw response plus model name."""

    model_name = BACKEND_MODELS[backend]
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if backend == "groq":
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
        else:
            response = await client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                },
            )

    response.raise_for_status()
    return response.json(), model_name


async def execute_generation(
    backend: BackendName,
    *,
    theme_name: str,
    tone_funny_pct: int,
    tone_emotion_pct: int,
    prompt_keywords: list[str],
    count: int,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Execute one backend call and normalize the result for storage and response."""

    started_at = perf_counter()
    target_tone = expected_tone(tone_funny_pct, tone_emotion_pct)
    model_name = BACKEND_MODELS[backend]

    try:
        raw_payload, model_name = await call_backend(
            backend,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        content, token_count = extract_content_and_tokens(backend, raw_payload)
        phrases = parse_phrase_payload(content, target_tone)
        phrases = phrases[:count]
        best_phrase = pick_best_phrase(phrases, target_tone)
        generation_time_ms = int((perf_counter() - started_at) * 1000)
        return {
            "backend": backend,
            "model_name": model_name,
            "prompt_type": "phrase_generation",
            "input_theme": theme_name,
            "input_tone_funny_pct": tone_funny_pct,
            "input_tone_emotion_pct": tone_emotion_pct,
            "input_keywords": json.dumps(prompt_keywords),
            "raw_output": json.dumps(raw_payload),
            "parsed_phrases": json.dumps(phrases),
            "best_phrase": best_phrase,
            "generation_time_ms": generation_time_ms,
            "token_count": token_count,
            "success": True,
            "error_message": None,
            "phrases": phrases,
            "error": None,
        }
    except Exception as exc:
        generation_time_ms = int((perf_counter() - started_at) * 1000)
        return {
            "backend": backend,
            "model_name": model_name,
            "prompt_type": "phrase_generation",
            "input_theme": theme_name,
            "input_tone_funny_pct": tone_funny_pct,
            "input_tone_emotion_pct": tone_emotion_pct,
            "input_keywords": json.dumps(prompt_keywords),
            "raw_output": "",
            "parsed_phrases": json.dumps([]),
            "best_phrase": "",
            "generation_time_ms": generation_time_ms,
            "token_count": None,
            "success": False,
            "error_message": str(exc),
            "phrases": [],
            "error": str(exc),
        }


async def persist_results(
    db: AsyncSession,
    *,
    run_id: str,
    results: Sequence[dict[str, Any]],
) -> None:
    """Persist a batch of generation results for one comparison run."""

    for item in results:
        db.add(
            GenerationResult(
                run_id=run_id,
                backend=item["backend"],
                model_name=item["model_name"],
                prompt_type=item["prompt_type"],
                input_theme=item["input_theme"],
                input_tone_funny_pct=item["input_tone_funny_pct"],
                input_tone_emotion_pct=item["input_tone_emotion_pct"],
                input_keywords=item["input_keywords"],
                raw_output=item["raw_output"],
                parsed_phrases=item["parsed_phrases"],
                best_phrase=item["best_phrase"],
                generation_time_ms=item["generation_time_ms"],
                token_count=item["token_count"],
                success=item["success"],
                error_message=item["error_message"],
            )
        )

    await db.commit()


def build_compare_response(
    run_id: str,
    theme_name: str,
    results: Sequence[dict[str, Any]],
) -> CompareResponse:
    """Convert normalized backend results into the public response schema."""

    successful_results = [item for item in results if item["success"]]
    fastest_backend = None
    if successful_results:
        fastest_backend = min(
            successful_results,
            key=lambda item: item["generation_time_ms"],
        )["backend"]

    return CompareResponse(
        run_id=run_id,
        theme_name=theme_name,
        results=[
            PhraseResult(
                backend=item["backend"],
                model_name=item["model_name"],
                success=item["success"],
                generation_time_ms=item["generation_time_ms"],
                phrases=item["phrases"],
                best_phrase=item["best_phrase"],
                error=item["error"],
            )
            for item in results
        ],
        fastest_backend=fastest_backend,
        all_succeeded=all(item["success"] for item in results),
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_generations(
    payload: CompareRequest,
    db: AsyncSession = Depends(get_db),
) -> CompareResponse:
    """Run the same prompt against multiple backends concurrently and store all outputs."""

    run_id = str(uuid4())
    user_prompt = build_user_prompt(payload)

    tasks = [
        execute_generation(
            backend,
            theme_name=payload.theme_name,
            tone_funny_pct=payload.tone_funny_pct,
            tone_emotion_pct=payload.tone_emotion_pct,
            prompt_keywords=payload.prompt_keywords,
            count=payload.count,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        for backend in payload.backends
    ]
    results = await asyncio.gather(*tasks)
    await persist_results(db, run_id=run_id, results=results)
    return build_compare_response(run_id, payload.theme_name, results)


@router.post("/single")
async def generate_single(
    payload: SingleRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Run one prompt against a single backend, store the result, and return it."""

    run_id = str(uuid4())
    user_prompt = build_user_prompt(payload)
    result = await execute_generation(
        payload.backend,
        theme_name=payload.theme_name,
        tone_funny_pct=payload.tone_funny_pct,
        tone_emotion_pct=payload.tone_emotion_pct,
        prompt_keywords=payload.prompt_keywords,
        count=payload.count,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    await persist_results(db, run_id=run_id, results=[result])

    return {
        "run_id": run_id,
        "theme_name": payload.theme_name,
        "backend": result["backend"],
        "model_name": result["model_name"],
        "success": result["success"],
        "generation_time_ms": result["generation_time_ms"],
        "phrases": result["phrases"],
        "best_phrase": result["best_phrase"],
        "error": result["error"],
    }
