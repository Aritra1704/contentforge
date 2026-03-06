"""PostgreSQL-backed quality memory for retrieval and run logging."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
import json
import logging
import re
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import settings
from app.quality import (
    blocked_cliche_hits,
    detect_json_leakage,
    to_plain_text,
)
from app.schemas import (
    GenerateCompareModelsRequest,
    GenerateSingleRequest,
    QualityRunHistoryItem,
    QualityScore,
)

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import branch depends on environment
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - import branch depends on environment
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

OPENING_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*•]|\d{1,3}[\)\].:-])?\s*")
_SCHEMA_READY = False
_SCHEMA_LOCK = Lock()
_DEPENDENCY_WARNING_EMITTED = False

CREATE_QUALITY_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS quality_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    theme_name TEXT NOT NULL,
    keywords TEXT[] NOT NULL DEFAULT '{}',
    tone_config JSONB NOT NULL,
    output_spec JSONB NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    output_text TEXT NOT NULL,
    quality_score_json JSONB NOT NULL,
    judge_json JSONB NULL,
    detected_cliches TEXT[] NOT NULL DEFAULT '{}',
    repetition_flags TEXT[] NOT NULL DEFAULT '{}',
    json_leak_flag BOOLEAN NOT NULL DEFAULT FALSE
);
"""

CREATE_QUALITY_RUNS_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_quality_runs_created_at ON quality_runs (created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_quality_runs_theme_name ON quality_runs (LOWER(theme_name));",
    "CREATE INDEX IF NOT EXISTS idx_quality_runs_keywords ON quality_runs USING GIN (keywords);",
]


def _emit_dependency_warning_once() -> None:
    """Log a single warning when psycopg is unavailable."""

    global _DEPENDENCY_WARNING_EMITTED
    if _DEPENDENCY_WARNING_EMITTED:
        return
    _DEPENDENCY_WARNING_EMITTED = True
    logger.warning("quality_memory_disabled reason=missing_psycopg")


def is_quality_memory_enabled() -> bool:
    """Return whether quality memory storage/retrieval should be active."""

    if not settings.quality_memory_enabled:
        return False
    if not settings.quality_memory_dsn.strip():
        return False
    if psycopg is None or dict_row is None:
        _emit_dependency_warning_once()
        return False
    return True


def _connect():
    """Open one psycopg connection with dict-row decoding."""

    assert psycopg is not None
    assert dict_row is not None
    return psycopg.connect(
        settings.quality_memory_dsn,
        row_factory=dict_row,
        autocommit=True,
    )


def _ensure_schema_sync() -> None:
    """Create quality memory table/indexes once per process when enabled."""

    global _SCHEMA_READY
    if not is_quality_memory_enabled() or _SCHEMA_READY:
        return

    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_QUALITY_RUNS_SQL)
                for statement in CREATE_QUALITY_RUNS_INDEX_SQL:
                    cur.execute(statement)
        _SCHEMA_READY = True


async def ensure_quality_memory_schema() -> None:
    """Async wrapper to ensure table schema exists."""

    if not is_quality_memory_enabled():
        return
    try:
        await asyncio.to_thread(_ensure_schema_sync)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("quality_memory_schema_failed message=%s", str(exc))


def _normalize_phrase(value: str) -> str:
    """Normalize phrase text for dedupe and comparison."""

    return " ".join(value.strip().split()).lower()


def _extract_openings(lines: Iterable[str]) -> list[str]:
    """Extract two-word openings from generated lines."""

    openings: list[str] = []
    for line in lines:
        cleaned = OPENING_PREFIX_PATTERN.sub("", line.strip())
        if not cleaned:
            continue
        tokens = [token for token in re.findall(r"[A-Za-z0-9']+", cleaned.lower()) if token]
        if len(tokens) < 2:
            continue
        openings.append(f"{tokens[0]} {tokens[1]}")
    return openings


def extract_repetition_flags_from_text(text: str) -> list[str]:
    """Detect repeated openings from one output text block."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    counts = Counter(_extract_openings(lines))
    return [opening for opening, _ in counts.most_common(10)]


def derive_memory_avoid_phrases(similar_runs: list[dict[str, Any]], *, cap: int = 10) -> list[str]:
    """Build capped avoid-phrase list from historical cliches and repeated openings."""

    cliche_counter: Counter[str] = Counter()
    opening_counter: Counter[str] = Counter()

    for row in similar_runs:
        cliches = row.get("detected_cliches") or []
        repetitions = row.get("repetition_flags") or []
        if isinstance(cliches, list):
            for phrase in cliches:
                normalized = _normalize_phrase(str(phrase))
                if normalized:
                    cliche_counter[normalized] += 1
        if isinstance(repetitions, list):
            for phrase in repetitions:
                normalized = _normalize_phrase(str(phrase))
                if normalized:
                    opening_counter[normalized] += 1

    ordered: list[str] = []
    for phrase, _ in cliche_counter.most_common():
        ordered.append(phrase)
    for phrase, _ in opening_counter.most_common():
        if phrase not in ordered:
            ordered.append(phrase)
    return ordered[:cap]


def _payload_keywords(payload: GenerateSingleRequest | GenerateCompareModelsRequest) -> list[str]:
    """Return normalized keyword tokens for storage/retrieval."""

    values: list[str] = []
    seen: set[str] = set()
    for item in payload.prompt_keywords:
        candidate = _normalize_phrase(item)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        values.append(candidate)
    return values


def _payload_tone_config(payload: GenerateSingleRequest | GenerateCompareModelsRequest) -> dict[str, Any]:
    """Return tone config snapshot for storage."""

    return {
        "tone_funny_pct": payload.tone_funny_pct,
        "tone_emotion_pct": payload.tone_emotion_pct,
        "tone_style": payload.tone_style,
        "emoji_policy": payload.emoji_policy,
        "audience": payload.audience,
        "avoid_cliches": payload.avoid_cliches,
    }


def _fetch_similar_runs_sync(theme_name: str, keywords: list[str], limit: int) -> list[dict[str, Any]]:
    """Fetch recent similar runs by theme match or keyword overlap."""

    _ensure_schema_sync()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    run_id,
                    created_at,
                    theme_name,
                    keywords,
                    tone_config,
                    output_spec,
                    backend,
                    model,
                    output_text,
                    quality_score_json,
                    judge_json,
                    detected_cliches,
                    repetition_flags,
                    json_leak_flag
                FROM quality_runs
                WHERE
                    LOWER(theme_name) = LOWER(%s)
                    OR LOWER(theme_name) LIKE LOWER(%s)
                    OR keywords && %s::text[]
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (theme_name, f"%{theme_name}%", keywords, limit),
            )
            rows = cur.fetchall()
    return [dict(row) for row in rows]


async def fetch_similar_runs(
    theme_name: str,
    keywords: list[str],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Async wrapper for fetching similar historical quality runs."""

    if not is_quality_memory_enabled():
        return []
    try:
        return await asyncio.to_thread(_fetch_similar_runs_sync, theme_name, keywords, limit)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("quality_memory_fetch_failed message=%s", str(exc))
        return []


async def augment_avoid_phrases_with_memory(
    payload: GenerateSingleRequest | GenerateCompareModelsRequest,
) -> None:
    """Augment avoid_phrases from similar runs when avoid_cliches is enabled."""

    if not payload.avoid_cliches:
        return

    similar_runs = await fetch_similar_runs(
        payload.theme_name,
        _payload_keywords(payload),
        limit=5,
    )
    if not similar_runs:
        return

    additions = derive_memory_avoid_phrases(similar_runs, cap=10)
    if not additions:
        return

    merged: list[str] = []
    seen: set[str] = set()
    for item in payload.avoid_phrases:
        normalized = _normalize_phrase(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(item.strip())

    added = 0
    for phrase in additions:
        if phrase in seen:
            continue
        if added >= 10:
            break
        seen.add(phrase)
        merged.append(phrase)
        added += 1

    payload.avoid_phrases = merged


def _insert_quality_run_sync(record: dict[str, Any]) -> None:
    """Insert one quality run row."""

    _ensure_schema_sync()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO quality_runs (
                    run_id,
                    created_at,
                    theme_name,
                    keywords,
                    tone_config,
                    output_spec,
                    backend,
                    model,
                    output_text,
                    quality_score_json,
                    judge_json,
                    detected_cliches,
                    repetition_flags,
                    json_leak_flag
                )
                VALUES (
                    %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb,
                    %s::jsonb, %s, %s, %s
                )
                """,
                (
                    record["run_id"],
                    record["created_at"],
                    record["theme_name"],
                    record["keywords"],
                    record["tone_config_json"],
                    record["output_spec_json"],
                    record["backend"],
                    record["model"],
                    record["output_text"],
                    record["quality_score_json"],
                    record["judge_json"],
                    record["detected_cliches"],
                    record["repetition_flags"],
                    record["json_leak_flag"],
                ),
            )


async def store_quality_run(
    *,
    payload: GenerateSingleRequest | GenerateCompareModelsRequest,
    backend: str,
    model: str,
    output_text: str,
    quality_score: QualityScore,
    judge_json: dict[str, Any] | None = None,
) -> str | None:
    """Persist one quality run row and return run_id when successful."""

    if not is_quality_memory_enabled():
        return None

    repetition_flags = extract_repetition_flags_from_text(output_text)
    detected_cliches = blocked_cliche_hits(  # type: ignore[arg-type]
        payload,  # both payload models share required fields
        output_text,
    )
    run_id = str(uuid4())
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc),
        "theme_name": payload.theme_name,
        "keywords": _payload_keywords(payload),
        "tone_config_json": json.dumps(_payload_tone_config(payload), ensure_ascii=True),
        "output_spec_json": json.dumps(
            payload.output_spec.model_dump(mode="json") if payload.output_spec is not None else {},
            ensure_ascii=True,
        ),
        "backend": backend,
        "model": model,
        "output_text": output_text,
        "quality_score_json": json.dumps(
            quality_score.model_dump(mode="json"),
            ensure_ascii=True,
        ),
        "judge_json": json.dumps(judge_json, ensure_ascii=True) if judge_json is not None else None,
        "detected_cliches": detected_cliches,
        "repetition_flags": repetition_flags,
        "json_leak_flag": detect_json_leakage(output_text),
    }

    try:
        await asyncio.to_thread(_insert_quality_run_sync, record)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("quality_memory_insert_failed message=%s", str(exc))
        return None
    return run_id


def _fetch_quality_history_sync(
    *,
    limit: int,
    theme_name: str | None,
    keyword: str | None,
) -> list[dict[str, Any]]:
    """Fetch quality history with optional filters."""

    _ensure_schema_sync()
    clauses: list[str] = []
    params: list[Any] = []

    if theme_name:
        clauses.append("LOWER(theme_name) LIKE LOWER(%s)")
        params.append(f"%{theme_name}%")
    if keyword:
        clauses.append("%s = ANY(keywords)")
        params.append(_normalize_phrase(keyword))

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    query = f"""
        SELECT
            run_id,
            created_at,
            theme_name,
            keywords,
            tone_config,
            output_spec,
            backend,
            model,
            output_text,
            quality_score_json,
            judge_json,
            detected_cliches,
            repetition_flags,
            json_leak_flag
        FROM quality_runs
        {where_sql}
        ORDER BY created_at DESC
        LIMIT %s
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return [dict(row) for row in rows]


async def fetch_quality_history(
    *,
    limit: int = 50,
    theme_name: str | None = None,
    keyword: str | None = None,
) -> list[QualityRunHistoryItem]:
    """Return typed quality history rows for API responses."""

    if not is_quality_memory_enabled():
        return []
    try:
        rows = await asyncio.to_thread(
            _fetch_quality_history_sync,
            limit=limit,
            theme_name=theme_name,
            keyword=keyword,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("quality_history_fetch_failed message=%s", str(exc))
        return []

    history: list[QualityRunHistoryItem] = []
    for row in rows:
        history.append(
            QualityRunHistoryItem(
                run_id=str(row.get("run_id", "")),
                created_at=row.get("created_at"),
                theme_name=str(row.get("theme_name", "")),
                keywords=list(row.get("keywords") or []),
                tone_config=dict(row.get("tone_config") or {}),
                output_spec=dict(row.get("output_spec") or {}),
                backend=str(row.get("backend", "")),
                model=str(row.get("model", "")),
                output_text=str(row.get("output_text", "")),
                quality_score_json=dict(row.get("quality_score_json") or {}),
                judge_json=dict(row.get("judge_json") or {}) if row.get("judge_json") is not None else None,
                detected_cliches=[str(item) for item in (row.get("detected_cliches") or [])],
                repetition_flags=[str(item) for item in (row.get("repetition_flags") or [])],
                json_leak_flag=bool(row.get("json_leak_flag", False)),
            )
        )
    return history


def output_text_for_storage(output: Any) -> str:
    """Convert GeneratedOutput-like objects to text for persistence."""

    return to_plain_text(output).strip()
