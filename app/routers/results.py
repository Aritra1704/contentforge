"""Endpoints for browsing and exporting stored LLM comparison results."""

from __future__ import annotations

import csv
import io
import json
from statistics import mean
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import GenerationResult

router = APIRouter(prefix="/results", tags=["results"])


def deserialize_phrases(raw_value: str) -> list[dict[str, Any]]:
    """Safely convert stored JSON phrase arrays back into Python objects."""

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def summarize_run(run_id: str, rows: list[GenerationResult]) -> dict[str, Any]:
    """Build one high-level run summary from all backend records."""

    fastest_backend = None
    successful_rows = [row for row in rows if row.success]
    if successful_rows:
        fastest_backend = min(successful_rows, key=lambda row: row.generation_time_ms).backend

    return {
        "run_id": run_id,
        "created_at": min(row.created_at for row in rows).isoformat(),
        "theme_name": rows[0].input_theme,
        "backends_tested": [row.backend for row in rows],
        "all_succeeded": all(row.success for row in rows),
        "fastest_backend": fastest_backend,
    }


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)) -> list[dict[str, Any]]:
    """Return one summary row per stored comparison run."""

    statement = select(GenerationResult).order_by(GenerationResult.created_at.desc(), GenerationResult.id.desc())
    result = await db.execute(statement)
    rows = list(result.scalars().all())

    grouped: dict[str, list[GenerationResult]] = {}
    for row in rows:
        grouped.setdefault(row.run_id, []).append(row)

    summaries = [summarize_run(run_id, group_rows) for run_id, group_rows in grouped.items()]
    summaries.sort(key=lambda item: item["created_at"], reverse=True)
    return summaries


@router.get("/runs/{run_id}")
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Return the full side-by-side comparison for one stored run."""

    statement = (
        select(GenerationResult)
        .where(GenerationResult.run_id == run_id)
        .order_by(GenerationResult.created_at.asc(), GenerationResult.id.asc())
    )
    result = await db.execute(statement)
    rows = list(result.scalars().all())

    if not rows:
        raise HTTPException(status_code=404, detail="Run not found.")

    successful_rows = [row for row in rows if row.success]
    fastest_backend = None
    if successful_rows:
        fastest_backend = min(successful_rows, key=lambda row: row.generation_time_ms).backend

    return {
        "run_id": run_id,
        "created_at": rows[0].created_at.isoformat(),
        "theme_name": rows[0].input_theme,
        "timing_comparison": {
            row.backend: row.generation_time_ms
            for row in rows
        },
        "fastest_backend": fastest_backend,
        "results": [
            {
                "backend": row.backend,
                "model_name": row.model_name,
                "prompt_type": row.prompt_type,
                "success": row.success,
                "generation_time_ms": row.generation_time_ms,
                "token_count": row.token_count,
                "best_phrase": row.best_phrase,
                "error_message": row.error_message,
                "phrases": deserialize_phrases(row.parsed_phrases),
                "raw_output": row.raw_output,
            }
            for row in rows
        ],
    }


@router.get("/runs/{run_id}/export")
async def export_run(run_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Export all rows from one run as a CSV attachment."""

    statement = (
        select(GenerationResult)
        .where(GenerationResult.run_id == run_id)
        .order_by(GenerationResult.created_at.asc(), GenerationResult.id.asc())
    )
    result = await db.execute(statement)
    rows = list(result.scalars().all())

    if not rows:
        raise HTTPException(status_code=404, detail="Run not found.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "run_id",
            "backend",
            "model_name",
            "prompt_type",
            "theme_name",
            "generation_time_ms",
            "token_count",
            "success",
            "best_phrase",
            "error_message",
            "parsed_phrases",
            "created_at",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.run_id,
                row.backend,
                row.model_name,
                row.prompt_type,
                row.input_theme,
                row.generation_time_ms,
                row.token_count,
                row.success,
                row.best_phrase,
                row.error_message,
                row.parsed_phrases,
                row.created_at.isoformat(),
            ]
        )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)) -> dict[str, Any]:
    """Return aggregate timing and success-rate statistics across all runs."""

    statement = select(GenerationResult)
    result = await db.execute(statement)
    rows = list(result.scalars().all())

    grouped: dict[str, list[GenerationResult]] = {}
    for row in rows:
        grouped.setdefault(row.backend, []).append(row)

    average_generation_time_ms = {
        backend: round(mean(item.generation_time_ms for item in backend_rows), 2)
        for backend, backend_rows in grouped.items()
    }
    success_rate = {
        backend: round(
            sum(1 for item in backend_rows if item.success) / len(backend_rows),
            4,
        )
        for backend, backend_rows in grouped.items()
    }

    return {
        "total_runs": len({row.run_id for row in rows}),
        "average_generation_time_ms": average_generation_time_ms,
        "success_rate": success_rate,
    }
