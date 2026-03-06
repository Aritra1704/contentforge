"""Pairwise round-robin judging endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.observability import update_request_context
from app.schemas import RoundRobinJudgeRequest, RoundRobinJudgeResponse
from src.judge.round_robin import JudgeProviderError, JudgeTimeoutError, run_round_robin_judge

router = APIRouter(prefix="/judge", tags=["judge"])


@router.post("/round-robin", response_model=RoundRobinJudgeResponse)
async def judge_round_robin(request: Request, payload: RoundRobinJudgeRequest) -> RoundRobinJudgeResponse:
    """Judge N candidates pairwise and return final round-robin ranking."""

    update_request_context(
        request,
        backend=f"judge:{settings.judge_provider}",
        model=settings.judge_model,
    )
    try:
        return await run_round_robin_judge(payload)
    except JudgeTimeoutError as exc:
        return JSONResponse(
            status_code=504,
            content={
                "error_type": "judge_timeout",
                "message": f"Judge timed out after {exc.timeout_seconds:g} seconds",
                "provider": exc.provider,
                "model": exc.model,
            },
        )
    except JudgeProviderError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_type": exc.error_type,
                "message": exc.message,
                "provider": exc.provider,
                "model": exc.model,
            },
        )
