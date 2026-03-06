"""Request context, logging, and exception-handling helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
import logging
import math
import re
import sys
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.errors import AppError, InternalServiceError, ValidationServiceError
from app.schemas import ErrorBody, ErrorResponse, ResponseMeta

REQUEST_ID_HEADER = "X-Request-Id"
TRACE_ID_HEADER = "X-Trace-Id"
logger = logging.getLogger("app.request")


def configure_logging() -> None:
    """Configure root logging once for stdout-based service logs."""

    root_logger = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        root_logger.setLevel(settings.log_level.upper())
        return

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    setattr(configure_logging, "_configured", True)


def _serialize_log_value(value: Any) -> str:
    """Serialize one logging field value as compact JSON."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def format_log_line(event: str, **fields: Any) -> str:
    """Return a consistent structured log line."""

    payload = {"event": event}
    payload.update({key: value for key, value in fields.items() if value is not None and value != ""})
    return " ".join(f"{key}={_serialize_log_value(value)}" for key, value in payload.items())


def sanitize_text(value: str | None, *, limit: int = 240) -> str | None:
    """Remove line breaks and obvious auth token patterns from text fields."""

    if value is None:
        return None

    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", value)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > limit:
        return f"{sanitized[: limit - 3]}..."
    return sanitized


def sanitize_details(value: Any) -> Any:
    """Recursively sanitize details before returning them to clients or logs."""

    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, BaseException):
        return sanitize_text(str(value))
    if isinstance(value, dict):
        return {str(key): sanitize_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_details(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_details(item) for item in value]
    return value


async def _extract_request_context_from_body(request: Request) -> None:
    """Best-effort extraction of correlation fields from JSON request bodies."""

    if request.method not in {"POST", "PUT", "PATCH"}:
        return

    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return

    body = await request.body()
    if not body:
        return

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return

    if not isinstance(payload, dict):
        return

    if not getattr(request.state, "trace_id", None):
        trace_id = payload.get("trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            request.state.trace_id = trace_id.strip()

    backend = payload.get("backend")
    if isinstance(backend, str) and backend.strip():
        request.state.backend = backend.strip()

    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        request.state.model = model.strip()


def get_request_context(request: Request) -> dict[str, Any]:
    """Return the current request correlation fields."""

    return {
        "request_id": getattr(request.state, "request_id", None),
        "trace_id": getattr(request.state, "trace_id", None),
        "backend": getattr(request.state, "backend", None),
        "model": getattr(request.state, "model", None),
        "method": request.method,
        "path": request.url.path,
    }


def update_request_context(
    request: Request,
    *,
    trace_id: str | None = None,
    backend: str | None = None,
    model: str | None = None,
) -> None:
    """Update request-scoped correlation fields from route payloads."""

    if trace_id:
        request.state.trace_id = trace_id
    if backend:
        request.state.backend = backend
    if model:
        request.state.model = model


def build_error_response(
    request: Request,
    error: AppError,
    *,
    latency_ms: int | None = None,
) -> JSONResponse:
    """Create the standard JSON error response and attach correlation headers."""

    meta = ResponseMeta(
        latency_ms=latency_ms,
        request_id=getattr(request.state, "request_id", ""),
        trace_id=getattr(request.state, "trace_id", None),
        busy=error.error_type == "busy",
    )
    response = JSONResponse(
        status_code=error.response_status,
        content=ErrorResponse(
            ok=False,
            error=ErrorBody(
                error_type=error.error_type,
                message=error.message,
                backend=error.backend,
                model=error.model,
                http_status=error.http_status,
                retry_after_ms=error.retry_after_ms,
                details=sanitize_details(error.details),
            ),
            meta=meta,
        ).model_dump(mode="json", exclude_none=True),
    )
    response.headers[REQUEST_ID_HEADER] = getattr(request.state, "request_id", "")
    if error.retry_after_ms is not None:
        response.headers["Retry-After"] = str(max(1, math.ceil(error.retry_after_ms / 1000)))
    return response


def _log_error(request: Request, error: AppError, exc: Exception) -> None:
    """Log one structured error with correlation IDs and traceback."""

    context = get_request_context(request)
    context.update(
        {
            "error_type": error.error_type,
            "message": error.message,
            "http_status": error.http_status,
            "retry_after_ms": error.retry_after_ms,
            "exception_class": exc.__class__.__name__,
        }
    )

    if error.error_type == "busy":
        logger.info(format_log_line("request_busy", **context))
        return

    logger.error(format_log_line("request_failed", **context), exc_info=exc)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Render one structured AppError response."""

    _log_error(request, exc, exc)
    return build_error_response(request, exc)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render request validation errors in the shared error format."""

    error = ValidationServiceError(
        "Request validation failed.",
        backend=getattr(request.state, "backend", None),
        model=getattr(request.state, "model", None),
        details={"errors": exc.errors()},
    )
    _log_error(request, error, exc)
    return build_error_response(request, error)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render unexpected errors without exposing internals to callers."""

    error = InternalServiceError()
    _log_error(request, error, exc)
    return build_error_response(request, error)


async def request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach correlation IDs, log start/end, and emit request headers."""

    request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
    request.state.trace_id = request.headers.get(TRACE_ID_HEADER) or None
    request.state.backend = None
    request.state.model = None

    await _extract_request_context_from_body(request)

    started_at = perf_counter()
    logger.info(format_log_line("request_started", **get_request_context(request)))
    response = await call_next(request)
    latency_ms = int((perf_counter() - started_at) * 1000)
    response.headers[REQUEST_ID_HEADER] = request.state.request_id
    logger.info(
        format_log_line(
            "request_completed",
            **get_request_context(request),
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
    )
    return response
