"""Structured application errors returned by the HTTP API."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application error with structured response metadata."""

    def __init__(
        self,
        *,
        error_type: str,
        message: str,
        response_status: int,
        backend: str | None = None,
        model: str | None = None,
        http_status: int | None = None,
        retry_after_ms: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.response_status = response_status
        self.backend = backend
        self.model = model
        self.http_status = http_status
        self.retry_after_ms = retry_after_ms
        self.details = details


class BusyServiceError(AppError):
    """Raised when the in-memory concurrency limiter rejects a request."""

    def __init__(self, *, backend: str | None, model: str | None, retry_after_ms: int) -> None:
        super().__init__(
            error_type="busy",
            message="The service is busy. Retry later.",
            response_status=429,
            backend=backend,
            model=model,
            http_status=429,
            retry_after_ms=retry_after_ms,
        )


class ValidationServiceError(AppError):
    """Raised when request or model validation fails."""

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_type="validation_error",
            message=message,
            response_status=400,
            backend=backend,
            model=model,
            http_status=400,
            details=details,
        )


class NotConfiguredError(AppError):
    """Raised when an optional backend has not been configured."""

    def __init__(self, message: str, *, backend: str, model: str | None) -> None:
        super().__init__(
            error_type="not_configured",
            message=message,
            response_status=503,
            backend=backend,
            model=model,
            http_status=503,
        )


class ProviderRateLimitedError(AppError):
    """Raised when an upstream provider throttles the request."""

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        model: str | None,
        http_status: int,
        retry_after_ms: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_type="rate_limited",
            message=message,
            response_status=429,
            backend=backend,
            model=model,
            http_status=http_status,
            retry_after_ms=retry_after_ms,
            details=details,
        )


class ProviderError(AppError):
    """Raised when an upstream provider returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        model: str | None,
        http_status: int | None = None,
        response_status: int = 502,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_type="provider_error",
            message=message,
            response_status=response_status,
            backend=backend,
            model=model,
            http_status=http_status,
            details=details,
        )


class NetworkError(AppError):
    """Raised when a network call times out or fails in transit."""

    def __init__(
        self,
        message: str,
        *,
        backend: str,
        model: str | None,
        response_status: int = 504,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_type="network_error",
            message=message,
            response_status=response_status,
            backend=backend,
            model=model,
            http_status=response_status,
            details=details,
        )


class ServiceUnreachableError(AppError):
    """Raised when a required local service cannot be reached."""

    def __init__(self, message: str, *, backend: str, model: str | None) -> None:
        super().__init__(
            error_type="service_unreachable",
            message=message,
            response_status=503,
            backend=backend,
            model=model,
            http_status=503,
        )


class InternalServiceError(AppError):
    """Raised for unexpected application failures."""

    def __init__(self, message: str = "Internal server error.") -> None:
        super().__init__(
            error_type="internal_error",
            message=message,
            response_status=500,
            http_status=500,
        )
