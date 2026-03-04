"""FastAPI entrypoint for the stateless content generation service."""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI

from app.config import settings
from app.errors import AppError
from app.observability import (
    app_error_handler,
    configure_logging,
    generic_error_handler,
    request_context_middleware,
    validation_error_handler,
)
from app.routers.generate import legacy_router as generation_router
from app.routers.generate import router as generate_router
from app.routers.system import router as system_router

configure_logging()
app = FastAPI(title=settings.service_name, version=settings.service_version)
app.middleware("http")(request_context_middleware)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, generic_error_handler)
app.include_router(system_router)
app.include_router(generate_router)
app.include_router(generation_router)
