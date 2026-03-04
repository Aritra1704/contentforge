"""FastAPI entrypoint for the stateless content generation service."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.routers.generate import router as generate_router
from app.routers.system import router as system_router

app = FastAPI(title=settings.service_name, version=settings.service_version)
app.include_router(system_router)
app.include_router(generate_router)
