"""FastAPI entrypoint for the standalone LLM comparator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import close_database, init_database
from app.routers.generate import router as generate_router
from app.routers.results import router as results_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize SQLite tables on startup and dispose of the engine on shutdown."""

    await init_database()
    try:
        yield
    finally:
        await close_database()


app = FastAPI(title="LLM Comparator", version="1.0.0", lifespan=lifespan)
app.include_router(generate_router)
app.include_router(results_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Return a minimal health payload for the standalone comparison service."""

    return {"status": "ok", "app": "LLM Comparator"}
