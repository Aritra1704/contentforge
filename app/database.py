"""Async SQLite database setup for the standalone LLM comparator."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base class for ORM models."""


engine = create_async_engine(settings.db_url, future=True)
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for one request."""

    async with async_session_factory() as session:
        yield session


async def init_database() -> None:
    """Create all tables on startup for the self-contained SQLite app."""

    import app.models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Dispose of the engine during application shutdown."""

    await engine.dispose()
