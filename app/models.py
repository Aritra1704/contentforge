"""ORM models for persisted generation comparison results."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class GenerationResult(Base):
    """Stored output for one backend execution in a comparison run."""

    __tablename__ = "generation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    backend: Mapped[str] = mapped_column(String(50), index=True)
    model_name: Mapped[str] = mapped_column(String(100))
    prompt_type: Mapped[str] = mapped_column(String(50))
    input_theme: Mapped[str] = mapped_column(String(255))
    input_tone_funny_pct: Mapped[int] = mapped_column(Integer)
    input_tone_emotion_pct: Mapped[int] = mapped_column(Integer)
    input_keywords: Mapped[str] = mapped_column(Text)
    raw_output: Mapped[str] = mapped_column(Text)
    parsed_phrases: Mapped[str] = mapped_column(Text)
    best_phrase: Mapped[str] = mapped_column(Text)
    generation_time_ms: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
