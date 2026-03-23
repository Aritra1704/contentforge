"""Tests for quality-memory augmentation helpers."""

from __future__ import annotations

import pytest

import app.quality_memory as quality_memory
from app.schemas import GenerateSingleRequest


def base_payload(**overrides) -> dict:
    """Return a valid single-generation payload."""

    payload = {
        "theme_name": "Warm Wishes",
        "tone_funny_pct": 20,
        "tone_emotion_pct": 70,
        "prompt_keywords": ["family", "gratitude"],
        "visual_style": "soft watercolor",
        "backend": "ollama",
        "model": "qwen2.5:7b-instruct",
        "count": 3,
        "max_tokens": 300,
        "temperature": 0.8,
        "avoid_cliches": True,
    }
    payload.update(overrides)
    return payload


def test_derive_memory_avoid_phrases_caps_to_ten() -> None:
    """Derived avoid phrases from history should be capped to 10."""

    rows = [
        {
            "detected_cliches": ["shine bright", "you got this", "rise and shine"],
            "repetition_flags": ["wishing you", "start your"],
        },
        {
            "detected_cliches": ["shine bright", "positive vibes", "make it happen"],
            "repetition_flags": ["wishing you", "wishing you"],
        },
        {
            "detected_cliches": ["inner strength", "new week", "believe in yourself"],
            "repetition_flags": ["start your", "keep going"],
        },
    ]

    phrases = quality_memory.derive_memory_avoid_phrases(rows, cap=10)
    assert len(phrases) <= 10
    assert "shine bright" in phrases
    assert "wishing you" in phrases


@pytest.mark.asyncio
async def test_augment_avoid_phrases_with_memory_adds_history_phrases(monkeypatch) -> None:
    """Request avoid_phrases should be augmented with memory-derived phrases."""

    request = GenerateSingleRequest.model_validate(
        base_payload(
            avoid_phrases=["new week"],
            prompt_keywords=["family"],
        )
    )

    async def fake_fetch_similar_runs(theme_name: str, keywords: list[str], *, limit: int = 5):
        assert theme_name == "Warm Wishes"
        assert keywords == ["family"]
        assert limit == 5
        return [
            {
                "detected_cliches": ["hope this message", "shine bright", "same old sparkle", "warm wishes friend"],
                "repetition_flags": ["wishing you", "start your"],
            }
        ]

    monkeypatch.setattr(quality_memory, "fetch_similar_runs", fake_fetch_similar_runs)
    await quality_memory.augment_avoid_phrases_with_memory(request)

    normalized = {item.lower() for item in request.avoid_phrases}
    assert "new week" in normalized
    assert "hope this message" in normalized
    assert "same old sparkle" in normalized
    assert "shine bright" not in normalized
    assert "wishing you" not in normalized
    assert "warm wishes friend" not in normalized
    assert len(request.avoid_phrases) <= 11


@pytest.mark.asyncio
async def test_augment_avoid_phrases_with_memory_skips_prompt_overlap(monkeypatch) -> None:
    """Memory phrases overlapping the prompt/theme should not become hard blockers."""

    request = GenerateSingleRequest.model_validate(
        base_payload(
            theme_name="Warm Birthday Message",
            prompt_keywords=["warm birthday message", "close friend"],
        )
    )

    async def fake_fetch_similar_runs(theme_name: str, keywords: list[str], *, limit: int = 5):
        assert theme_name == "Warm Birthday Message"
        assert keywords == ["warm birthday message", "close friend"]
        assert limit == 5
        return [
            {
                "detected_cliches": ["warm birthday", "close friend note", "hope this message"],
                "repetition_flags": [],
            }
        ]

    monkeypatch.setattr(quality_memory, "fetch_similar_runs", fake_fetch_similar_runs)
    await quality_memory.augment_avoid_phrases_with_memory(request)

    normalized = {item.lower() for item in request.avoid_phrases}
    assert "warm birthday" not in normalized
    assert "close friend note" not in normalized
    assert "hope this message" in normalized
