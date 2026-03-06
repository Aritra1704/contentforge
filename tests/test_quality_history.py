"""Tests for /quality/history endpoint."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from tests.test_generate import reload_app


@pytest.mark.asyncio
async def test_quality_history_endpoint_returns_runs(monkeypatch) -> None:
    """History endpoint should return rows from the quality memory backend."""

    main_module, _ = reload_app(monkeypatch)

    async def fake_fetch_quality_history(*, limit: int = 50, theme_name: str | None = None, keyword: str | None = None):
        assert limit == 5
        assert theme_name == "Warm"
        assert keyword == "family"
        return [
            {
                "run_id": "run-1",
                "created_at": datetime(2026, 3, 6, tzinfo=timezone.utc),
                "theme_name": "Warm Wishes",
                "keywords": ["family", "gratitude"],
                "tone_config": {"tone_funny_pct": 20, "tone_emotion_pct": 70},
                "output_spec": {"format": "one_liner"},
                "backend": "ollama",
                "model": "qwen2.5:7b-instruct",
                "output_text": "Kind words steady difficult mornings.",
                "quality_score_json": {"total": 88},
                "judge_json": None,
                "detected_cliches": ["shine bright"],
                "repetition_flags": ["wishing you"],
                "json_leak_flag": False,
            }
        ]

    monkeypatch.setattr("app.routers.quality.fetch_quality_history", fake_fetch_quality_history)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/quality/history", params={"limit": 5, "theme_name": "Warm", "keyword": "family"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["runs"]) == 1
    assert payload["runs"][0]["run_id"] == "run-1"
