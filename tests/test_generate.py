"""HTTP tests for the standalone LLM comparator application."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeResponse:
    """Minimal httpx-style response wrapper for mocked backend calls."""

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeAsyncClient:
    """Configurable async client used to mock Groq and Ollama backend calls."""

    response_map = {}
    default_delay = 0.0
    calls = []

    def __init__(self, *args, **kwargs):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @classmethod
    def reset(cls) -> None:
        cls.response_map = {}
        cls.default_delay = 0.0
        cls.calls = []

    async def post(self, url: str, *, headers=None, json=None):
        FakeAsyncClient.calls.append({"url": url, "json": json})
        if FakeAsyncClient.default_delay:
            await asyncio.sleep(FakeAsyncClient.default_delay)

        model = None
        if isinstance(json, dict):
            model = json.get("model")

        key = model or url
        payload = FakeAsyncClient.response_map.get(key)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise RuntimeError(f"No fake payload configured for {key}")
        return FakeResponse(payload)


def reload_modules(monkeypatch, tmp_path: Path):
    """Reload the local standalone app package against a temporary SQLite file."""

    db_path = tmp_path / "llm-comparator-test.db"
    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://fake-ollama")
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{db_path}")

    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            sys.modules.pop(module_name, None)

    importlib.invalidate_caches()
    main_module = importlib.import_module("app.main")
    database_module = importlib.import_module("app.database")
    models_module = importlib.import_module("app.models")
    generate_module = importlib.import_module("app.routers.generate")
    results_module = importlib.import_module("app.routers.results")
    return main_module, database_module, models_module, generate_module, results_module


def configure_success_payloads() -> None:
    """Install one successful response per backend model."""

    FakeAsyncClient.response_map = {
        "llama-3.3-70b-versatile": {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "phrases": [
                                    {"text": "May your Monday bloom with courage, warmth, and bright new hope!", "tone": "emotional", "word_count": 11},
                                    {"text": "Sending you bold energy and gentle joy for the week ahead!", "tone": "balanced", "word_count": 11},
                                ]
                            }
                        )
                    }
                }
            ],
            "usage": {"total_tokens": 123},
        },
        "qwen2.5:7b-instruct": {
            "message": {
                "content": json.dumps(
                    {
                        "phrases": [
                            {"text": "May your day begin with strength, smiles, and steady confidence ahead!", "tone": "balanced", "word_count": 11}
                        ]
                    }
                )
            },
            "prompt_eval_count": 40,
            "eval_count": 60,
        },
        "llama3.1:8b": {
            "message": {
                "content": json.dumps(
                    {
                        "phrases": [
                            {"text": "Wishing you a brave heart and a beautifully focused morning today!", "tone": "emotional", "word_count": 11}
                        ]
                    }
                )
            },
            "prompt_eval_count": 35,
            "eval_count": 55,
        },
        "mistral:7b": {
            "message": {
                "content": json.dumps(
                    {
                        "phrases": [
                            {"text": "Here is your reminder that this week can still surprise you kindly!", "tone": "balanced", "word_count": 12}
                        ]
                    }
                )
            },
            "prompt_eval_count": 30,
            "eval_count": 50,
        },
    }


async def count_rows(database_module, models_module) -> int:
    """Count persisted rows in the temporary SQLite database."""

    from sqlalchemy import select

    async with database_module.async_session_factory() as session:
        result = await session.execute(select(models_module.GenerationResult))
        return len(result.scalars().all())


def test_compare_endpoint_returns_run_id(monkeypatch, tmp_path) -> None:
    """The compare endpoint should return a new run identifier and backend results."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Motivational",
                "tone_funny_pct": 30,
                "tone_emotion_pct": 70,
                "prompt_keywords": ["strength", "monday", "energy"],
                "visual_style": "minimal sunrise",
                "count": 2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"]
    assert payload["theme_name"] == "Motivational"
    assert len(payload["results"]) == 4


def test_compare_runs_all_backends_concurrently(monkeypatch, tmp_path) -> None:
    """The compare endpoint should fire all backend requests concurrently."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    FakeAsyncClient.default_delay = 0.05
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    started_at = perf_counter()
    with TestClient(main_module.app) as client:
        response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Festival Joy",
                "tone_funny_pct": 65,
                "tone_emotion_pct": 20,
                "prompt_keywords": ["celebration", "family"],
                "visual_style": "bright festive",
                "count": 1,
            },
        )
    elapsed = perf_counter() - started_at

    assert response.status_code == 200
    assert len(FakeAsyncClient.calls) == 4
    assert elapsed < 0.18


def test_single_endpoint_returns_phrases(monkeypatch, tmp_path) -> None:
    """The single endpoint should return normalized phrases for one backend."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/generate/single",
            json={
                "theme_name": "Warm Wishes",
                "tone_funny_pct": 20,
                "tone_emotion_pct": 70,
                "prompt_keywords": ["family", "gratitude"],
                "visual_style": "soft watercolor",
                "backend": "groq",
                "count": 2,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["backend"] == "groq"
    assert payload["phrases"]
    assert payload["best_phrase"]


def test_failed_backend_returns_error_not_crash(monkeypatch, tmp_path) -> None:
    """A failing backend should be captured as one failed result, not crash the whole run."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    FakeAsyncClient.response_map["llama3.1:8b"] = RuntimeError("ollama unavailable")
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Steady Hope",
                "tone_funny_pct": 40,
                "tone_emotion_pct": 40,
                "prompt_keywords": ["hope"],
                "visual_style": "calm sunrise",
                "count": 1,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["all_succeeded"] is False
    failed_items = [item for item in payload["results"] if item["backend"] == "ollama_llama31"]
    assert failed_items[0]["success"] is False
    assert "ollama unavailable" in failed_items[0]["error"]


def test_results_stored_in_sqlite(monkeypatch, tmp_path) -> None:
    """Each compare run should persist one row per backend in SQLite."""

    main_module, database_module, models_module, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Morning Spark",
                "tone_funny_pct": 50,
                "tone_emotion_pct": 50,
                "prompt_keywords": ["spark"],
                "visual_style": "clean modern",
                "count": 1,
            },
        )

    assert response.status_code == 200
    assert asyncio.run(count_rows(database_module, models_module)) == 4


def test_get_runs_returns_list(monkeypatch, tmp_path) -> None:
    """The runs index should return stored run summaries."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        create_response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Joyful Start",
                "tone_funny_pct": 60,
                "tone_emotion_pct": 20,
                "prompt_keywords": ["joy"],
                "visual_style": "vibrant",
                "count": 1,
            },
        )
        runs_response = client.get("/results/runs")

    assert create_response.status_code == 200
    assert runs_response.status_code == 200
    payload = runs_response.json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["run_id"] == create_response.json()["run_id"]


def test_get_run_by_id_returns_comparison(monkeypatch, tmp_path) -> None:
    """The per-run detail endpoint should return all backend results side by side."""

    main_module, _, _, generate_module, _ = reload_modules(monkeypatch, tmp_path)
    FakeAsyncClient.reset()
    configure_success_payloads()
    monkeypatch.setattr(generate_module.httpx, "AsyncClient", FakeAsyncClient)

    with TestClient(main_module.app) as client:
        create_response = client.post(
            "/generate/compare",
            json={
                "theme_name": "Gentle Grace",
                "tone_funny_pct": 20,
                "tone_emotion_pct": 80,
                "prompt_keywords": ["grace"],
                "visual_style": "soft floral",
                "count": 1,
            },
        )
        run_id = create_response.json()["run_id"]
        detail_response = client.get(f"/results/runs/{run_id}")

    assert detail_response.status_code == 200
    payload = detail_response.json()
    assert payload["run_id"] == run_id
    assert len(payload["results"]) == 4
    assert "timing_comparison" in payload
