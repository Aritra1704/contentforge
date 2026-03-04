"""HTTP tests for the stateless llm-comparator service."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeResponse:
    """Minimal httpx-style response wrapper for mocked upstream calls."""

    def __init__(self, payload: dict, *, url: str, method: str = "GET", status_code: int = 200) -> None:
        self._payload = payload
        self.url = url
        self.method = method
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return

        request = httpx.Request(self.method, self.url)
        response = httpx.Response(self.status_code, request=request, json=self._payload)
        raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)


class FakeAsyncClient:
    """Configurable async client used to mock Ollama and Groq calls."""

    get_payloads: dict[str, dict | Exception] = {}
    post_payloads: dict[str, dict | Exception] = {}
    calls: list[dict] = []
    started_event: asyncio.Event | None = None
    release_event: asyncio.Event | None = None

    def __init__(self, *args, **kwargs) -> None:
        return None

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    @classmethod
    def reset(cls) -> None:
        cls.get_payloads = {}
        cls.post_payloads = {}
        cls.calls = []
        cls.started_event = None
        cls.release_event = None

    async def get(self, url: str, *, headers=None, json=None):
        FakeAsyncClient.calls.append({"method": "GET", "url": url, "headers": headers, "json": json})
        payload = FakeAsyncClient.get_payloads.get(url)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise RuntimeError(f"No fake GET payload configured for {url}")
        return FakeResponse(payload, url=url, method="GET")

    async def post(self, url: str, *, headers=None, json=None):
        FakeAsyncClient.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})

        if url.endswith("/api/chat") and FakeAsyncClient.started_event is not None:
            FakeAsyncClient.started_event.set()
        if url.endswith("/api/chat") and FakeAsyncClient.release_event is not None:
            await FakeAsyncClient.release_event.wait()

        key = url
        if isinstance(json, dict) and "model" in json:
            key = str(json["model"])

        payload = FakeAsyncClient.post_payloads.get(key)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise RuntimeError(f"No fake POST payload configured for {key}")
        return FakeResponse(payload, url=url, method="POST")


def reload_app(
    monkeypatch,
    *,
    groq_api_key: str | None = "test-groq-key",
    max_concurrent_jobs: str = "1",
    max_queue: str = "0",
):
    """Reload the app package against a clean environment snapshot."""

    monkeypatch.chdir(PROJECT_ROOT)
    monkeypatch.setenv("SERVICE_VERSION", "test-version")
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODELS", "mistral:7b,qwen2.5:7b-instruct,llama3.1:8b")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODELS", "nomic-embed-text:latest")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", max_concurrent_jobs)
    monkeypatch.setenv("MAX_QUEUE", max_queue)
    monkeypatch.setenv("BUSY_RETRY_AFTER_MS", "2000")
    monkeypatch.setenv("REQUEST_TIMEOUT_SEC", "120")

    if groq_api_key is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", groq_api_key)

    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            sys.modules.pop(module_name, None)

    importlib.invalidate_caches()
    main_module = importlib.import_module("app.main")
    llm_module = importlib.import_module("app.llm")
    return main_module, llm_module


def configure_default_payloads() -> None:
    """Install the default Ollama tag listing and successful model responses."""

    FakeAsyncClient.get_payloads = {
        "http://fake-ollama/api/tags": {
            "models": [
                {"name": "mistral:7b"},
                {"name": "qwen2.5:7b-instruct"},
                {"name": "llama3.1:8b"},
                {"name": "nomic-embed-text:latest"},
            ]
        }
    }
    FakeAsyncClient.post_payloads = {
        "qwen2.5:7b-instruct": {
            "message": {
                "content": json.dumps(
                    {
                        "items": [
                            "Warm wishes that stay gentle, grateful, and deeply personal.",
                            "Gratitude looks brighter when it is shared with family.",
                            "Soft words can still carry strong love and quiet warmth.",
                        ]
                    }
                )
            }
        },
        "mistral:7b": {
            "message": {
                "content": json.dumps(
                    {
                        "items": [
                            "A calm message can still feel rich, modern, and heartfelt.",
                            "Send kind energy with a lighter, cleaner emotional touch.",
                            "Keep the warmth high and the wording simple and clear.",
                        ]
                    }
                )
            }
        },
        "llama3.1:8b": {
            "message": {
                "content": json.dumps(
                    {
                        "items": [
                            "Steady hope lands better when the line feels honest and direct.",
                            "A focused wish can still sound warm without sounding dramatic.",
                            "Give the message room to breathe and it feels more sincere.",
                        ]
                    }
                )
            }
        },
        "llama-3.3-70b-versatile": {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "items": [
                                    "Groq can return crisp content when the prompt stays strict.",
                                    "Short celebratory lines work best when they stay grounded.",
                                    "Clear structure helps the model stay inside the requested tone.",
                                ]
                            }
                        )
                    }
                }
            ]
        },
    }


def sample_payload(**overrides) -> dict:
    """Return a valid default request body for POST /generate/single."""

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
        "trace_id": "trace-123",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_health_reports_busy_and_ollama_reachability(monkeypatch) -> None:
    """The health endpoint should return static service info and Ollama reachability."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "ok": True,
        "service": "llm-comparator",
        "version": "test-version",
        "busy": False,
        "ollama_reachable": True,
    }


@pytest.mark.asyncio
async def test_models_endpoint_splits_chat_and_embedding_models(monkeypatch) -> None:
    """The models endpoint should split discovered Ollama models by capability."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["ollama"]["chat_models"] == ["mistral:7b", "qwen2.5:7b-instruct", "llama3.1:8b"]
    assert payload["ollama"]["embedding_models"] == ["nomic-embed-text:latest"]


@pytest.mark.asyncio
async def test_generate_single_ollama_returns_items_and_meta(monkeypatch) -> None:
    """The single generation endpoint should proxy one Ollama chat call and normalize the result."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/generate/single", json=sample_payload(seed=7))

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["backend"] == "ollama"
    assert payload["model"] == "qwen2.5:7b-instruct"
    assert len(payload["items"]) == 3
    assert payload["meta"]["trace_id"] == "trace-123"
    assert payload["meta"]["busy"] is False
    assert payload["meta"]["latency_ms"] >= 0
    assert payload["meta"]["request_id"]

    upstream_call = next(call for call in FakeAsyncClient.calls if call["method"] == "POST")
    assert upstream_call["url"] == "http://fake-ollama/api/chat"
    assert upstream_call["json"]["options"]["num_predict"] == 300
    assert upstream_call["json"]["options"]["temperature"] == 0.8
    assert upstream_call["json"]["options"]["seed"] == 7


@pytest.mark.asyncio
async def test_generate_single_rejects_embedding_model(monkeypatch) -> None:
    """Embedding-only Ollama models must be rejected for chat generation."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(model="nomic-embed-text:latest"),
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Embedding model cannot be used for chat generation"}
    assert not any(call["method"] == "POST" for call in FakeAsyncClient.calls)


@pytest.mark.asyncio
async def test_generate_single_supports_groq(monkeypatch) -> None:
    """Groq support should remain available for configured callers."""

    main_module, llm_module = reload_app(monkeypatch, groq_api_key="groq-test-key")
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(backend="groq", model="llama-3.3-70b-versatile"),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["backend"] == "groq"
    assert payload["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_busy_request_returns_429_when_slot_is_in_use(monkeypatch) -> None:
    """A second request should get a busy response when the only slot is occupied."""

    main_module, llm_module = reload_app(monkeypatch, max_concurrent_jobs="1", max_queue="0")
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.started_event = asyncio.Event()
    FakeAsyncClient.release_event = asyncio.Event()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_request = asyncio.create_task(client.post("/generate/single", json=sample_payload()))
        await asyncio.wait_for(FakeAsyncClient.started_event.wait(), timeout=1.0)

        second_response = await client.post(
            "/generate/single",
            json=sample_payload(trace_id="trace-456"),
        )

        assert second_response.status_code == 429
        assert second_response.headers["Retry-After"] == "2"
        assert second_response.json() == {
            "ok": False,
            "error": "busy",
            "retry_after_ms": 2000,
            "meta": {"busy": True},
        }

        FakeAsyncClient.release_event.set()
        first_response = await first_request

    assert first_response.status_code == 200
