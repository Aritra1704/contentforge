"""HTTP tests for observability, structured errors, and generation flows."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from pathlib import Path
import sys

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeResponse:
    """Minimal httpx-style response wrapper for mocked upstream calls."""

    def __init__(
        self,
        payload: dict,
        *,
        url: str,
        method: str = "GET",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.url = url
        self.method = method
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return

        request = httpx.Request(self.method, self.url)
        response = httpx.Response(
            self.status_code,
            headers=self.headers,
            request=request,
            json=self._payload,
        )
        raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)


class FakeAsyncClient:
    """Configurable async client used to mock Ollama and Groq calls."""

    get_payloads: dict[str, FakeResponse | dict | Exception] = {}
    post_payloads: dict[str, FakeResponse | dict | Exception | list[FakeResponse | dict | Exception]] = {}
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
        if isinstance(payload, FakeResponse):
            return payload
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
        if isinstance(payload, list):
            if not payload:
                raise RuntimeError(f"No fake POST payload configured for {key}")
            payload = payload.pop(0)
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            raise RuntimeError(f"No fake POST payload configured for {key}")
        if isinstance(payload, FakeResponse):
            return payload
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
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("OLLAMA_URL", "http://fake-ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODELS", "mistral:7b,qwen2.5:7b-instruct,llama3.1:8b")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODELS", "nomic-embed-text:latest")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", max_concurrent_jobs)
    monkeypatch.setenv("MAX_QUEUE", max_queue)
    monkeypatch.setenv("BUSY_RETRY_AFTER_MS", "2000")
    monkeypatch.setenv("REQUEST_TIMEOUT_SEC", "120")

    if groq_api_key is None:
        monkeypatch.setenv("GROQ_API_KEY", "")
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


def compare_payload(**overrides) -> dict:
    """Return a valid default request body for compare-models."""

    payload = {
        "theme_name": "Warm Wishes",
        "tone_funny_pct": 20,
        "tone_emotion_pct": 70,
        "prompt_keywords": ["family", "gratitude"],
        "visual_style": "soft watercolor",
        "targets": [
            {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
            {"backend": "groq", "model": "llama-3.3-70b-versatile"},
        ],
        "count": 3,
        "max_tokens": 300,
        "temperature": 0.8,
        "trace_id": "trace-compare",
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_generate_single_returns_request_id_header_and_meta(monkeypatch) -> None:
    """Successful generation should expose the same request ID in header and body."""

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
    assert payload["meta"]["trace_id"] == "trace-123"
    assert payload["meta"]["request_id"] == response.headers["X-Request-Id"]
    assert payload["meta"]["busy"] is False
    assert payload["meta"]["latency_ms"] >= 0
    assert payload["meta"]["applied_settings"] == {
        "max_words": 16,
        "emoji_policy": "none",
        "tone_style": "conversational",
        "avoid_cliches": True,
    }
    assert payload["errors"] is None


@pytest.mark.asyncio
async def test_generate_single_returns_exact_count(monkeypatch) -> None:
    """The endpoint should always return exactly `count` phrases when extras are present."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "1. Phrase one with clear intent",
                    "2. Phrase two with clear intent",
                    "3. Phrase three with clear intent",
                    "4. Phrase four with clear intent",
                    "5. Phrase five with clear intent",
                ]
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/generate/single", json=sample_payload(count=3))

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 3


@pytest.mark.asyncio
async def test_generate_single_enforces_no_emoji_policy(monkeypatch) -> None:
    """When emoji policy is none, retry once and return emoji-free items."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = [
        {
            "message": {
                "content": "\n".join(
                    [
                        "1. Power through Monday 💪 with a bold smile",
                        "2. Keep your focus sharp 🔥 and steady today",
                        "3. Own your schedule and protect your energy ✨",
                    ]
                )
            }
        },
        {
            "message": {
                "content": "\n".join(
                    [
                        "1. Power through Monday with a bold smile",
                        "2. Keep your focus sharp and steady today",
                        "3. Own your schedule and protect your energy",
                    ]
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(emoji_policy="none"),
        )

    assert response.status_code == 200
    payload = response.json()
    assert all("💪" not in item and "🔥" not in item and "✨" not in item for item in payload["items"])
    ollama_calls = [call for call in FakeAsyncClient.calls if call["url"] == "http://fake-ollama/api/chat"]
    assert len(ollama_calls) == 2


@pytest.mark.asyncio
async def test_generate_single_enforces_max_words(monkeypatch) -> None:
    """When max_words is strict, retry once and return phrases within the limit."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = [
        {
            "message": {
                "content": "\n".join(
                    [
                        "1. This line is intentionally too long for a strict word cap",
                        "2. Another phrase that breaks the requested compact word limit",
                        "3. Third line also exceeds the maximum words required today",
                    ]
                )
            }
        },
        {
            "message": {
                "content": "\n".join(
                    [
                        "1. Keep focus and move calmly",
                        "2. Protect energy and stay clear",
                        "3. Start strong and stay steady",
                    ]
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(max_words=5),
        )

    assert response.status_code == 200
    payload = response.json()
    assert all(len(item.split()) <= 5 for item in payload["items"])
    ollama_calls = [call for call in FakeAsyncClient.calls if call["url"] == "http://fake-ollama/api/chat"]
    assert len(ollama_calls) == 2


@pytest.mark.asyncio
async def test_generate_single_salvages_broken_qwen_json_without_leaking_blob(monkeypatch) -> None:
    """Malformed JSON-like model output should yield phrases, not raw JSON container text."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": '\n'.join(
                [
                    'Embrace Monday with a strength boost, like hitting snooze but for motivation!',
                    '{"items": ["Embrace Monday with a strength boost, like hitting snooze but for motivation!"], ["Start] your week with a dose of energy, not an alarm clock. Feel the warmth of a new day!","Monday\'s here, but your spirit\'s strong. Find your energy, not your pillow."]}',
                ]
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/generate/single", json=sample_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["items"]) == 3
    assert all(not item.strip().startswith('{"items"') for item in payload["items"])
    assert any(
        item.startswith("Start] your week with a dose of energy, not an alarm clock.")
        for item in payload["items"]
    )
    assert "Monday's here, but your spirit's strong. Find your energy, not your pillow." in payload["items"]


@pytest.mark.asyncio
async def test_validation_error_is_structured(monkeypatch) -> None:
    """Embedding-only Ollama models must be rejected with structured validation errors."""

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
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "validation_error"
    assert payload["error"]["message"] == "Embedding model cannot be used for chat generation"
    assert payload["error"]["backend"] == "ollama"
    assert payload["error"]["model"] == "nomic-embed-text:latest"
    assert payload["meta"]["request_id"] == response.headers["X-Request-Id"]
    assert payload["meta"]["trace_id"] == "trace-123"
    assert not any(call["url"] == "http://fake-ollama/api/chat" for call in FakeAsyncClient.calls)


@pytest.mark.asyncio
async def test_missing_groq_configuration_returns_structured_error_and_logs(
    monkeypatch,
    caplog,
) -> None:
    """Missing Groq config should produce not_configured with traceback-bearing logs."""

    main_module, llm_module = reload_app(monkeypatch, groq_api_key=None)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)
    caplog.set_level(logging.INFO)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(backend="groq", model="llama-3.3-70b-versatile", trace_id="trace-groq"),
        )

    assert response.status_code == 503
    payload = response.json()
    assert payload["ok"] is False
    assert payload["error"]["error_type"] == "not_configured"
    assert payload["error"]["backend"] == "groq"
    assert payload["error"]["model"] == "llama-3.3-70b-versatile"
    assert payload["meta"]["request_id"] == response.headers["X-Request-Id"]
    assert payload["meta"]["trace_id"] == "trace-groq"

    error_record = next(record for record in caplog.records if "request_failed" in record.message)
    assert error_record.levelno == logging.ERROR
    assert error_record.exc_info is not None
    assert 'request_id="' in error_record.message
    assert 'trace_id="trace-groq"' in error_record.message
    assert 'backend="groq"' in error_record.message
    assert 'model="llama-3.3-70b-versatile"' in error_record.message


@pytest.mark.asyncio
async def test_groq_rate_limit_maps_retry_after(monkeypatch) -> None:
    """Groq 429 responses should map to structured rate-limited errors."""

    main_module, llm_module = reload_app(monkeypatch, groq_api_key="groq-test-key")
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["llama-3.3-70b-versatile"] = FakeResponse(
        {"error": {"message": "Too many requests"}},
        url="https://api.groq.com/openai/v1/chat/completions",
        method="POST",
        status_code=429,
        headers={"Retry-After": "3"},
    )
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(backend="groq", model="llama-3.3-70b-versatile"),
        )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3"
    payload = response.json()
    assert payload["error"]["error_type"] == "rate_limited"
    assert payload["error"]["http_status"] == 429
    assert payload["error"]["retry_after_ms"] == 3000
    assert payload["error"]["details"]["body_snippet"] == '{"error": {"message": "Too many requests"}}'


@pytest.mark.asyncio
async def test_busy_request_returns_structured_busy_and_logs_info(monkeypatch, caplog) -> None:
    """A second request should get busy details and log at INFO, not ERROR."""

    main_module, llm_module = reload_app(monkeypatch, max_concurrent_jobs="1", max_queue="0")
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.started_event = asyncio.Event()
    FakeAsyncClient.release_event = asyncio.Event()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)
    caplog.set_level(logging.INFO)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        first_request = asyncio.create_task(client.post("/generate/single", json=sample_payload()))
        await asyncio.wait_for(FakeAsyncClient.started_event.wait(), timeout=1.0)

        second_response = await client.post(
            "/generate/single",
            json=sample_payload(trace_id="trace-busy"),
        )

        FakeAsyncClient.release_event.set()
        first_response = await first_request

    assert first_response.status_code == 200
    assert second_response.status_code == 429
    assert second_response.headers["Retry-After"] == "2"
    payload = second_response.json()
    assert payload == {
        "ok": False,
        "error": {
            "error_type": "busy",
            "message": "The service is busy. Retry later.",
            "backend": "ollama",
            "model": "qwen2.5:7b-instruct",
            "http_status": 429,
            "retry_after_ms": 2000,
        },
        "meta": {
            "request_id": second_response.headers["X-Request-Id"],
            "trace_id": "trace-busy",
            "busy": True,
        },
    }

    busy_record = next(record for record in caplog.records if "request_busy" in record.message)
    assert busy_record.levelno == logging.INFO
    assert busy_record.exc_info is None
    assert not any(record.levelno >= logging.ERROR and "request_busy" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_compare_models_alias_returns_per_model_errors(monkeypatch) -> None:
    """Compare-models should keep per-model failures in-band instead of collapsing to a string."""

    main_module, llm_module = reload_app(monkeypatch, groq_api_key=None)
    FakeAsyncClient.reset()
    configure_default_payloads()
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/generation/compare-models", json=compare_payload())

    assert response.status_code == 200
    assert response.headers["X-Request-Id"]
    payload = response.json()
    assert payload["ok"] is False
    assert payload["meta"]["request_id"] == response.headers["X-Request-Id"]
    assert payload["meta"]["trace_id"] == "trace-compare"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["ok"] is True
    assert payload["results"][0]["items"]
    assert payload["results"][1]["ok"] is False
    assert payload["results"][1]["error"]["error_type"] == "not_configured"
    assert payload["results"][1]["error"]["backend"] == "groq"
    assert payload["results"][1]["error"]["model"] == "llama-3.3-70b-versatile"
