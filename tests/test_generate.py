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
    judge_enabled: str = "false",
    judge_mode: str = "tie_break",
    judge_provider: str = "ollama",
    judge_model: str = "qwen2.5:7b-instruct",
    openai_judge_enabled: str = "false",
    openai_api_key: str = "",
    openai_judge_model: str = "gpt-4o-mini",
    judge_tie_threshold: str = "7",
    judge_timeout_sec: str = "120",
    judge_connect_timeout_sec: str = "10",
    judge_fallback_to_baseline: str = "true",
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
    monkeypatch.setenv("JUDGE_ENABLED", judge_enabled)
    monkeypatch.setenv("JUDGE_MODE", judge_mode)
    monkeypatch.setenv("JUDGE_PROVIDER", judge_provider)
    monkeypatch.setenv("JUDGE_BACKEND", judge_provider)
    monkeypatch.setenv("JUDGE_MODEL", judge_model)
    monkeypatch.setenv("OPENAI_JUDGE_ENABLED", openai_judge_enabled)
    monkeypatch.setenv("OPENAI_API_KEY", openai_api_key)
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", openai_judge_model)
    monkeypatch.setenv("JUDGE_TIE_THRESHOLD", judge_tie_threshold)
    monkeypatch.setenv("JUDGE_TIMEOUT_SEC", judge_timeout_sec)
    monkeypatch.setenv("JUDGE_CONNECT_TIMEOUT_SEC", judge_connect_timeout_sec)
    monkeypatch.setenv("JUDGE_FALLBACK_TO_BASELINE", judge_fallback_to_baseline)
    monkeypatch.setenv("QUALITY_MEMORY_ENABLED", "false")
    monkeypatch.setenv("QUALITY_MEMORY_DSN", "")

    if groq_api_key is None:
        monkeypatch.setenv("GROQ_API_KEY", "")
    else:
        monkeypatch.setenv("GROQ_API_KEY", groq_api_key)

    for module_name in list(sys.modules):
        if (
            module_name == "app"
            or module_name.startswith("app.")
            or module_name == "src"
            or module_name.startswith("src.")
        ):
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
async def test_generate_single_strips_forbidden_prefixes_without_failing(monkeypatch) -> None:
    """Hard-preface text should be cleaned during fallback, not returned as 502."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = [
        {
            "message": {
                "content": "\n".join(
                    [
                        "Sure, keep focus and move calmly today",
                        "Here's protect energy and stay clear now",
                        "Heres start strong and stay steady always",
                    ]
                )
            }
        },
        {
            "message": {
                "content": "\n".join(
                    [
                        "Sure, keep focus and move calmly today",
                        "Here's protect energy and stay clear now",
                        "Heres start strong and stay steady always",
                    ]
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/generate/single", json=sample_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["items"]) == 3
    assert all(not item.lower().startswith(("sure", "here's", "heres")) for item in payload["items"])


@pytest.mark.asyncio
async def test_generate_single_splits_single_paragraph_into_requested_lines(monkeypatch) -> None:
    """One-liner requests should salvage line items from sentence-style paragraph output."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = [
        {
            "message": {
                "content": (
                    "Keep your tone steady and practical. Share gratitude clearly with family. "
                    "Start the week with calm confidence."
                )
            }
        },
        {
            "message": {
                "content": (
                    "Keep your tone steady and practical. Share gratitude clearly with family. "
                    "Start the week with calm confidence."
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(
                output_spec={"format": "one_liner", "structure": {"items": 3, "no_numbering": True}}
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["items"]) == 3


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


@pytest.mark.asyncio
async def test_generate_single_returns_raw_text_for_paragraph_format(monkeypatch) -> None:
    """Paragraph format should return raw_text and no structured_output."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": (
                "A gentle start helps set the day with steady attention and practical care. "
                "Calm words hold focus when schedules tighten and expectations begin to stack up. "
                "Small acts of gratitude keep relationships steady while people handle ordinary pressure. "
                "Clear intent makes the message land because the language remains concrete, warm, and direct. "
                "When details stay practical, readers trust the voice and remember the point after the first read."
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(output_spec={"format": "paragraph"}),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["raw_text"]
    assert payload["structured_output"] is None


@pytest.mark.asyncio
async def test_generate_single_accepts_length_only_miss_for_paragraph(monkeypatch) -> None:
    """Length-only drift should not fail the request when structure is otherwise valid."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = [
        {
            "message": {
                "content": (
                    "Warm notes help teams reset after busy mornings. "
                    "Shared gratitude keeps conversations practical and kind. "
                    "Small routines lower stress and improve focus. "
                    "Simple language makes support feel real."
                )
            }
        },
        {
            "message": {
                "content": (
                    "Warm notes help teams reset after busy mornings. "
                    "Shared gratitude keeps conversations practical and kind. "
                    "Small routines lower stress and improve focus. "
                    "Simple language makes support feel real."
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(output_spec={"format": "paragraph"}),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["raw_text"] is not None


@pytest.mark.asyncio
async def test_generate_single_returns_structured_output_for_pros_cons(monkeypatch) -> None:
    """Pros/cons format should return parsed structured_output sections."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Pros:",
                    "- Fast onboarding",
                    "- Lower cost",
                    "Cons:",
                    "- Less customization",
                    "- Requires manual review",
                ]
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/single",
            json=sample_payload(
                output_spec={
                    "format": "pros_cons",
                    "structure": {"items": 2},
                }
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["raw_text"] is None
    assert payload["structured_output"] == {
        "pros": ["Fast onboarding", "Lower cost"],
        "cons": ["Less customization", "Requires manual review"],
    }


@pytest.mark.asyncio
async def test_compare_models_winner_is_quality_first_not_latency(monkeypatch) -> None:
    """Compare winner should be chosen by quality score, not faster response time."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Kind planning keeps the morning calm and focused.",
                    "Shared gratitude helps teams move with trust.",
                    "Small wins today create steady confidence tomorrow.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["mistral:7b"] = {
        "message": {
            "content": "\n".join(
                [
                    "Wishing you steady progress this week.",
                    "Wishing you steady progress today.",
                    "Wishing you steady progress always.",
                ]
            )
        }
    }

    class DelayedQwenClient(FakeAsyncClient):
        async def post(self, url: str, *, headers=None, json=None):  # type: ignore[override]
            model_name = json.get("model") if isinstance(json, dict) else None
            if model_name == "qwen2.5:7b-instruct":
                await asyncio.sleep(0.05)
            return await super().post(url, headers=headers, json=json)

    monkeypatch.setattr(llm_module, "AsyncClient", DelayedQwenClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                targets=[
                    {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                    {"backend": "ollama", "model": "mistral:7b"},
                ],
                output_spec={
                    "format": "one_liner",
                    "structure": {"items": 3, "no_numbering": True},
                },
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner"] is not None
    assert payload["winner"]["model"] == "qwen2.5:7b-instruct"

    results_by_model = {item["model"]: item for item in payload["results"]}
    assert results_by_model["qwen2.5:7b-instruct"]["latency_ms"] >= 50
    assert (
        results_by_model["qwen2.5:7b-instruct"]["quality"]["total"]
        > results_by_model["mistral:7b"]["quality"]["total"]
    )


@pytest.mark.asyncio
async def test_compare_models_bland_groq_loses_to_more_human_output(monkeypatch) -> None:
    """Bland generic Groq output should lose against stronger human-sounding content."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Your steady care turns hard days into shared strength.",
                    "Family gratitude lands deeper when the words sound honest.",
                    "Tonight we hold each other close and keep moving with hope.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["llama-3.3-70b-versatile"] = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "items": [
                                "Wishing you joy and love on your special day.",
                                "On your special day, may your heart be filled with joy and love.",
                                "Wishing you warmth, happiness, and beautiful memories today.",
                            ]
                        }
                    )
                }
            }
        ]
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                output_spec={"format": "one_liner", "structure": {"items": 3, "no_numbering": True}}
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner"] is not None
    assert payload["winner"]["model"] == "qwen2.5:7b-instruct"
    assert payload["winner_source"] == "baseline"
    assert payload["why_winner"]

    results_by_model = {item["model"]: item for item in payload["results"]}
    assert results_by_model["llama-3.3-70b-versatile"]["quality"]["bland_generic_penalty"] > 0


@pytest.mark.asyncio
async def test_compare_models_incomplete_paragraph_cannot_win(monkeypatch) -> None:
    """Incomplete paragraph output should be ineligible and should not win."""

    main_module, llm_module = reload_app(monkeypatch)
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": (
                "A thoughtful note can calm a tense morning and make people feel seen. "
                "Gratitude works best when the language is specific and grounded in real moments. "
                "Clear phrasing keeps the tone warm without sounding generic. "
                "That balance helps the message feel complete and ready to send."
            )
        }
    }
    FakeAsyncClient.post_payloads["llama-3.3-70b-versatile"] = {
        "choices": [
            {
                "message": {
                    "content": (
                        "A thoughtful note can calm a tense morning and make people feel seen. "
                        "Gratitude works best when the language is specific and grounded in real moments. "
                        "Clear phrasing keeps the tone warm and"
                    )
                }
            }
        ]
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(output_spec={"format": "paragraph"}),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner"] is not None
    assert payload["winner"]["model"] == "qwen2.5:7b-instruct"

    results_by_model = {item["model"]: item for item in payload["results"]}
    groq_quality = results_by_model["llama-3.3-70b-versatile"]["quality"]
    assert groq_quality["incomplete_ending_penalty"] > 0
    assert any(reason.startswith("Hard penalty:") for reason in groq_quality["reasons"])


@pytest.mark.asyncio
async def test_compare_models_uses_judge_winner_when_enabled(monkeypatch) -> None:
    """When judge is enabled in always mode, final winner should follow judge_result winner."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_enabled="true",
        judge_mode="always",
        judge_provider="ollama",
        judge_model="judge-ranker",
    )
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Kind planning keeps the morning calm and focused.",
                    "Shared gratitude helps teams move with trust.",
                    "Small wins today create steady confidence tomorrow.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["mistral:7b"] = {
        "message": {
            "content": "\n".join(
                [
                    "Wishing you steady progress this week.",
                    "Wishing you steady progress today.",
                    "Wishing you steady progress always.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["judge-ranker"] = {
        "message": {
            "content": json.dumps(
                {
                    "winner_key": "modelB",
                    "ranking": ["modelB", "modelA"],
                    "scores": {
                        "modelB": {
                            "task_fit": 24,
                            "originality": 18,
                            "emotional_authenticity": 17,
                            "completeness": 14,
                            "clarity_and_flow": 9,
                            "policy_cleanliness": 10,
                            "total": 92,
                            "reason": "More original and clearer.",
                            "issues": [],
                        },
                        "modelA": {
                            "task_fit": 17,
                            "originality": 10,
                            "emotional_authenticity": 12,
                            "completeness": 10,
                            "clarity_and_flow": 8,
                            "policy_cleanliness": 10,
                            "total": 74,
                            "reason": "Less original.",
                            "issues": ["cliche"],
                        },
                    },
                }
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                targets=[
                    {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                    {"backend": "ollama", "model": "mistral:7b"},
                ],
                output_spec={
                    "format": "one_liner",
                    "structure": {"items": 3, "no_numbering": True},
                },
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner_source"] == "judge"
    assert payload["judge_result"] is not None
    assert payload["judge_result"]["winner_key"] == "modelB"
    assert payload["judge_json"] is not None
    assert payload["judge_json"]["winner_key"] == "modelB"
    assert payload["judge_reason"] == "More original and clearer."
    assert payload["winner"] is not None
    assert payload["winner"]["model"] == "mistral:7b"

    results_by_model = {item["model"]: item for item in payload["results"]}
    assert (
        results_by_model["qwen2.5:7b-instruct"]["quality"]["total"]
        > results_by_model["mistral:7b"]["quality"]["total"]
    )

    judge_calls = [
        call
        for call in FakeAsyncClient.calls
        if call["url"] == "http://fake-ollama/api/chat"
        and isinstance(call.get("json"), dict)
        and call["json"].get("model") == "judge-ranker"
    ]
    assert len(judge_calls) == 1
    judge_messages = judge_calls[0]["json"]["messages"]
    combined_prompt = "\n".join(str(message.get("content", "")) for message in judge_messages)
    assert "Do NOT rank by latency/speed." in combined_prompt
    assert "Do NOT use word count as a proxy for quality" in combined_prompt


def test_judge_prompt_includes_speed_and_requirements(monkeypatch) -> None:
    """Judge prompt should include no-speed rule and prompt requirement context."""

    reload_app(
        monkeypatch,
        judge_enabled="true",
        judge_mode="always",
        judge_provider="openai",
        openai_judge_enabled="true",
        openai_api_key="test-openai-key",
    )
    judge_module = importlib.import_module("app.judge")
    schemas_module = importlib.import_module("app.schemas")

    payload = schemas_module.GenerateCompareModelsRequest.model_validate(
        compare_payload(
            targets=[
                {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                {"backend": "ollama", "model": "mistral:7b"},
            ],
            cultural_context="bengali",
            output_spec={"format": "one_liner", "structure": {"items": 3}},
        )
    )
    score = schemas_module.QualityScore(
        task_fit=22,
        originality=14,
        emotional_authenticity=15,
        completeness=13,
        clarity_and_flow=8,
        policy_cleanliness=9,
        total=83,
        reasons=[],
        warnings=[],
    )
    candidate_map = {
        "modelA": schemas_module.CompareModelResult(
            ok=True,
            backend="ollama",
            model="qwen2.5:7b-instruct",
            items=["Line one", "Line two", "Line three"],
            quality=score,
        ),
        "modelB": schemas_module.CompareModelResult(
            ok=True,
            backend="ollama",
            model="mistral:7b",
            items=["Alt one", "Alt two", "Alt three"],
            quality=score,
        ),
    }

    messages = judge_module.build_judge_messages(payload, candidate_map)
    combined_prompt = "\n".join(str(message.get("content", "")) for message in messages)
    assert "Do NOT rank by latency/speed." in combined_prompt
    assert "Do NOT use word count as a proxy for quality" in combined_prompt
    assert "Rank by writing quality and task fit, not speed or superficial cleanliness." in combined_prompt
    assert "Task fit must include alignment with requested cultural_context when relevant." in combined_prompt
    assert "If one output is structurally clean but emotionally bland" in combined_prompt
    assert "Full request context JSON:" in combined_prompt
    assert "Cultural context: bengali" in combined_prompt
    assert "Output format:" in combined_prompt


@pytest.mark.asyncio
async def test_compare_models_tie_break_uses_openai_judge(monkeypatch) -> None:
    """Tie-break mode should invoke OpenAI judge when top-two scores are within threshold."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_enabled="true",
        judge_mode="tie_break",
        judge_provider="openai",
        openai_judge_enabled="true",
        openai_api_key="test-openai-key",
        judge_tie_threshold="100",
    )
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Kind planning keeps the morning calm and focused.",
                    "Shared gratitude helps teams move with trust.",
                    "Small wins today create steady confidence tomorrow.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["mistral:7b"] = {
        "message": {
            "content": "\n".join(
                [
                    "Wishing you steady progress this week.",
                    "Wishing you steady progress today.",
                    "Wishing you steady progress always.",
                ]
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    judge_module = importlib.import_module("app.judge")
    schemas_module = importlib.import_module("app.schemas")

    async def fake_openai_judge(context, candidates):
        assert context["theme_name"] == "Warm Wishes"
        assert sorted(candidates.keys()) == ["modelA", "modelB"]
        return schemas_module.JudgeResult.model_validate(
            {
                "winner_key": "modelB",
                "ranking": ["modelB", "modelA"],
                "scores": {
                    "modelA": {
                        "task_fit": 22,
                        "originality": 15,
                        "emotional_authenticity": 16,
                        "completeness": 13,
                        "clarity_and_flow": 9,
                        "policy_cleanliness": 10,
                        "total": 87,
                        "reason": "Good quality.",
                        "issues": [],
                    },
                    "modelB": {
                        "task_fit": 24,
                        "originality": 18,
                        "emotional_authenticity": 17,
                        "completeness": 14,
                        "clarity_and_flow": 9,
                        "policy_cleanliness": 10,
                        "total": 92,
                        "reason": "More original and clearer.",
                        "issues": [],
                    },
                },
            }
        )

    monkeypatch.setattr(judge_module, "openai_judge_candidates", fake_openai_judge)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                targets=[
                    {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                    {"backend": "ollama", "model": "mistral:7b"},
                ],
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner_source"] == "judge"
    assert payload["judge_result"] is not None
    assert payload["judge_result"]["winner_key"] == "modelB"
    assert payload["winner"]["model"] == "mistral:7b"


@pytest.mark.asyncio
async def test_compare_models_missing_openai_key_falls_back_to_baseline(monkeypatch) -> None:
    """Missing OPENAI_API_KEY should keep baseline winner and avoid crashes."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_enabled="true",
        judge_mode="always",
        judge_provider="openai",
        openai_judge_enabled="true",
        openai_api_key="",
    )
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Kind planning keeps the morning calm and focused.",
                    "Shared gratitude helps teams move with trust.",
                    "Small wins today create steady confidence tomorrow.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["mistral:7b"] = {
        "message": {
            "content": "\n".join(
                [
                    "Wishing you steady progress this week.",
                    "Wishing you steady progress today.",
                    "Wishing you steady progress always.",
                ]
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    judge_module = importlib.import_module("app.judge")

    async def should_not_run(*args, **kwargs):
        raise AssertionError("OpenAI judge should not be called when API key is missing.")

    monkeypatch.setattr(judge_module, "openai_judge_candidates", should_not_run)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                targets=[
                    {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                    {"backend": "ollama", "model": "mistral:7b"},
                ],
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner_source"] == "baseline"
    assert payload["judge_result"] is None
    assert "OPENAI_API_KEY" in (payload.get("judge_reason") or "")
    assert payload["winner"] is not None
    assert payload["winner"]["model"] == "qwen2.5:7b-instruct"


@pytest.mark.asyncio
async def test_compare_models_openai_disabled_uses_default_ollama_judge(monkeypatch) -> None:
    """OpenAI judge must not run unless OPENAI_JUDGE_ENABLED is explicitly true."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_enabled="true",
        judge_mode="always",
        judge_provider="openai",
        openai_judge_enabled="false",
        judge_model="judge-ranker",
        openai_api_key="test-openai-key",
    )
    FakeAsyncClient.reset()
    configure_default_payloads()
    FakeAsyncClient.post_payloads["qwen2.5:7b-instruct"] = {
        "message": {
            "content": "\n".join(
                [
                    "Kind planning keeps the morning calm and focused.",
                    "Shared gratitude helps teams move with trust.",
                    "Small wins today create steady confidence tomorrow.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["mistral:7b"] = {
        "message": {
            "content": "\n".join(
                [
                    "Wishing you steady progress this week.",
                    "Wishing you steady progress today.",
                    "Wishing you steady progress always.",
                ]
            )
        }
    }
    FakeAsyncClient.post_payloads["judge-ranker"] = {
        "message": {
            "content": json.dumps(
                {
                    "winner_key": "modelB",
                    "ranking": ["modelB", "modelA"],
                    "scores": {
                        "modelA": {
                            "task_fit": 22,
                            "originality": 15,
                            "emotional_authenticity": 16,
                            "completeness": 13,
                            "clarity_and_flow": 9,
                            "policy_cleanliness": 10,
                            "total": 87,
                            "reason": "Good quality.",
                            "issues": [],
                        },
                        "modelB": {
                            "task_fit": 24,
                            "originality": 18,
                            "emotional_authenticity": 17,
                            "completeness": 14,
                            "clarity_and_flow": 9,
                            "policy_cleanliness": 10,
                            "total": 92,
                            "reason": "More original and clearer.",
                            "issues": [],
                        },
                    },
                }
            )
        }
    }
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    judge_module = importlib.import_module("app.judge")

    async def should_not_run_openai(*args, **kwargs):
        raise AssertionError("OpenAI judge should not run when OPENAI_JUDGE_ENABLED=false.")

    monkeypatch.setattr(judge_module, "openai_judge_candidates", should_not_run_openai)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/generate/compare-models",
            json=compare_payload(
                targets=[
                    {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
                    {"backend": "ollama", "model": "mistral:7b"},
                ],
            ),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["winner_source"] == "judge"
    assert payload["judge_result"] is not None
    assert payload["judge_result"]["winner_key"] == "modelB"
