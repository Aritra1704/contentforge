"""Tests for POST /judge/round-robin endpoint."""

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

from test_generate import FakeAsyncClient, reload_app


def round_robin_payload(**overrides) -> dict:
    """Return a valid default payload for /judge/round-robin."""

    payload = {
        "prompt_context": {
            "theme_name": "Warm Wishes",
            "tone_funny_pct": 20,
            "tone_emotion_pct": 75,
            "tone_style": "conversational",
            "audience": "friends",
            "cultural_context": "global",
            "output_spec": {
                "format": "paragraph",
                "length": {"min_words": 60, "max_words": 110, "target_words": 80},
                "structure": {"no_lists": True, "no_numbering": True},
            },
            "avoid_cliches": True,
        },
        "candidates": [
            {
                "model": "qwen2.5:7b-instruct",
                "backend": "ollama",
                "text": "A quiet morning settled over the room as we remembered how your kindness shows up in ordinary moments, and it made the whole day feel warmer and more hopeful.",
            },
            {
                "model": "mistral:7b",
                "backend": "ollama",
                "text": "Wishing you joy on your special day and a life filled with joy and love.",
            },
            {
                "model": "llama3.1:8b",
                "backend": "groq",
                "text": "By dusk, the conversation had turned into gratitude and laughter, and we left knowing that your presence keeps people steady in difficult seasons.",
            },
        ],
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_round_robin_ranks_by_wins_then_points_then_head_to_head(monkeypatch) -> None:
    """Ties on wins and points should be resolved by direct head-to-head winner."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
        judge_mode="always",
    )
    FakeAsyncClient.reset()
    FakeAsyncClient.post_payloads["judge-ranker"] = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_1",
                        "reason": "Candidate 1 sounds more human and less generic.",
                        "scores": {
                            "candidate_1": {
                                "prompt_fit": 18,
                                "human_feel": 18,
                                "originality": 17,
                                "emotional_authenticity": 13,
                                "completeness": 14,
                                "publishability": 10,
                                "total_points": 90,
                            },
                            "candidate_2": {
                                "prompt_fit": 15,
                                "human_feel": 12,
                                "originality": 12,
                                "emotional_authenticity": 13,
                                "completeness": 14,
                                "publishability": 10,
                                "total_points": 80,
                            },
                        },
                    }
                )
            }
        },
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_3",
                        "reason": "Candidate 3 has stronger completeness and originality.",
                        "scores": {
                            "candidate_1": {
                                "prompt_fit": 16,
                                "human_feel": 16,
                                "originality": 14,
                                "emotional_authenticity": 13,
                                "completeness": 13,
                                "publishability": 10,
                                "total_points": 82,
                            },
                            "candidate_3": {
                                "prompt_fit": 18,
                                "human_feel": 18,
                                "originality": 17,
                                "emotional_authenticity": 14,
                                "completeness": 15,
                                "publishability": 10,
                                "total_points": 92,
                            },
                        },
                    }
                )
            }
        },
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_2",
                        "reason": "Candidate 2 is clearer and more publishable in this pair.",
                        "scores": {
                            "candidate_2": {
                                "prompt_fit": 18,
                                "human_feel": 18,
                                "originality": 17,
                                "emotional_authenticity": 14,
                                "completeness": 15,
                                "publishability": 10,
                                "total_points": 92,
                            },
                            "candidate_3": {
                                "prompt_fit": 16,
                                "human_feel": 16,
                                "originality": 14,
                                "emotional_authenticity": 13,
                                "completeness": 13,
                                "publishability": 10,
                                "total_points": 82,
                            },
                        },
                    }
                )
            }
        },
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=round_robin_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["pairwise_results"]) == 3
    assert body["winner"]["candidate_key"] == "candidate_3"

    keys_in_order = [entry["candidate_key"] for entry in body["leaderboard"]]
    assert keys_in_order == ["candidate_3", "candidate_1", "candidate_2"]

    candidate_1_entry = body["leaderboard"][1]
    candidate_2_entry = body["leaderboard"][2]
    assert candidate_1_entry["wins"] == candidate_2_entry["wins"] == 1
    assert candidate_1_entry["points"] == candidate_2_entry["points"] == 172


@pytest.mark.asyncio
async def test_round_robin_prompt_requires_quality_only_comparison(monkeypatch) -> None:
    """Judge prompt should explicitly forbid speed/backend/model-size bias."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
    )
    FakeAsyncClient.reset()
    FakeAsyncClient.post_payloads["judge-ranker"] = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_1",
                        "reason": "Candidate 1 is stronger.",
                        "scores": {
                            "candidate_1": {
                                "prompt_fit": 18,
                                "human_feel": 18,
                                "originality": 18,
                                "emotional_authenticity": 14,
                                "completeness": 14,
                                "publishability": 9,
                                "total_points": 91,
                            },
                            "candidate_2": {
                                "prompt_fit": 14,
                                "human_feel": 14,
                                "originality": 14,
                                "emotional_authenticity": 11,
                                "completeness": 11,
                                "publishability": 8,
                                "total_points": 72,
                            },
                        },
                    }
                )
            }
        }
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200

    judge_calls = [
        call
        for call in FakeAsyncClient.calls
        if call["url"] == "http://fake-ollama/api/chat"
        and isinstance(call.get("json"), dict)
        and call["json"].get("model") == "judge-ranker"
    ]
    assert len(judge_calls) == 1
    messages = judge_calls[0]["json"]["messages"]
    combined_prompt = "\n".join(str(message.get("content", "")) for message in messages)

    assert "Ignore latency" in combined_prompt
    assert "Do not reward speed." in combined_prompt
    assert "Ignore latency, backend identity, and model size." in combined_prompt
    assert "Prefer content that feels more human, complete, original" in combined_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("omit_avoid_cliches", "null_avoid_cliches"),
    [
        (True, False),
        (False, True),
    ],
    ids=["missing", "null"],
)
async def test_round_robin_avoid_cliches_missing_or_null_defaults_false(
    monkeypatch,
    omit_avoid_cliches: bool,
    null_avoid_cliches: bool,
) -> None:
    """Missing/null avoid_cliches should pass validation and normalize to false."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
    )
    FakeAsyncClient.reset()
    FakeAsyncClient.post_payloads["judge-ranker"] = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_1",
                        "reason": "Candidate 1 is stronger.",
                        "scores": {
                            "candidate_1": {
                                "prompt_fit": 18,
                                "human_feel": 18,
                                "originality": 18,
                                "emotional_authenticity": 14,
                                "completeness": 14,
                                "publishability": 9,
                                "total_points": 91,
                            },
                            "candidate_2": {
                                "prompt_fit": 14,
                                "human_feel": 14,
                                "originality": 14,
                                "emotional_authenticity": 11,
                                "completeness": 11,
                                "publishability": 8,
                                "total_points": 72,
                            },
                        },
                    }
                )
            }
        }
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])
    if omit_avoid_cliches:
        payload["prompt_context"].pop("avoid_cliches", None)
    if null_avoid_cliches:
        payload["prompt_context"]["avoid_cliches"] = None

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200

    judge_calls = [
        call
        for call in FakeAsyncClient.calls
        if call["url"] == "http://fake-ollama/api/chat"
        and isinstance(call.get("json"), dict)
        and call["json"].get("model") == "judge-ranker"
    ]
    assert len(judge_calls) == 1
    messages = judge_calls[0]["json"]["messages"]
    combined_prompt = "\n".join(str(message.get("content", "")) for message in messages)
    assert "- Avoid cliches: False" in combined_prompt


@pytest.mark.asyncio
async def test_round_robin_tie_break_mode_judges_only_top_two(monkeypatch) -> None:
    """Tie-break mode should run only one pair when candidate count is greater than two."""

    main_module, llm_module = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
        judge_mode="tie_break",
    )
    FakeAsyncClient.reset()
    FakeAsyncClient.post_payloads["judge-ranker"] = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "winner_key": "candidate_1",
                        "reason": "Top baseline candidate remains stronger.",
                        "scores": {},
                    }
                )
            }
        }
    ]
    monkeypatch.setattr(llm_module, "AsyncClient", FakeAsyncClient)

    candidates = [
        *round_robin_payload()["candidates"],
        {
            "model": "phi4:latest",
            "backend": "ollama",
            "text": "A steady line that stays usable and concise for everyday sharing.",
        },
    ]
    payload = round_robin_payload(candidates=candidates)

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["pairwise_results"]) == 1
    assert len(body["leaderboard"]) == 4
    assert body["timeout_seconds_used"] == 120.0
    assert body["judge_provider"] == "ollama"
    assert body["judge_model"] == "judge-ranker"

    judge_calls = [
        call
        for call in FakeAsyncClient.calls
        if call["url"] == "http://fake-ollama/api/chat"
        and isinstance(call.get("json"), dict)
        and call["json"].get("model") == "judge-ranker"
    ]
    assert len(judge_calls) == 1


@pytest.mark.asyncio
async def test_round_robin_timeout_falls_back_to_baseline_when_enabled(monkeypatch) -> None:
    """Timeout should return baseline winner with warning when fallback is enabled."""

    main_module, _ = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
        judge_mode="always",
        judge_timeout_sec="0.01",
        judge_fallback_to_baseline="true",
    )
    round_robin_module = importlib.import_module("src.judge.round_robin")

    async def slow_pair(*args, **kwargs):
        await asyncio.sleep(0.05)
        return None  # pragma: no cover - wait_for timeout should trigger first

    monkeypatch.setattr(round_robin_module, "_judge_one_pair", slow_pair)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["warning"] == "judge timed out, baseline used"
    assert body["pairwise_results"] == []
    assert body["winner"] is not None
    assert body["timeout_seconds_used"] == 0.01
    assert body["judge_provider"] == "ollama"
    assert body["judge_model"] == "judge-ranker"


@pytest.mark.asyncio
async def test_round_robin_timeout_returns_structured_error_when_fallback_disabled(monkeypatch) -> None:
    """Timeout should return the requested structured judge_timeout error when fallback is disabled."""

    main_module, _ = reload_app(
        monkeypatch,
        judge_provider="ollama",
        judge_model="judge-ranker",
        judge_mode="always",
        judge_timeout_sec="0.01",
        judge_fallback_to_baseline="false",
    )
    round_robin_module = importlib.import_module("src.judge.round_robin")

    async def slow_pair(*args, **kwargs):
        await asyncio.sleep(0.05)
        return None  # pragma: no cover - wait_for timeout should trigger first

    monkeypatch.setattr(round_robin_module, "_judge_one_pair", slow_pair)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 504
    body = response.json()
    assert body["error_type"] == "judge_timeout"
    assert body["provider"] == "ollama"
    assert body["model"] == "judge-ranker"
    assert "Judge timed out after 0.01 seconds" in body["message"]


@pytest.mark.asyncio
async def test_round_robin_provider_failure_falls_back_to_baseline_when_enabled(monkeypatch) -> None:
    """Provider failures should fall back to baseline winner when fallback is enabled."""

    main_module, _ = reload_app(
        monkeypatch,
        judge_provider="groq",
        judge_model="judge-ranker",
        judge_mode="always",
        judge_fallback_to_baseline="true",
    )
    round_robin_module = importlib.import_module("src.judge.round_robin")
    errors_module = importlib.import_module("app.errors")

    async def failing_pair(*args, **kwargs):
        raise errors_module.NotConfiguredError(
            "Groq backend is not configured.",
            backend="groq",
            model="judge-ranker",
        )

    monkeypatch.setattr(round_robin_module, "_judge_one_pair", failing_pair)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["warning"] == "judge failed (not_configured), baseline used"
    assert body["pairwise_results"] == []
    assert body["winner"] is not None
    assert body["judge_provider"] == "groq"
    assert body["judge_model"] == "judge-ranker"


@pytest.mark.asyncio
async def test_round_robin_provider_failure_returns_structured_error_when_fallback_disabled(monkeypatch) -> None:
    """Provider failures should return structured judge error when fallback is disabled."""

    main_module, _ = reload_app(
        monkeypatch,
        judge_provider="groq",
        judge_model="judge-ranker",
        judge_mode="always",
        judge_fallback_to_baseline="false",
    )
    round_robin_module = importlib.import_module("src.judge.round_robin")
    errors_module = importlib.import_module("app.errors")

    async def failing_pair(*args, **kwargs):
        raise errors_module.NotConfiguredError(
            "Groq backend is not configured.",
            backend="groq",
            model="judge-ranker",
        )

    monkeypatch.setattr(round_robin_module, "_judge_one_pair", failing_pair)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])
    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["error_type"] == "not_configured"
    assert body["provider"] == "groq"
    assert body["model"] == "judge-ranker"
    assert body["message"] == "Groq backend is not configured."


@pytest.mark.asyncio
async def test_round_robin_openai_provider_uses_judge_model_env(monkeypatch) -> None:
    """OpenAI pairwise judging must use JUDGE_MODEL, not OPENAI_JUDGE_MODEL."""

    main_module, _ = reload_app(
        monkeypatch,
        judge_provider="openai",
        judge_model="rr-judge-model",
        openai_api_key="test-openai-key",
        openai_judge_model="unused-openai-model",
    )

    round_robin_module = importlib.import_module("src.judge.round_robin")
    config_module = importlib.import_module("app.config")
    seen: dict[str, object] = {}

    async def fake_openai_pairwise(messages):
        seen["model"] = config_module.settings.judge_model
        seen["messages"] = messages
        return json.dumps(
            {
                "winner_key": "candidate_2",
                "reason": "Candidate 2 is more complete.",
                "scores": {
                    "candidate_1": {
                        "prompt_fit": 15,
                        "human_feel": 15,
                        "originality": 14,
                        "emotional_authenticity": 11,
                        "completeness": 10,
                        "publishability": 8,
                        "total_points": 73,
                    },
                    "candidate_2": {
                        "prompt_fit": 18,
                        "human_feel": 18,
                        "originality": 17,
                        "emotional_authenticity": 14,
                        "completeness": 15,
                        "publishability": 9,
                        "total_points": 91,
                    },
                },
            }
        )

    monkeypatch.setattr(round_robin_module, "_call_openai_pairwise", fake_openai_pairwise)

    payload = round_robin_payload(candidates=round_robin_payload()["candidates"][:2])

    transport = httpx.ASGITransport(app=main_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/judge/round-robin", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["winner"]["candidate_key"] == "candidate_2"
    assert seen["model"] == "rr-judge-model"
    assert isinstance(seen["messages"], list)
