"""Prompt template tests for format-first OutputSpec construction."""

from __future__ import annotations

import pytest

from app.schemas import GenerateSingleRequest
from src.prompts.phrase_prompt import build_messages


def base_payload(**overrides) -> dict:
    """Return a valid request payload with optional overrides."""

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
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("format_name", "expected_fragment"),
    [
        ("one_liner", "Return exactly 3 lines."),
        ("paragraph", "Return a single paragraph."),
        ("one_page", "Return 2 to 4 short paragraphs."),
        ("pros_cons", 'Return exactly two sections: "Pros:" and "Cons:".'),
        ("verse", "No title unless explicitly requested."),
        ("story", "Section headers must be: Setup, Turn, Resolution."),
    ],
)
def test_build_messages_uses_format_specific_template(format_name: str, expected_fragment: str) -> None:
    """Prompt content should include the dedicated template for each format."""

    request = GenerateSingleRequest.model_validate(base_payload(output_spec={"format": format_name}))
    messages = build_messages(request)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Never prefix output with 'Sure' or 'Here's'." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert expected_fragment in messages[1]["content"]
    assert "Must not return JSON." in messages[1]["content"]


def test_build_messages_includes_cultural_context_guidance() -> None:
    """Prompt should include safe cultural guidance and non-stereotype constraints."""

    request = GenerateSingleRequest.model_validate(
        base_payload(cultural_context="bengali")
    )
    messages = build_messages(request)
    user_prompt = messages[1]["content"]

    assert "Cultural context: bengali" in user_prompt
    assert "adda/chai/rain/Kolkata-friendly imagery" in user_prompt
    assert "Do not stereotype or force cultural markers." in user_prompt
    assert "Use cultural context only when relevant to the request." in user_prompt
