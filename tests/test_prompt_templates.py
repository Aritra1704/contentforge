"""Prompt template tests for format-first OutputSpec construction."""

from __future__ import annotations

import pytest

from app.schemas import GenerateSingleRequest
from src.prompts.phrase_prompt import build_messages, selected_voice_pack


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


def test_build_messages_uses_creative_brief_voice_pack_and_guardrails() -> None:
    """Creative brief fields should flow into the rendered prompt."""

    request = GenerateSingleRequest.model_validate(
        base_payload(
            app_id="ecard_factory",
            content_type="ecard_message",
            creative_brief={
                "voice_pack": "festival_respectful",
                "audience_age_band": "adults",
                "cultural_guardrails": ["avoid parody", "keep references respectful"],
                "taboo_phrases": ["party hard"],
                "target_structure": "single sincere greeting",
                "desired_emotional_effect": "quiet warmth",
            },
        )
    )
    messages = build_messages(request)
    user_prompt = messages[1]["content"]

    assert "App ID: ecard_factory" in user_prompt
    assert "Content type: ecard_message" in user_prompt
    assert "Voice pack: festival_respectful." in user_prompt
    assert "Avoid slang, parody, or flippant humor." in user_prompt
    assert "Cultural guardrails: avoid parody, keep references respectful." in user_prompt
    assert "Desired emotional effect: quiet warmth." in user_prompt
    assert "Also avoid these taboo phrases: party hard." in user_prompt


def test_selected_voice_pack_auto_detects_romantic_and_playful_briefs() -> None:
    """Auto voice-pack selection should route common eCard themes to the intended pack."""

    valentine_request = GenerateSingleRequest.model_validate(
        base_payload(
            theme_name="Valentine Special",
            prompt_keywords=["love", "date night"],
            tone_funny_pct=45,
            creative_brief={"voice_pack": "auto"},
        )
    )
    netflix_request = GenerateSingleRequest.model_validate(
        base_payload(
            theme_name="Netflix and Chill",
            prompt_keywords=["sofa", "movie night"],
            tone_funny_pct=70,
            creative_brief={"voice_pack": "auto"},
        )
    )

    assert selected_voice_pack(valentine_request) == "romantic_witty"
    assert selected_voice_pack(netflix_request) == "playful_modern"
