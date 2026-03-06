"""Output parsing shape tests per OutputSpec format."""

from __future__ import annotations

import pytest

from app.llm import parse_items
from app.schemas import GenerateSingleRequest


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
    ("format_name", "content"),
    [
        ("one_liner", "1. Keep hope alive\n2. Stay warm and kind\n3. Share gratitude daily"),
        (
            "paragraph",
            "Today feels calm and bright. Family moments feel meaningful. Gratitude keeps the tone grounded. We carry that warmth forward.",
        ),
        (
            "one_page",
            "First short paragraph with clear tone.\nStill concise and simple.\n\nSecond short paragraph with steady warmth.\nEnds with a practical note.",
        ),
        ("verse", "Soft dawn arrives\nKind words stay\nCalm hearts rise\nLight returns"),
        (
            "story",
            "Setup:\nA quiet room held patient hope.\nTurn:\nA hard choice changed the day.\nResolution:\nThey chose kindness and moved ahead.",
        ),
    ],
)
def test_parse_items_non_pros_cons_stores_raw_text(format_name: str, content: str) -> None:
    """Non-pros/cons formats should produce raw text and no structured output."""

    request = GenerateSingleRequest.model_validate(base_payload(output_spec={"format": format_name}))
    parsed = parse_items(content, payload=request)

    assert parsed.raw_text is not None
    assert parsed.structured_output is None
    assert parsed.items


def test_parse_items_pros_cons_returns_structured_output() -> None:
    """Pros/cons format should parse sections into structured JSON."""

    request = GenerateSingleRequest.model_validate(
        base_payload(
            output_spec={
                "format": "pros_cons",
                "structure": {"items": 2},
            }
        )
    )
    parsed = parse_items(
        "Pros:\n- Fast setup\n- Lower cost\nCons:\n- Limited customization\n- More manual checks",
        payload=request,
    )

    assert parsed.raw_text is None
    assert parsed.structured_output is not None
    assert parsed.structured_output.pros == ["Fast setup", "Lower cost"]
    assert parsed.structured_output.cons == ["Limited customization", "More manual checks"]
