"""Unit tests for OutputSpec normalization and legacy compatibility."""

from __future__ import annotations

import pytest

from app.schemas import GenerateSingleRequest, normalize_output_spec


def base_single_payload(**overrides) -> dict:
    """Return a valid GenerateSingleRequest body with optional overrides."""

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


def test_generate_single_request_maps_legacy_fields_to_output_spec() -> None:
    """Legacy request fields should normalize into the unified OutputSpec."""

    payload = GenerateSingleRequest.model_validate(
        base_single_payload(
            count=5,
            min_words=6,
            max_words=14,
            output_format="lines",
        )
    )

    assert payload.output_spec is not None
    assert payload.output_spec.format == "one_liner"
    assert payload.output_spec.structure.items == 5
    assert payload.output_spec.structure.no_numbering is True
    assert payload.output_spec.length.target_words == 10
    assert payload.output_spec.length.min_words == 6
    assert payload.output_spec.length.max_words == 14


@pytest.mark.parametrize(
    ("format_name", "expected"),
    [
        (
            "paragraph",
            {"min_words": 60, "max_words": 110, "target_words": 80, "no_lists": True},
        ),
        (
            "one_page",
            {"min_words": 180, "max_words": 320, "target_words": 250, "no_lists": True},
        ),
        (
            "pros_cons",
            {"items": 4, "sections": ["Pros", "Cons"]},
        ),
        (
            "verse",
            {"items": 8, "max_lines": 12, "max_words_per_line": 8},
        ),
        (
            "story",
            {"min_words": 350, "max_words": 600, "target_words": 450, "sections": ["Act I", "Act II", "Act III"]},
        ),
    ],
)
def test_normalize_output_spec_applies_format_defaults(format_name: str, expected: dict) -> None:
    """Each format should receive its documented default constraints."""

    spec = normalize_output_spec({"output_spec": {"format": format_name}})
    assert spec.format == format_name

    if "min_words" in expected:
        assert spec.length.min_words == expected["min_words"]
    if "max_words" in expected:
        assert spec.length.max_words == expected["max_words"]
    if "target_words" in expected:
        assert spec.length.target_words == expected["target_words"]
    if "items" in expected:
        assert spec.structure.items == expected["items"]
    if "sections" in expected:
        assert spec.structure.sections == expected["sections"]
    if "max_lines" in expected:
        assert spec.structure.max_lines == expected["max_lines"]
    if "max_words_per_line" in expected:
        assert spec.structure.max_words_per_line == expected["max_words_per_line"]
    if "no_lists" in expected:
        assert spec.structure.no_lists is expected["no_lists"]


def test_explicit_output_spec_not_overridden_by_legacy_defaults() -> None:
    """New OutputSpec input should drive normalization when present."""

    spec = normalize_output_spec(
        {
            "count": 9,
            "max_words": 12,
            "output_format": "numbered",
            "output_spec": {
                "format": "paragraph",
                "length": {"target_words": 90},
            },
        }
    )

    assert spec.format == "paragraph"
    assert spec.length.target_words == 90
    assert spec.length.min_words == 60
    assert spec.length.max_words == 110
    assert spec.structure.items is None


def test_cultural_context_alias_normalization() -> None:
    """Hyphenated cultural context aliases should normalize to enum values."""

    payload = GenerateSingleRequest.model_validate(
        base_single_payload(cultural_context="south-indian")
    )
    assert payload.cultural_context == "south_indian"
