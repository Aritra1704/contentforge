"""Unit tests for quality-first scoring and winner selection."""

from __future__ import annotations

from app.llm import GeneratedOutput
from app.quality import pick_quality_winner, score_quality
from app.schemas import CompareModelResult, GenerateSingleRequest, QualityScore


def base_payload(**overrides) -> dict:
    """Return a valid request body for quality-scoring tests."""

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


def test_quality_score_one_liner_format_compliance_passes_exact_shape() -> None:
    """One-liner outputs with exact line count should score full format compliance."""

    request = GenerateSingleRequest.model_validate(
        base_payload(output_spec={"format": "one_liner", "structure": {"items": 3}})
    )
    output = GeneratedOutput(
        items=[
            "Kind words steady difficult mornings.",
            "Shared gratitude keeps teams grounded.",
            "Small progress compounds into confidence.",
        ],
        raw_text="Kind words steady difficult mornings.\nShared gratitude keeps teams grounded.\nSmall progress compounds into confidence.",
        structured_output=None,
    )

    quality, is_valid = score_quality(request, output)
    assert is_valid is True
    assert quality.format_compliance == 30
    assert quality.total >= 60


def test_quality_score_hard_penalty_for_wrong_structure() -> None:
    """Wrong format structure should trigger hard penalties and invalid winner eligibility."""

    request = GenerateSingleRequest.model_validate(
        base_payload(output_spec={"format": "one_liner", "structure": {"items": 3}})
    )
    output = GeneratedOutput(
        items=["Only one line returned."],
        raw_text="Only one line returned.",
        structured_output=None,
    )

    quality, is_valid = score_quality(request, output)
    assert is_valid is False
    assert quality.format_compliance == 0
    assert any(reason.startswith("Hard penalty:") for reason in quality.reasons)


def test_pick_quality_winner_ignores_latency() -> None:
    """Winner selection must prefer higher total quality even when slower."""

    high_quality = QualityScore(
        format_compliance=30,
        tone_alignment=18,
        originality=17,
        clarity_coherence=17,
        policy_cleanliness=10,
        total=92,
        reasons=["Strong quality across dimensions."],
        warnings=[],
    )
    low_quality = QualityScore(
        format_compliance=30,
        tone_alignment=12,
        originality=8,
        clarity_coherence=11,
        policy_cleanliness=10,
        total=71,
        reasons=["Quality is acceptable but repetitive."],
        warnings=[],
    )

    results = [
        CompareModelResult(
            ok=True,
            backend="ollama",
            model="slow-high-quality",
            latency_ms=900,
            items=["Line A", "Line B", "Line C"],
            quality=high_quality,
        ),
        CompareModelResult(
            ok=True,
            backend="ollama",
            model="fast-lower-quality",
            latency_ms=50,
            items=["Line A", "Line A2", "Line A3"],
            quality=low_quality,
        ),
    ]

    winner = pick_quality_winner(results)
    assert winner is not None
    assert winner.model == "slow-high-quality"
    assert winner.total_score == 92
