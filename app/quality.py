"""Quality-first scoring for compare-model outputs."""

from __future__ import annotations

import json
import re
from typing import Protocol

from app.schemas import (
    CompareModelResult,
    CompareModelsWinner,
    GenerateSingleRequest,
    OutputSpec,
    ProsConsStructuredOutput,
    QualityScore,
)

EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
JSON_HINT_PATTERN = re.compile(r'(?:"items"|"phrases"|"pros"|"cons")')
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:\d{1,3}[\)\].:-]\s*|[-*•]\s*)")
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
FORBIDDEN_PREFIXES = ("sure", "here's", "heres")

EMOTION_WORDS = {
    "heart",
    "love",
    "warm",
    "gratitude",
    "grateful",
    "hope",
    "kind",
    "kindness",
    "care",
    "gentle",
    "sincere",
}
HUMOR_WORDS = {
    "laugh",
    "joke",
    "witty",
    "playful",
    "smile",
    "chuckle",
    "funny",
    "humor",
}
POETIC_WORDS = {"moon", "river", "dawn", "echo", "breeze", "stars", "quiet", "whisper"}
CONNECTORS = {"then", "but", "so", "after", "because", "when", "while", "therefore", "meanwhile"}

OVERUSED_TEMPLATE_PATTERNS = [
    re.compile(r"\byou got this\b"),
    re.compile(r"\bnever give up\b"),
    re.compile(r"\bmake it happen\b"),
    re.compile(r"\bbelieve in yourself\b"),
    re.compile(r"\bstay positive\b"),
]
OPENING_PATTERNS = ("start your", "wishing you")


class OutputLike(Protocol):
    """Minimum output shape required for quality scoring."""

    items: list[str]
    raw_text: str | None
    structured_output: ProsConsStructuredOutput | None


def clamp(value: int, lower: int, upper: int) -> int:
    """Clamp one integer into an inclusive range."""

    return max(lower, min(upper, value))


def tokenize(value: str) -> list[str]:
    """Split text into normalized word tokens."""

    return [token.lower() for token in WORD_PATTERN.findall(value)]


def word_count(value: str) -> int:
    """Count words in a string."""

    return len(tokenize(value))


def sentence_count(value: str) -> int:
    """Approximate sentence count."""

    return len([part.strip() for part in SENTENCE_SPLIT_PATTERN.split(value) if part.strip()])


def paragraph_count(value: str) -> int:
    """Count paragraphs split by blank lines."""

    chunks = re.split(r"\n\s*\n", value.strip())
    return len([chunk for chunk in chunks if chunk.strip()])


def strip_prefix(value: str) -> str:
    """Remove numbering and bullet prefixes."""

    return LIST_PREFIX_PATTERN.sub("", value.strip()).strip()


def to_plain_text(output: OutputLike) -> str:
    """Render one output payload as plain text for scoring heuristics."""

    if output.raw_text:
        return output.raw_text.strip()
    if output.structured_output is not None:
        pros = "\n".join(f"- {item}" for item in output.structured_output.pros)
        cons = "\n".join(f"- {item}" for item in output.structured_output.cons)
        return f"Pros:\n{pros}\nCons:\n{cons}".strip()
    return "\n".join(output.items).strip()


def has_emoji(value: str) -> bool:
    """Return whether text contains emoji glyphs."""

    return bool(EMOJI_PATTERN.search(value))


def starts_with_forbidden_prefix(value: str) -> bool:
    """Return whether text begins with disallowed assistant-preface words."""

    lowered = value.lstrip().lower()
    return any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def detect_json_leakage(value: str) -> bool:
    """Best-effort JSON leakage detection."""

    stripped = value.strip()
    if not stripped:
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, (dict, list)):
        return True
    if stripped.startswith(("{", "[")) or stripped.endswith(("}", "]")):
        return True
    return bool(JSON_HINT_PATTERN.search(stripped))


def blocked_cliche_hits(payload: GenerateSingleRequest, text: str) -> list[str]:
    """Return blocked cliches found in output text."""

    if not payload.avoid_cliches:
        return []

    lowered = text.lower()
    hits: list[str] = []
    for phrase in payload.avoid_phrases:
        candidate = phrase.strip().lower()
        if not candidate:
            continue
        if candidate in lowered:
            hits.append(phrase)
    return hits


def one_liner_lines(output: OutputLike) -> list[str]:
    """Return normalized one-liner lines for quality heuristics."""

    if output.items:
        return [strip_prefix(item) for item in output.items if strip_prefix(item)]
    text = to_plain_text(output)
    return [strip_prefix(line) for line in text.splitlines() if strip_prefix(line)]


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""

    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def format_compliance_score(
    payload: GenerateSingleRequest,
    output: OutputLike,
) -> tuple[int, bool, list[str], list[str]]:
    """Return format compliance score and structure-hard-failure metadata."""

    spec = payload.output_spec or OutputSpec()
    reasons: list[str] = []
    warnings: list[str] = []
    wrong_structure = False
    score = 30

    if spec.format == "one_liner":
        lines = one_liner_lines(output)
        expected = spec.structure.items or payload.count
        if len(lines) != expected:
            wrong_structure = True
            reasons.append(
                f"Hard penalty: one_liner must return exactly {expected} lines (received {len(lines)})."
            )
        if spec.structure.max_words_per_line is not None and any(
            word_count(line) > spec.structure.max_words_per_line for line in lines
        ):
            wrong_structure = True
            reasons.append(
                f"Hard penalty: one_liner line length exceeded max_words_per_line={spec.structure.max_words_per_line}."
            )

    elif spec.format == "pros_cons":
        expected = spec.structure.items or 4
        structured = output.structured_output
        if structured is None:
            wrong_structure = True
            reasons.append("Hard penalty: pros_cons must include Pros and Cons sections.")
        else:
            if len(structured.pros) != expected or len(structured.cons) != expected:
                wrong_structure = True
                reasons.append(
                    f"Hard penalty: pros_cons requires exactly {expected} bullets per section."
                )

    elif spec.format == "verse":
        lines = one_liner_lines(output)
        min_lines = spec.structure.items or 8
        max_lines = spec.structure.max_lines or 12
        text = to_plain_text(output)
        if len(lines) < min_lines or len(lines) > max_lines:
            wrong_structure = True
            reasons.append(
                f"Hard penalty: verse requires {min_lines}-{max_lines} lines (received {len(lines)})."
            )
        if paragraph_count(text) > 1:
            wrong_structure = True
            reasons.append("Hard penalty: verse cannot use paragraph blocks.")

    elif spec.format == "story":
        text = to_plain_text(output)
        lowered = text.lower()
        has_setup = "setup" in lowered
        has_turn = "turn" in lowered
        has_resolution = "resolution" in lowered
        if not (has_setup and has_turn and has_resolution):
            wrong_structure = True
            reasons.append("Hard penalty: story must have Setup, Turn, and Resolution sections.")
        connector_hits = sum(1 for token in tokenize(text) if token in CONNECTORS)
        if connector_hits < 2:
            score -= 6
            warnings.append("Story continuity looks weak; add more causal progression.")

    elif spec.format == "paragraph":
        text = to_plain_text(output)
        if paragraph_count(text) != 1 or sentence_count(text) < 3 or sentence_count(text) > 6:
            wrong_structure = True
            reasons.append("Hard penalty: paragraph format requires one paragraph and 3-6 sentences.")

    elif spec.format == "one_page":
        text = to_plain_text(output)
        count = paragraph_count(text)
        if count < 2 or count > 4:
            wrong_structure = True
            reasons.append("Hard penalty: one_page requires 2-4 short paragraphs.")

    if wrong_structure:
        return 0, True, reasons, warnings
    return clamp(score, 0, 30), False, reasons, warnings


def tone_alignment_score(payload: GenerateSingleRequest, text: str) -> tuple[int, list[str], list[str]]:
    """Score tone alignment to requested sliders and style hints."""

    tokens = tokenize(text)
    reasons: list[str] = []
    warnings: list[str] = []
    score = 10

    emotion_hits = sum(1 for token in tokens if token in EMOTION_WORDS)
    humor_hits = sum(1 for token in tokens if token in HUMOR_WORDS)
    poetic_hits = sum(1 for token in tokens if token in POETIC_WORDS)

    if payload.tone_emotion_pct >= 60:
        score += 6 if emotion_hits >= 2 else 1
    elif payload.tone_emotion_pct <= 30:
        score += 4 if emotion_hits == 0 else 1
    else:
        score += 3 if emotion_hits >= 1 else 1

    if payload.tone_funny_pct >= 60:
        score += 6 if humor_hits >= 1 else 1
    elif payload.tone_funny_pct <= 30:
        score += 4 if humor_hits == 0 else 1
    else:
        score += 3 if humor_hits >= 1 else 1

    if payload.tone_style == "poetic":
        score += 2 if poetic_hits >= 1 else -1
    if payload.tone_style == "witty":
        score += 2 if humor_hits >= 1 else -1

    final_score = clamp(score, 0, 20)
    if final_score >= 16:
        reasons.append("Tone is well aligned with requested emotional/humor direction.")
    elif final_score <= 8:
        warnings.append("Tone alignment is weak relative to requested sliders.")
    return final_score, reasons, warnings


def originality_score(output: OutputLike, text: str) -> tuple[int, list[str], list[str]]:
    """Score originality via repetition and similarity heuristics."""

    reasons: list[str] = []
    warnings: list[str] = []
    lines = one_liner_lines(output)
    if not lines:
        lines = [segment.strip() for segment in text.splitlines() if segment.strip()]

    score = 20
    openings = [" ".join(tokenize(line)[:2]) for line in lines if tokenize(line)]
    repeated_openings = len(openings) - len(set(openings))
    if repeated_openings > 0:
        score -= min(8, repeated_openings * 3)
        warnings.append("Repeated line openings reduce originality.")

    lowered_lines = [line.lower() for line in lines]
    for opener in OPENING_PATTERNS:
        repeated = sum(1 for line in lowered_lines if line.startswith(opener))
        if repeated >= 2:
            score -= min(6, repeated * 2)
            warnings.append(f'Repeated opening "{opener}" detected.')

    lowered_text = text.lower()
    template_hits = sum(len(pattern.findall(lowered_text)) for pattern in OVERUSED_TEMPLATE_PATTERNS)
    if template_hits > 0:
        score -= min(8, template_hits * 2)
        warnings.append("Overused motivational templates detected.")

    similarity_penalty = 0
    token_sets = [set(tokenize(line)) for line in lines if tokenize(line)]
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            similarity = jaccard_similarity(left, right)
            if similarity > 0.8:
                similarity_penalty += 5
            elif similarity > 0.65:
                similarity_penalty += 3
    if similarity_penalty:
        score -= min(8, similarity_penalty)
        warnings.append("Intra-output similarity is high.")

    final_score = clamp(score, 0, 20)
    if final_score >= 16:
        reasons.append("Output shows good lexical variety and distinct phrasing.")
    return final_score, reasons, warnings


def clarity_coherence_score(payload: GenerateSingleRequest, text: str, output: OutputLike) -> tuple[int, list[str], list[str]]:
    """Score clarity and narrative coherence heuristically."""

    reasons: list[str] = []
    warnings: list[str] = []
    lines = one_liner_lines(output)
    score = 20

    short_lines = sum(1 for line in lines if word_count(line) < 3)
    if short_lines:
        score -= min(6, short_lines * 2)
        warnings.append("Some lines are too short to be clear.")

    sentence_lengths = [len(tokenize(sentence)) for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if sentence_lengths:
        avg_sentence = sum(sentence_lengths) / len(sentence_lengths)
        if avg_sentence > 30:
            score -= 4
            warnings.append("Average sentence length is high; readability may drop.")
        if avg_sentence < 5:
            score -= 3
            warnings.append("Sentences are very short; cohesion may be weak.")

    if payload.output_spec and payload.output_spec.format == "story":
        connector_hits = sum(1 for token in tokenize(text) if token in CONNECTORS)
        if connector_hits < 2:
            score -= 4
            warnings.append("Story continuity needs stronger transitions.")

    final_score = clamp(score, 0, 20)
    if final_score >= 16:
        reasons.append("Output is clear and coherent.")
    return final_score, reasons, warnings


def policy_cleanliness_score(payload: GenerateSingleRequest, text: str) -> tuple[int, bool, bool, list[str], list[str]]:
    """Score policy cleanliness and return hard-penalty flags."""

    reasons: list[str] = []
    warnings: list[str] = []
    score = 10

    json_leakage = detect_json_leakage(text)
    blocked_hits = blocked_cliche_hits(payload, text)

    if json_leakage:
        score = 0
        reasons.append("Hard penalty: JSON leakage detected.")

    if blocked_hits:
        score = 0
        reasons.append(f"Hard penalty: blocked cliches detected ({', '.join(blocked_hits[:3])}).")

    if payload.emoji_policy == "none" and has_emoji(text):
        score = max(0, score - 4)
        warnings.append("Emoji policy violation detected.")

    if starts_with_forbidden_prefix(text):
        score = max(0, score - 2)
        warnings.append("Output starts with a forbidden prefix.")

    return clamp(score, 0, 10), json_leakage, bool(blocked_hits), reasons, warnings


def length_extreme_warning(payload: GenerateSingleRequest, text: str) -> list[str]:
    """Return weak warning for extreme length relative to target words."""

    spec = payload.output_spec or OutputSpec()
    target = spec.length.target_words
    if target is None or target <= 0:
        return []

    total_words = word_count(text)
    low_cutoff = max(1, int(target * 0.35))
    high_cutoff = int(target * 2.5)
    if total_words < low_cutoff:
        return [f"Length warning: output may be too short for target_words={target}."]
    if total_words > high_cutoff:
        return [f"Length warning: output may be too long for target_words={target}."]
    return []


def score_quality(payload: GenerateSingleRequest, output: OutputLike) -> tuple[QualityScore, bool]:
    """Compute the quality-first score and winner-eligibility flag for one output."""

    text = to_plain_text(output)
    reasons: list[str] = []
    warnings: list[str] = []

    format_score, wrong_structure, format_reasons, format_warnings = format_compliance_score(payload, output)
    reasons.extend(format_reasons)
    warnings.extend(format_warnings)

    tone_score, tone_reasons, tone_warnings = tone_alignment_score(payload, text)
    reasons.extend(tone_reasons)
    warnings.extend(tone_warnings)

    originality, originality_reasons, originality_warnings = originality_score(output, text)
    reasons.extend(originality_reasons)
    warnings.extend(originality_warnings)

    clarity, clarity_reasons, clarity_warnings = clarity_coherence_score(payload, text, output)
    reasons.extend(clarity_reasons)
    warnings.extend(clarity_warnings)

    policy, json_leakage, blocked_cliches, policy_reasons, policy_warnings = policy_cleanliness_score(payload, text)
    reasons.extend(policy_reasons)
    warnings.extend(policy_warnings)

    warnings.extend(length_extreme_warning(payload, text))

    subtotal = format_score + tone_score + originality + clarity + policy
    hard_penalty = 0
    if wrong_structure:
        hard_penalty += 30
    if json_leakage:
        hard_penalty += 30
    if blocked_cliches:
        hard_penalty += 20

    total = clamp(subtotal - hard_penalty, 0, 100)
    score = QualityScore(
        format_compliance=format_score,
        tone_alignment=tone_score,
        originality=originality,
        clarity_coherence=clarity,
        policy_cleanliness=policy,
        total=total,
        reasons=reasons,
        warnings=warnings,
    )
    is_valid = not (wrong_structure or json_leakage or blocked_cliches)
    if not is_valid:
        score.warnings.append("Invalid for winner selection due to hard penalties.")
    return score, is_valid


def is_quality_valid(score: QualityScore | None) -> bool:
    """Return whether a quality score is eligible for winner selection."""

    if score is None:
        return False
    return not any(item.startswith("Hard penalty:") for item in score.reasons)


def pick_quality_winner(results: list[CompareModelResult]) -> CompareModelsWinner | None:
    """Choose the winner by quality score among valid outputs only."""

    candidates = [item for item in results if item.ok and is_quality_valid(item.quality)]
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (
            item.quality.total if item.quality is not None else 0,
            item.quality.format_compliance if item.quality is not None else 0,
            item.quality.clarity_coherence if item.quality is not None else 0,
            item.quality.originality if item.quality is not None else 0,
        ),
    )
    assert best.quality is not None
    return CompareModelsWinner(
        backend=best.backend,
        model=best.model,
        total_score=best.quality.total,
    )
