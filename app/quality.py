"""Quality-first scoring for compare-model outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
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
    "miss",
    "cherish",
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
    re.compile(r"\bkeep shining\b"),
]
OPENING_PATTERNS = ("start your", "wishing you", "on your", "may your")
BLAND_GENERIC_PATTERNS = [
    re.compile(r"\bwishing you\b"),
    re.compile(r"\bon your special day\b"),
    re.compile(r"\breminded of the journey\b"),
    re.compile(r"\bfilled with joy and love\b"),
    re.compile(r"\bmay this day bring\b"),
    re.compile(r"\bhope this message finds you well\b"),
]
ROBOTIC_PATTERNS = [
    re.compile(r"\bthis message is to\b"),
    re.compile(r"\baccording to\b"),
    re.compile(r"\bin conclusion\b"),
    re.compile(r"\bit is important to note\b"),
    re.compile(r"\bas an ai\b"),
]
CLOSURE_HINTS = {
    "finally",
    "in the end",
    "at last",
    "with love",
    "thank you",
    "forever",
    "together",
    "always",
    "today",
    "tonight",
    "tomorrow",
}
DANGLING_ENDINGS = {
    "and",
    "or",
    "but",
    "because",
    "with",
    "to",
    "for",
    "when",
    "while",
    "if",
    "that",
    "which",
    "as",
    "so",
}


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


def _line_tokens(output: OutputLike, text: str) -> list[list[str]]:
    """Return tokenized non-empty lines for repeated-structure checks."""

    lines = one_liner_lines(output)
    if not lines:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [tokenize(line) for line in lines if tokenize(line)]


def _closing_sentence(text: str) -> str:
    """Return the trailing sentence-ish segment."""

    stripped = text.strip()
    if not stripped:
        return ""
    parts = [part.strip() for part in re.split(r"[\n.!?]+", stripped) if part.strip()]
    if not parts:
        return ""
    return parts[-1]


def _seems_incomplete_ending(text: str) -> bool:
    """Return whether text likely ends abruptly."""

    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped.endswith("..."):
        return True
    if stripped[-1] in {",", ":", ";", "-", "("}:
        return True

    closing = _closing_sentence(stripped)
    tokens = tokenize(closing)
    if tokens and tokens[-1] in DANGLING_ENDINGS:
        return True

    # Long-form content should usually end with sentence punctuation.
    if len(stripped) > 80 and stripped[-1].isalnum():
        return True

    return False


def _has_natural_closure(text: str) -> bool:
    """Return whether ending looks complete and resolved."""

    if _seems_incomplete_ending(text):
        return False
    closing = _closing_sentence(text).lower()
    if not closing:
        return False
    if any(hint in closing for hint in CLOSURE_HINTS):
        return True
    return word_count(closing) >= 5


def task_fit_score(
    payload: GenerateSingleRequest,
    output: OutputLike,
    text: str,
) -> tuple[int, bool, list[str], list[str]]:
    """Score fit to requested task, style, audience, and format."""

    spec = payload.output_spec or OutputSpec()
    reasons: list[str] = []
    warnings: list[str] = []
    wrong_structure = False
    score = 25

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
        if structured is None or not structured.pros or not structured.cons:
            wrong_structure = True
            reasons.append("Hard penalty: pros_cons must include both Pros and Cons sections.")
        elif len(structured.pros) != expected or len(structured.cons) != expected:
            wrong_structure = True
            reasons.append(f"Hard penalty: pros_cons requires exactly {expected} bullets per section.")

    elif spec.format == "verse":
        lines = one_liner_lines(output)
        min_lines = spec.structure.items or 8
        max_lines = spec.structure.max_lines or 12
        if len(lines) < min_lines or len(lines) > max_lines:
            wrong_structure = True
            reasons.append(
                f"Hard penalty: verse requires {min_lines}-{max_lines} lines (received {len(lines)})."
            )
        if paragraph_count(text) > 1:
            wrong_structure = True
            reasons.append("Hard penalty: verse cannot use paragraph blocks.")

    elif spec.format == "story":
        lowered = text.lower()
        has_beginning = any(token in lowered for token in ("begin", "setup", "once", "at first", "one day"))
        has_progression = any(token in lowered for token in ("then", "next", "after", "but", "meanwhile"))
        has_resolution = any(token in lowered for token in ("finally", "in the end", "resolution", "at last"))
        if not (has_beginning and has_progression and has_resolution):
            wrong_structure = True
            reasons.append("Hard penalty: story must show beginning, progression, and resolution.")

    elif spec.format == "paragraph":
        if paragraph_count(text) != 1:
            wrong_structure = True
            reasons.append("Hard penalty: paragraph format requires exactly one paragraph.")
        sentences = sentence_count(text)
        if sentences < 3 or sentences > 6:
            wrong_structure = True
            reasons.append("Hard penalty: paragraph format requires 3-6 sentences.")

    elif spec.format == "one_page":
        paragraphs = paragraph_count(text)
        if paragraphs < 2 or paragraphs > 4:
            wrong_structure = True
            reasons.append("Hard penalty: one_page requires 2-4 short paragraphs.")

    tokens = tokenize(text)
    emotion_hits = sum(1 for token in tokens if token in EMOTION_WORDS)
    humor_hits = sum(1 for token in tokens if token in HUMOR_WORDS)
    poetic_hits = sum(1 for token in tokens if token in POETIC_WORDS)

    if payload.tone_emotion_pct >= 60 and emotion_hits < 2:
        score -= 4
        warnings.append("Task fit: emotional tone requested but emotional language is weak.")
    elif payload.tone_emotion_pct <= 30 and emotion_hits > 2:
        score -= 3
        warnings.append("Task fit: output is more emotional than requested.")

    if payload.tone_funny_pct >= 60 and humor_hits < 1:
        score -= 4
        warnings.append("Task fit: humor requested but humorous language is weak.")
    elif payload.tone_funny_pct <= 30 and humor_hits > 2:
        score -= 3
        warnings.append("Task fit: output is more playful than requested.")

    if payload.tone_style == "poetic" and poetic_hits < 1:
        score -= 3
        warnings.append("Task fit: poetic style requested but imagery is limited.")
    if payload.tone_style == "witty" and humor_hits < 1:
        score -= 3
        warnings.append("Task fit: witty style requested but wit cues are limited.")

    keyword_hits = 0
    lowered = text.lower()
    for keyword in payload.prompt_keywords:
        candidate = keyword.strip().lower()
        if candidate and candidate in lowered:
            keyword_hits += 1
    if payload.prompt_keywords:
        coverage = keyword_hits / max(1, len(payload.prompt_keywords))
        if coverage < 0.5:
            score -= 3
            warnings.append("Task fit: keyword coverage is low.")

    if wrong_structure:
        return 0, True, reasons, warnings
    return clamp(score, 0, 25), False, reasons, warnings


def originality_score(output: OutputLike, text: str) -> tuple[int, int, int, list[str], list[str]]:
    """Score originality and return bland/overused penalties."""

    reasons: list[str] = []
    warnings: list[str] = []
    line_tokens = _line_tokens(output, text)
    score = 20
    bland_penalty = 0
    overused_penalty = 0

    openings = [" ".join(tokens[:2]) for tokens in line_tokens if len(tokens) >= 2]
    repeated_openings = len(openings) - len(set(openings))
    if repeated_openings > 0:
        penalty = min(6, repeated_openings * 2)
        score -= penalty
        overused_penalty += penalty
        warnings.append("Repeated line openings reduce originality.")

    lowered_text = text.lower()
    template_hits = sum(len(pattern.findall(lowered_text)) for pattern in OVERUSED_TEMPLATE_PATTERNS)
    if template_hits > 0:
        penalty = min(8, template_hits * 2)
        score -= penalty
        overused_penalty += penalty
        warnings.append("Overused motivational templates detected.")

    bland_hits = sum(len(pattern.findall(lowered_text)) for pattern in BLAND_GENERIC_PATTERNS)
    if bland_hits > 0:
        penalty = min(10, bland_hits * 3)
        score -= penalty
        bland_penalty += penalty
        warnings.append("Bland generic phrasing detected.")

    token_sets = [set(tokens) for tokens in line_tokens if tokens]
    similarity_penalty = 0
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            similarity = jaccard_similarity(left, right)
            if similarity > 0.8:
                similarity_penalty += 4
            elif similarity > 0.65:
                similarity_penalty += 2
    if similarity_penalty > 0:
        penalty = min(6, similarity_penalty)
        score -= penalty
        overused_penalty += penalty
        warnings.append("High intra-output similarity detected.")

    final = clamp(score, 0, 20)
    if final >= 16:
        reasons.append("Output avoids generic templates and uses distinct phrasing.")
    return final, clamp(bland_penalty, 0, 30), clamp(overused_penalty, 0, 30), reasons, warnings


def emotional_authenticity_score(text: str) -> tuple[int, int, list[str], list[str]]:
    """Score whether writing sounds human and emotionally believable."""

    reasons: list[str] = []
    warnings: list[str] = []
    score = 14
    robotic_penalty = 0

    tokens = tokenize(text)
    emotion_hits = sum(1 for token in tokens if token in EMOTION_WORDS)
    human_voice_hits = sum(1 for token in tokens if token in {"you", "your", "we", "us", "our", "together"})

    if emotion_hits >= 2:
        score += 3
    else:
        score -= 2

    if human_voice_hits >= 2:
        score += 2
    else:
        score -= 2

    lowered = text.lower()
    robotic_hits = sum(len(pattern.findall(lowered)) for pattern in ROBOTIC_PATTERNS)
    if robotic_hits > 0:
        penalty = min(10, robotic_hits * 4)
        score -= penalty
        robotic_penalty += penalty
        warnings.append("Robotic/template-like tone detected.")

    if starts_with_forbidden_prefix(text):
        score -= 2
        robotic_penalty += 2
        warnings.append("Output begins with assistant-like preface wording.")

    lexical = set(tokens)
    if tokens and len(lexical) / len(tokens) < 0.35:
        score -= 2
        warnings.append("Low lexical variety makes tone feel less human.")

    final = clamp(score, 0, 20)
    if final >= 16:
        reasons.append("Emotional voice feels human and believable.")
    return final, clamp(robotic_penalty, 0, 30), reasons, warnings


def completeness_score(
    payload: GenerateSingleRequest,
    output: OutputLike,
    text: str,
    wrong_structure: bool,
) -> tuple[int, bool, int, list[str], list[str]]:
    """Score completeness and detect unfinished endings."""

    reasons: list[str] = []
    warnings: list[str] = []
    score = 15
    incomplete_penalty = 0
    spec = payload.output_spec or OutputSpec()

    long_form = spec.format in {"paragraph", "story", "one_page"}
    incomplete_long_form = False

    if wrong_structure:
        score -= 8

    if spec.format == "paragraph":
        if not _has_natural_closure(text):
            penalty = 12
            score -= penalty
            incomplete_penalty += penalty
            incomplete_long_form = True
            reasons.append("Hard penalty: paragraph output lacks a complete natural ending.")

    elif spec.format == "story":
        lowered = text.lower()
        has_resolution = any(token in lowered for token in ("finally", "in the end", "resolution", "at last"))
        if not has_resolution or not _has_natural_closure(text):
            penalty = 12
            score -= penalty
            incomplete_penalty += penalty
            incomplete_long_form = True
            reasons.append("Hard penalty: story output lacks clear closure/resolution.")

    elif spec.format == "one_page":
        if not _has_natural_closure(text):
            penalty = 10
            score -= penalty
            incomplete_penalty += penalty
            incomplete_long_form = True
            reasons.append("Hard penalty: one_page output lacks a complete ending.")

    elif spec.format == "verse":
        lines = one_liner_lines(output)
        if lines:
            avg_words = sum(word_count(line) for line in lines) / len(lines)
            if avg_words > 12:
                score -= 8
                warnings.append("Verse looks like prose split by line breaks.")

    elif spec.format == "pros_cons":
        structured = output.structured_output
        if structured is None or not structured.pros or not structured.cons:
            score -= 10
            warnings.append("Pros/cons output is incomplete.")

    if long_form and _seems_incomplete_ending(text):
        penalty = 8
        score -= penalty
        incomplete_penalty += penalty
        incomplete_long_form = True
        warnings.append("Long-form ending appears abrupt.")

    final = clamp(score, 0, 15)
    if final >= 12 and not incomplete_long_form:
        reasons.append("Output feels complete and ready to use.")
    return final, incomplete_long_form, clamp(incomplete_penalty, 0, 30), reasons, warnings


def clarity_and_flow_score(payload: GenerateSingleRequest, text: str, output: OutputLike) -> tuple[int, list[str], list[str]]:
    """Score readability, coherence, and flow."""

    reasons: list[str] = []
    warnings: list[str] = []
    lines = one_liner_lines(output)
    score = 10

    short_lines = sum(1 for line in lines if word_count(line) < 3)
    if short_lines:
        score -= min(4, short_lines)
        warnings.append("Some lines are too short to read naturally.")

    sentence_lengths = [len(tokenize(sentence)) for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if sentence_lengths:
        avg_sentence = sum(sentence_lengths) / len(sentence_lengths)
        if avg_sentence > 30:
            score -= 2
            warnings.append("Average sentence length is high; readability may drop.")
        if avg_sentence < 5:
            score -= 2
            warnings.append("Sentences are very short; flow may feel choppy.")

    if re.search(r"[!?]{3,}", text):
        score -= 1
        warnings.append("Excess punctuation hurts readability.")

    if payload.output_spec and payload.output_spec.format == "story":
        connector_hits = sum(1 for token in tokenize(text) if token in CONNECTORS)
        if connector_hits < 2:
            score -= 2
            warnings.append("Story progression lacks connective transitions.")

    final = clamp(score, 0, 10)
    if final >= 8:
        reasons.append("Text is clear and easy to follow.")
    return final, reasons, warnings


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


def _legacy_field_projection(
    *,
    task_fit: int,
    emotional_authenticity: int,
    clarity_and_flow: int,
) -> tuple[int, int, int]:
    """Map new dimensions into legacy field ranges for compatibility."""

    format_compliance = clamp(int(round(task_fit * 30 / 25)), 0, 30)
    tone_alignment = clamp(emotional_authenticity, 0, 20)
    clarity_coherence = clamp(int(round(clarity_and_flow * 2)), 0, 20)
    return format_compliance, tone_alignment, clarity_coherence


def score_quality(payload: GenerateSingleRequest, output: OutputLike) -> tuple[QualityScore, bool]:
    """Compute quality-first score and winner-eligibility flag for one output."""

    text = to_plain_text(output)
    reasons: list[str] = []
    warnings: list[str] = []

    task_fit, wrong_structure, task_reasons, task_warnings = task_fit_score(payload, output, text)
    reasons.extend(task_reasons)
    warnings.extend(task_warnings)

    originality, bland_penalty, overused_penalty, originality_reasons, originality_warnings = originality_score(
        output,
        text,
    )
    reasons.extend(originality_reasons)
    warnings.extend(originality_warnings)

    emotional_authenticity, robotic_penalty, emotional_reasons, emotional_warnings = emotional_authenticity_score(text)
    reasons.extend(emotional_reasons)
    warnings.extend(emotional_warnings)

    completeness, incomplete_long_form, incomplete_penalty, completeness_reasons, completeness_warnings = completeness_score(
        payload,
        output,
        text,
        wrong_structure,
    )
    reasons.extend(completeness_reasons)
    warnings.extend(completeness_warnings)

    clarity_and_flow, clarity_reasons, clarity_warnings = clarity_and_flow_score(payload, text, output)
    reasons.extend(clarity_reasons)
    warnings.extend(clarity_warnings)

    policy_cleanliness, json_leakage, blocked_cliches, policy_reasons, policy_warnings = policy_cleanliness_score(
        payload,
        text,
    )
    reasons.extend(policy_reasons)
    warnings.extend(policy_warnings)

    warnings.extend(length_extreme_warning(payload, text))

    soft_penalty_total = bland_penalty + incomplete_penalty + overused_penalty + robotic_penalty

    hard_penalty = 0
    if wrong_structure:
        hard_penalty += 30
    if json_leakage:
        hard_penalty += 30
    if blocked_cliches:
        hard_penalty += 20
    if incomplete_long_form:
        hard_penalty += 25

    subtotal = task_fit + originality + emotional_authenticity + completeness + clarity_and_flow + policy_cleanliness
    total = clamp(subtotal - soft_penalty_total - hard_penalty, 0, 100)

    format_compliance, tone_alignment, clarity_coherence = _legacy_field_projection(
        task_fit=task_fit,
        emotional_authenticity=emotional_authenticity,
        clarity_and_flow=clarity_and_flow,
    )

    score = QualityScore(
        task_fit=task_fit,
        originality=originality,
        emotional_authenticity=emotional_authenticity,
        completeness=completeness,
        clarity_and_flow=clarity_and_flow,
        policy_cleanliness=policy_cleanliness,
        bland_generic_penalty=clamp(bland_penalty, 0, 30),
        incomplete_ending_penalty=clamp(incomplete_penalty, 0, 30),
        overused_pattern_penalty=clamp(overused_penalty, 0, 30),
        robotic_tone_penalty=clamp(robotic_penalty, 0, 30),
        total=total,
        format_compliance=format_compliance,
        tone_alignment=tone_alignment,
        clarity_coherence=clarity_coherence,
        reasons=reasons,
        warnings=warnings,
    )

    is_valid = not (wrong_structure or json_leakage or blocked_cliches or incomplete_long_form)
    if not is_valid:
        score.warnings.append("Invalid for winner selection due to hard penalties.")
    return score, is_valid


def is_quality_valid(score: QualityScore | None) -> bool:
    """Return whether a quality score is eligible for winner selection."""

    if score is None:
        return False
    return not any(item.startswith("Hard penalty:") for item in score.reasons)


def apply_compare_quality_penalties(results: list[CompareModelResult]) -> None:
    """Apply cross-candidate overused-pattern penalties in compare mode."""

    candidates = [item for item in results if item.ok and item.quality is not None]
    if len(candidates) < 2:
        return

    opening_to_models: dict[str, list[str]] = defaultdict(list)
    text_tokens: dict[str, set[str]] = {}
    model_keyed: dict[str, CompareModelResult] = {}

    for item in candidates:
        key = f"{item.backend}:{item.model}"
        model_keyed[key] = item
        text = to_plain_text(item)
        lines = [line for line in one_liner_lines(item) if line]
        if not lines:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_opening = ""
        if lines:
            tokens = tokenize(lines[0])
            if len(tokens) >= 2:
                first_opening = " ".join(tokens[:2])
        if first_opening:
            opening_to_models[first_opening].append(key)
        text_tokens[key] = set(tokenize(text))

    penalties: Counter[str] = Counter()
    for model_keys in opening_to_models.values():
        if len(model_keys) < 2:
            continue
        for key in model_keys:
            penalties[key] += 4

    keys = list(text_tokens.keys())
    for index, left_key in enumerate(keys):
        for right_key in keys[index + 1 :]:
            similarity = jaccard_similarity(text_tokens[left_key], text_tokens[right_key])
            if similarity > 0.75:
                penalties[left_key] += 3
                penalties[right_key] += 3

    for key, penalty_total in penalties.items():
        result = model_keyed.get(key)
        if result is None or result.quality is None:
            continue

        penalty = clamp(penalty_total, 0, 10)
        if penalty <= 0:
            continue

        result.quality.overused_pattern_penalty = clamp(
            result.quality.overused_pattern_penalty + penalty,
            0,
            30,
        )
        result.quality.originality = clamp(result.quality.originality - min(penalty, 6), 0, 20)
        result.quality.total = clamp(result.quality.total - penalty, 0, 100)
        result.quality.warnings.append("Cross-candidate overused structure detected.")


def pick_quality_winner(results: list[CompareModelResult]) -> CompareModelsWinner | None:
    """Choose winner by quality score among valid outputs only."""

    candidates = [item for item in results if item.ok and is_quality_valid(item.quality)]
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda item: (
            item.quality.total if item.quality is not None else 0,
            item.quality.task_fit if item.quality is not None else 0,
            item.quality.completeness if item.quality is not None else 0,
            item.quality.emotional_authenticity if item.quality is not None else 0,
            item.quality.originality if item.quality is not None else 0,
            item.quality.clarity_and_flow if item.quality is not None else 0,
        ),
    )
    assert best.quality is not None
    return CompareModelsWinner(
        backend=best.backend,
        model=best.model,
        total_score=best.quality.total,
    )
