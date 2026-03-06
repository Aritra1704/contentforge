"""LLM backend helpers for stateless content generation."""

from __future__ import annotations

from dataclasses import dataclass
from json import JSONDecodeError
import json
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.errors import (
    NetworkError,
    NotConfiguredError,
    ProviderError,
    ProviderRateLimitedError,
    ServiceUnreachableError,
    ValidationServiceError,
)
from app.observability import sanitize_text
from app.schemas import GenerateSingleRequest, OutputSpec, ProsConsStructuredOutput
from src.prompts.phrase_prompt import build_messages

logger = logging.getLogger(__name__)
AsyncClient = httpx.AsyncClient

LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:\d{1,3}[\)\].:-]\s*|[-*•]\s*)")
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
LONG_FORM_FORMATS = {"paragraph", "one_page", "story"}
FORBIDDEN_PREFIXES = ("sure", "here's", "heres")
PROS_CONS_HEADING_PATTERN = re.compile(r"^\s*(pros|cons)\s*:\s*(.*)$", flags=re.IGNORECASE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
FORBIDDEN_PREFIX_PATTERN = re.compile(r"^\s*(?:sure|here(?:'|)s)\b[\s,:-]*", flags=re.IGNORECASE)
SOFT_VALIDATION_ISSUES = {
    "min_words_not_met",
    "max_words_exceeded",
    "max_words_per_line_exceeded",
}


@dataclass
class GeneratedOutput:
    """Parsed output container produced from one model response."""

    items: list[str]
    raw_text: str | None = None
    structured_output: ProsConsStructuredOutput | None = None


def is_embedding_model(model_name: str) -> bool:
    """Return whether a model name refers to an embedding-only model."""

    normalized = model_name.strip().lower()
    if normalized in {name.lower() for name in settings.ollama_embedding_models}:
        return True
    return any(normalized.startswith(prefix.lower()) for prefix in settings.ollama_embedding_prefixes)


def extract_json_fragment(content: str) -> Any:
    """Best-effort JSON extraction for chat models that wrap or prefix their output."""

    try:
        return json.loads(content)
    except JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = content.find(opener)
        end = content.rfind(closer)
        if start != -1 and end != -1 and end > start:
            fragment = content[start : end + 1]
            try:
                return json.loads(fragment)
            except JSONDecodeError:
                continue

    return None


def looks_like_json_fragment(value: str) -> bool:
    """Return whether a string appears to be a raw JSON container or fragment."""

    stripped = value.strip()
    if not stripped:
        return False

    markers = ('{"', '{"items"', '{"phrases"', '["', "{}")
    if stripped.startswith(markers):
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if stripped.endswith("}") or stripped.endswith("]"):
        return True
    return '"items"' in stripped or '"phrases"' in stripped


def normalize_item(value: Any) -> str:
    """Convert one parsed model output item into display text."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value).strip()


def decode_quoted_candidate(value: str) -> str:
    """Best-effort decode of one quoted string extracted from malformed JSON."""

    try:
        return json.loads(f'"{value}"').strip()
    except JSONDecodeError:
        return value.strip()


def extract_quoted_items(content: str, *, count: int) -> list[str]:
    """Salvage phrase-like strings from malformed JSON output."""

    quoted_strings = re.findall(r'"((?:[^"\\]|\\.)+)"', content)
    ignored_tokens = {"items", "phrases", "text", "tone", "word_count"}
    items: list[str] = []
    seen: set[str] = set()

    for raw_value in quoted_strings:
        item = decode_quoted_candidate(raw_value)
        normalized = item.strip()
        if not normalized or normalized.lower() in ignored_tokens:
            continue
        if not any(character.isalpha() for character in normalized):
            continue
        if looks_like_json_fragment(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(normalized)
        if len(items) == count:
            return items

    return items


def strip_list_prefix(value: str, *, remove_numbering: bool) -> str:
    """Remove leading numbering/bullets from one line when required."""

    stripped = value.strip()
    if not stripped:
        return ""
    if remove_numbering:
        stripped = LIST_PREFIX_PATTERN.sub("", stripped).strip()
    return stripped


def contains_emoji(value: str) -> bool:
    """Return whether a phrase contains emoji characters."""

    return bool(EMOJI_PATTERN.search(value))


def remove_emojis(value: str) -> str:
    """Strip emojis from a phrase and normalize whitespace."""

    cleaned = EMOJI_PATTERN.sub("", value)
    return " ".join(cleaned.split()).strip()


def word_count(value: str) -> int:
    """Compute a simple whitespace-based word count."""

    return len([token for token in value.strip().split() if token])


def trim_to_word_limit(value: str, max_words: int) -> str:
    """Trim one phrase down to the configured word limit."""

    words = [token for token in value.strip().split() if token]
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).strip()


def output_spec(payload: GenerateSingleRequest) -> OutputSpec:
    """Return the normalized output spec for one request."""

    if payload.output_spec is not None:
        return payload.output_spec
    return OutputSpec()


def output_item_floor(payload: GenerateSingleRequest) -> int:
    """Return minimum required output items based on OutputSpec."""

    spec = output_spec(payload)
    if spec.format == "one_liner":
        return spec.structure.items or payload.count
    if spec.format == "pros_cons":
        sections = spec.structure.sections or ["Pros", "Cons"]
        items_per_section = spec.structure.items or 4
        return max(1, len(sections)) * items_per_section
    if spec.format == "verse":
        return spec.structure.items or 8
    return 1


def output_item_ceiling(payload: GenerateSingleRequest) -> int:
    """Return maximum parsed output items based on OutputSpec."""

    spec = output_spec(payload)
    if spec.format == "one_liner":
        return spec.structure.items or payload.count
    if spec.format == "pros_cons":
        sections = spec.structure.sections or ["Pros", "Cons"]
        items_per_section = spec.structure.items or 4
        return max(1, len(sections)) * items_per_section
    if spec.format == "verse":
        return spec.structure.max_lines or 12
    return 1


def should_strip_list_prefix(payload: GenerateSingleRequest) -> bool:
    """Return whether numbering/bullet prefixes should be removed."""

    spec = output_spec(payload)
    return bool(spec.structure.no_numbering or spec.structure.no_lists)


def extract_structured_items(content: str) -> list[Any]:
    """Extract list-like item candidates from JSON-shaped model output."""

    payload = extract_json_fragment(content)
    if isinstance(payload, dict):
        candidate = payload.get("items")
        if not isinstance(candidate, list):
            candidate = payload.get("phrases")
        if isinstance(candidate, list):
            return candidate
    if isinstance(payload, list):
        return payload
    return []


def is_section_heading(value: str, sections: list[str]) -> bool:
    """Return whether one line matches a section heading."""

    normalized = value.strip().rstrip(":").lower()
    return any(normalized == section.strip().rstrip(":").lower() for section in sections)


def starts_with_forbidden_prefix(value: str) -> bool:
    """Return whether one output begins with disallowed assistant-preface words."""

    lowered = value.lstrip().lower()
    return any(lowered.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def strip_forbidden_prefix(value: str) -> str:
    """Remove leading assistant prefaces like 'Sure' and 'Here's'."""

    cleaned = value.strip()
    while cleaned and starts_with_forbidden_prefix(cleaned):
        updated = FORBIDDEN_PREFIX_PATTERN.sub("", cleaned).strip()
        if not updated or updated == cleaned:
            break
        cleaned = updated
    return cleaned


def sentence_count(value: str) -> int:
    """Return the approximate number of sentences in one text block."""

    return len([part.strip() for part in SENTENCE_SPLIT_PATTERN.split(value) if part.strip()])


def paragraph_count(value: str) -> int:
    """Return the paragraph count split by blank lines."""

    parts = re.split(r"\n\s*\n", value.strip())
    return len([part for part in parts if part.strip()])


def split_candidate_lines(value: str) -> list[str]:
    """Split one block of text into short line-like candidates."""

    candidates: list[str] = []
    text = value.strip()
    if not text:
        return candidates

    sentence_chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    if len(sentence_chunks) <= 1:
        sentence_chunks = [chunk.strip() for chunk in re.split(r"[;\n]+", text) if chunk.strip()]

    for chunk in sentence_chunks:
        sub_chunks = [piece.strip() for piece in re.split(r",\s+", chunk) if piece.strip()]
        if not sub_chunks:
            continue
        if len(sub_chunks) == 1:
            candidates.append(sub_chunks[0])
            continue
        candidates.extend(sub_chunks)

    if not candidates:
        candidates.append(text)
    return candidates


def has_required_pros_cons_headings(value: str) -> bool:
    """Return whether text includes Pros: and Cons: section headings."""

    headings = {"pros": 0, "cons": 0}
    for raw_line in value.splitlines():
        line = raw_line.strip()
        match = PROS_CONS_HEADING_PATTERN.match(line)
        if not match:
            continue
        section = match.group(1).lower()
        headings[section] = headings.get(section, 0) + 1
    return headings.get("pros", 0) >= 1 and headings.get("cons", 0) >= 1


def parse_pros_cons_sections(content: str, *, items_per_section: int) -> ProsConsStructuredOutput:
    """Extract Pros/Cons sections from model text into a structured payload."""

    parsed_json = extract_json_fragment(content)
    if isinstance(parsed_json, dict):
        pros_value = parsed_json.get("pros", parsed_json.get("Pros", []))
        cons_value = parsed_json.get("cons", parsed_json.get("Cons", []))
        if isinstance(pros_value, list) and isinstance(cons_value, list):
            pros = dedupe_items(
                [
                    strip_list_prefix(normalize_item(item), remove_numbering=True)
                    for item in pros_value
                    if strip_list_prefix(normalize_item(item), remove_numbering=True)
                ],
                limit=items_per_section,
            )
            cons = dedupe_items(
                [
                    strip_list_prefix(normalize_item(item), remove_numbering=True)
                    for item in cons_value
                    if strip_list_prefix(normalize_item(item), remove_numbering=True)
                ],
                limit=items_per_section,
            )
            return ProsConsStructuredOutput(pros=pros, cons=cons)

    pros: list[str] = []
    cons: list[str] = []
    current: str | None = None
    fallback_lines: list[str] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or looks_like_json_fragment(line):
            continue

        heading_match = PROS_CONS_HEADING_PATTERN.match(line)
        if heading_match:
            current = heading_match.group(1).lower()
            inline_value = strip_list_prefix(heading_match.group(2), remove_numbering=True)
            if inline_value:
                if current == "pros":
                    pros.append(inline_value)
                else:
                    cons.append(inline_value)
            continue

        candidate = strip_list_prefix(line, remove_numbering=True)
        if not candidate or looks_like_json_fragment(candidate):
            continue
        fallback_lines.append(candidate)

        if current == "pros":
            pros.append(candidate)
        elif current == "cons":
            cons.append(candidate)

    if not pros and not cons and fallback_lines:
        pros = fallback_lines[:items_per_section]
        cons = fallback_lines[items_per_section : items_per_section * 2]

    return ProsConsStructuredOutput(
        pros=dedupe_items(pros, limit=items_per_section),
        cons=dedupe_items(cons, limit=items_per_section),
    )


def dedupe_items(items: list[str], *, limit: int | None = None) -> list[str]:
    """Deduplicate while preserving order and applying an optional limit."""

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if limit is not None and len(deduped) == limit:
            return deduped
    return deduped


def parse_items(
    content: str,
    *,
    payload: GenerateSingleRequest,
) -> GeneratedOutput:
    """Parse model text into output payload with JSON leakage safeguards."""

    spec = output_spec(payload)
    if spec.format == "pros_cons":
        items_per_section = spec.structure.items or 4
        structured_output = parse_pros_cons_sections(content, items_per_section=items_per_section)
        items = dedupe_items(
            [*structured_output.pros, *structured_output.cons],
            limit=output_item_ceiling(payload),
        )
        return GeneratedOutput(
            items=items,
            raw_text=None,
            structured_output=structured_output,
        )

    if spec.format in LONG_FORM_FORMATS:
        raw_items = extract_structured_items(content)
        if raw_items:
            separator = "\n\n" if spec.format == "one_page" else "\n"
            merged = separator.join(filter(None, (normalize_item(item) for item in raw_items))).strip()
            if merged and not looks_like_json_fragment(merged):
                return GeneratedOutput(items=[merged], raw_text=merged, structured_output=None)

        cleaned_lines: list[str] = []
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if looks_like_json_fragment(stripped):
                continue
            cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines).strip()
        if text:
            return GeneratedOutput(items=[text], raw_text=text, structured_output=None)

        quoted = extract_quoted_items(content, count=1)
        raw_text = quoted[0] if quoted else None
        return GeneratedOutput(items=quoted[:1], raw_text=raw_text, structured_output=None)

    raw_items = extract_structured_items(content)
    remove_numbering = should_strip_list_prefix(payload)
    parsed: list[str] = []
    for raw_item in raw_items:
        item = strip_list_prefix(normalize_item(raw_item), remove_numbering=remove_numbering)
        if not item or looks_like_json_fragment(item):
            continue
        parsed.append(item)

    for item in extract_quoted_items(content, count=max(1, output_item_ceiling(payload))):
        candidate = strip_list_prefix(item, remove_numbering=remove_numbering)
        parsed.append(candidate)

    for raw_line in content.splitlines():
        line = strip_list_prefix(raw_line, remove_numbering=remove_numbering)
        if not line or looks_like_json_fragment(line):
            continue
        parsed.append(line)

    items = dedupe_items(parsed, limit=output_item_ceiling(payload))
    raw_text = "\n".join(items) if items else None
    return GeneratedOutput(items=items, raw_text=raw_text, structured_output=None)


def is_json_leakage(content: str) -> bool:
    """Detect whether a response appears to contain JSON output."""

    stripped = content.strip()
    if not stripped:
        return False
    if "{" not in stripped and "[" not in stripped:
        return False
    if extract_json_fragment(stripped) is not None:
        return True
    return '"items"' in stripped or '"phrases"' in stripped


def validate_items(
    *,
    payload: GenerateSingleRequest,
    content: str,
    output: GeneratedOutput,
    check_content_for_json: bool = True,
) -> list[str]:
    """Return validation issue codes for one generation attempt."""

    spec = output_spec(payload)
    items = output.items
    issues: list[str] = []
    if starts_with_forbidden_prefix(content):
        issues.append("forbidden_prefix")
    if output.raw_text is not None and starts_with_forbidden_prefix(output.raw_text):
        issues.append("forbidden_prefix")
    if any(starts_with_forbidden_prefix(item) for item in items):
        issues.append("forbidden_prefix")

    if spec.format == "pros_cons":
        expected = spec.structure.items or 4
        structured = output.structured_output or ProsConsStructuredOutput()
        if len(structured.pros) < expected or len(structured.cons) < expected:
            issues.append("insufficient_items")
        if not has_required_pros_cons_headings(content):
            issues.append("invalid_structure")
    else:
        if len(items) < output_item_floor(payload):
            issues.append("insufficient_items")

    if spec.format == "verse":
        if len(items) > output_item_ceiling(payload):
            issues.append("line_count_out_of_range")
        if output.raw_text and paragraph_count(output.raw_text) > 1:
            issues.append("line_count_out_of_range")
        first_line = items[0].strip().lower() if items else ""
        if first_line.startswith("title:"):
            issues.append("invalid_structure")

    if spec.format == "paragraph":
        if output.raw_text is None:
            issues.append("insufficient_items")
        elif sentence_count(output.raw_text) < 3 or sentence_count(output.raw_text) > 6:
            issues.append("invalid_structure")

    if spec.format == "one_page" and output.raw_text is not None:
        paragraph_total = paragraph_count(output.raw_text)
        if paragraph_total < 2 or paragraph_total > 4:
            issues.append("invalid_structure")

    if spec.format == "story":
        text = (output.raw_text or "").lower()
        if "setup" not in text or "turn" not in text or "resolution" not in text:
            issues.append("invalid_structure")

    if payload.emoji_policy == "none" and any(contains_emoji(item) for item in items):
        issues.append("emoji_not_allowed")

    if spec.format in LONG_FORM_FORMATS:
        if items:
            generated_words = word_count(items[0])
            if spec.length.min_words is not None and generated_words < spec.length.min_words:
                issues.append("min_words_not_met")
            if spec.length.max_words is not None and generated_words > spec.length.max_words:
                issues.append("max_words_exceeded")
    elif spec.format == "verse":
        max_words_per_line = spec.structure.max_words_per_line
        if max_words_per_line is not None and any(word_count(item) > max_words_per_line for item in items):
            issues.append("max_words_per_line_exceeded")
        if spec.length.min_words is not None and any(word_count(item) < spec.length.min_words for item in items):
            issues.append("min_words_not_met")
        if spec.length.max_words is not None and any(word_count(item) > spec.length.max_words for item in items):
            issues.append("max_words_exceeded")
    else:
        if spec.length.min_words is not None and any(word_count(item) < spec.length.min_words for item in items):
            issues.append("min_words_not_met")
        if spec.length.max_words is not None and any(word_count(item) > spec.length.max_words for item in items):
            issues.append("max_words_exceeded")

    if any(looks_like_json_fragment(item) for item in items):
        issues.append("json_leakage")
    if check_content_for_json and is_json_leakage(content):
        issues.append("json_leakage")
    return sorted(set(issues))


def build_retry_reminder(payload: GenerateSingleRequest, issues: list[str]) -> str:
    """Build a strict one-time retry reminder from validation issues."""

    spec = output_spec(payload)
    if spec.format == "one_liner":
        reminders: list[str] = [f"Return exactly {output_item_floor(payload)} one-liners."]
    elif spec.format == "pros_cons":
        sections = spec.structure.sections or ["Pros", "Cons"]
        reminders = [
            f"Return {spec.structure.items or 4} points per section for {', '.join(sections)}."
        ]
    elif spec.format == "verse":
        reminders = [
            f"Return verse with {output_item_floor(payload)} to {output_item_ceiling(payload)} lines."
        ]
    elif spec.format == "paragraph":
        reminders = ["Return exactly one paragraph."]
    elif spec.format == "one_page":
        reminders = ["Return exactly one page-length response."]
    else:
        reminders = ["Return exactly one three-act story."]

    if "json_leakage" in issues:
        reminders.append("Output ONLY the requested content, no JSON.")
    if "forbidden_prefix" in issues:
        reminders.append("Do not start with 'Sure' or 'Here's'.")
    if "insufficient_items" in issues:
        reminders.append("Do not omit items; satisfy the requested structure.")
    if "invalid_structure" in issues:
        if spec.format == "paragraph":
            reminders.append("Return one plain-text paragraph with 3 to 6 sentences.")
        elif spec.format == "one_page":
            reminders.append("Return 2 to 4 short plain-text paragraphs.")
        elif spec.format == "story":
            reminders.append("Return exactly three sections: Setup, Turn, Resolution.")
        elif spec.format == "verse":
            reminders.append("Return only verse lines; no title and no paragraphs.")
    if "line_count_out_of_range" in issues:
        reminders.append(
            f"Keep line count between {output_item_floor(payload)} and {output_item_ceiling(payload)}."
        )
    if "emoji_not_allowed" in issues:
        reminders.append("Do not use emojis.")
    if "min_words_not_met" in issues and spec.length.min_words is not None:
        if spec.format in LONG_FORM_FORMATS:
            reminders.append(f"Use at least {spec.length.min_words} words in total.")
        else:
            reminders.append(f"Keep each item at or above {spec.length.min_words} words.")
    if "max_words_per_line_exceeded" in issues and spec.structure.max_words_per_line is not None:
        reminders.append(f"Keep each line at or under {spec.structure.max_words_per_line} words.")
    if "max_words_exceeded" in issues:
        if spec.format in LONG_FORM_FORMATS and spec.length.max_words is not None:
            reminders.append(f"Keep the full response at or under {spec.length.max_words} words.")
        elif spec.length.max_words is not None:
            reminders.append(f"Keep each item at or under {spec.length.max_words} words.")
    if spec.structure.no_lists:
        reminders.append("Do not use bullet lists.")
    if spec.structure.no_numbering:
        reminders.append("Do not number lines.")
    elif spec.format == "one_liner":
        reminders.append("Use numbered lines only, one item per line.")
    return " ".join(reminders)


def append_retry_reminder(messages: list[dict[str, str]], reminder: str) -> list[dict[str, str]]:
    """Append one strict reminder turn for the retry attempt."""

    return [*messages, {"role": "user", "content": f"STRICT REMINDER: {reminder}"}]


def apply_last_resort_fixes(payload: GenerateSingleRequest, output: GeneratedOutput) -> GeneratedOutput:
    """Apply best-effort post-processing after retry before hard failure."""

    spec = output_spec(payload)
    remove_numbering = should_strip_list_prefix(payload)
    max_words = spec.structure.max_words_per_line if spec.format == "verse" else spec.length.max_words

    if spec.format == "pros_cons":
        structured = output.structured_output or ProsConsStructuredOutput()
        items_per_section = spec.structure.items or 4

        def _clean_section(values: list[str]) -> list[str]:
            cleaned_section: list[str] = []
            for value in values:
                candidate = strip_list_prefix(value, remove_numbering=True)
                candidate = strip_forbidden_prefix(candidate)
                if payload.emoji_policy == "none":
                    candidate = remove_emojis(candidate)
                if max_words is not None:
                    candidate = trim_to_word_limit(candidate, max_words)
                candidate = " ".join(candidate.split()).strip()
                if not candidate or looks_like_json_fragment(candidate):
                    continue
                cleaned_section.append(candidate)
            return dedupe_items(cleaned_section, limit=items_per_section)

        pros = _clean_section(structured.pros)
        cons = _clean_section(structured.cons)
        return GeneratedOutput(
            items=dedupe_items([*pros, *cons], limit=output_item_ceiling(payload)),
            raw_text=None,
            structured_output=ProsConsStructuredOutput(pros=pros, cons=cons),
        )

    cleaned: list[str] = []
    for item in output.items:
        candidate = strip_list_prefix(item, remove_numbering=remove_numbering)
        candidate = strip_forbidden_prefix(candidate)
        if payload.emoji_policy == "none":
            candidate = remove_emojis(candidate)
        if max_words is not None:
            candidate = trim_to_word_limit(candidate, max_words)
        candidate = " ".join(candidate.split()).strip()
        if not candidate or looks_like_json_fragment(candidate):
            continue
        cleaned.append(candidate)

    normalized = dedupe_items(cleaned, limit=output_item_ceiling(payload))
    required_items = output_item_floor(payload)

    if spec.format in {"one_liner", "verse"} and len(normalized) < required_items:
        seed_text = output.raw_text or "\n".join(output.items)
        for fragment in split_candidate_lines(seed_text):
            candidate = strip_list_prefix(fragment, remove_numbering=remove_numbering)
            candidate = strip_forbidden_prefix(candidate)
            if payload.emoji_policy == "none":
                candidate = remove_emojis(candidate)
            if max_words is not None:
                candidate = trim_to_word_limit(candidate, max_words)
            candidate = " ".join(candidate.split()).strip()
            if not candidate or looks_like_json_fragment(candidate):
                continue
            normalized.append(candidate)
            normalized = dedupe_items(normalized, limit=output_item_ceiling(payload))
            if len(normalized) >= required_items:
                break

    if spec.format == "paragraph" and normalized:
        text = " ".join(part for part in normalized[0].splitlines() if part.strip()).strip()
        if text:
            normalized = [text]
            return GeneratedOutput(items=normalized, raw_text=text, structured_output=None)

    if spec.format == "one_page" and normalized:
        text = normalized[0]
        if paragraph_count(text) < 2:
            sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
            if len(sentences) >= 4:
                split_point = max(2, len(sentences) // 2)
                text = " ".join(sentences[:split_point]).strip() + "\n\n" + " ".join(sentences[split_point:]).strip()
        normalized = [text.strip()]
        return GeneratedOutput(items=normalized, raw_text=normalized[0], structured_output=None)

    if spec.format == "story" and normalized:
        text = normalized[0]
        lowered = text.lower()
        if "setup" not in lowered or "turn" not in lowered or "resolution" not in lowered:
            sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
            if len(sentences) >= 3:
                third = max(1, len(sentences) // 3)
                setup = " ".join(sentences[:third]).strip()
                turn = " ".join(sentences[third : third * 2]).strip()
                resolution = " ".join(sentences[third * 2 :]).strip()
                text = f"Setup:\n{setup}\n\nTurn:\n{turn}\n\nResolution:\n{resolution}"
        normalized = [text.strip()]
        return GeneratedOutput(items=normalized, raw_text=normalized[0], structured_output=None)

    if spec.format in LONG_FORM_FORMATS and normalized:
        return GeneratedOutput(items=[normalized[0]], raw_text=normalized[0], structured_output=None)

    raw_text = "\n".join(normalized) if normalized else None
    return GeneratedOutput(items=normalized, raw_text=raw_text, structured_output=None)


def parse_retry_after_ms(response: httpx.Response) -> int | None:
    """Translate Retry-After style headers into milliseconds."""

    header_value = response.headers.get("Retry-After")
    if header_value is None:
        return None

    try:
        seconds = float(header_value.strip())
    except ValueError:
        return None

    return max(0, int(seconds * 1000))


async def fetch_ollama_tags(*, timeout_sec: float | None = None) -> list[str]:
    """Fetch model names from the local Ollama instance."""

    timeout = httpx.Timeout(timeout_sec or settings.request_timeout_sec)
    url = f"{settings.ollama_url.rstrip('/')}/api/tags"

    try:
        async with AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise NetworkError(
            "Ollama request timed out.",
            backend="ollama",
            model=None,
            response_status=504,
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            "Ollama returned an error response.",
            backend="ollama",
            model=None,
            http_status=exc.response.status_code,
            response_status=502,
            details={"body_snippet": sanitize_text(exc.response.text)},
        ) from exc
    except httpx.RequestError as exc:
        raise ServiceUnreachableError(
            "Ollama is unavailable.",
            backend="ollama",
            model=None,
        ) from exc

    payload = response.json()
    models = payload.get("models", [])
    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def classify_ollama_models(model_names: list[str]) -> tuple[list[str], list[str]]:
    """Split Ollama tags into chat-capable and embedding-only models."""

    chat_models: list[str] = []
    embedding_models: list[str] = []
    for name in model_names:
        if is_embedding_model(name):
            embedding_models.append(name)
        else:
            chat_models.append(name)
    return chat_models, embedding_models


async def fetch_ollama_catalog() -> tuple[list[str], list[str]]:
    """Return discovered chat and embedding models from Ollama."""

    names = await fetch_ollama_tags()
    if not names:
        return settings.ollama_chat_models, settings.ollama_embedding_models
    return classify_ollama_models(names)


async def is_ollama_reachable() -> bool:
    """Return whether the local Ollama server responds to a tag listing."""

    try:
        await fetch_ollama_tags(timeout_sec=settings.healthcheck_timeout_sec)
    except ProviderError:
        return False
    except NetworkError:
        return False
    except ServiceUnreachableError:
        return False
    return True


def _resolve_timeout(
    *,
    timeout_sec: float | None = None,
    connect_timeout_sec: float | None = None,
) -> httpx.Timeout:
    """Build a request timeout object with optional connect override."""

    total_timeout = timeout_sec if timeout_sec is not None else settings.request_timeout_sec
    connect_timeout = connect_timeout_sec if connect_timeout_sec is not None else total_timeout
    return httpx.Timeout(total_timeout, connect=connect_timeout)


async def call_ollama(
    payload: GenerateSingleRequest,
    *,
    messages: list[dict[str, str]],
    timeout_sec: float | None = None,
    connect_timeout_sec: float | None = None,
) -> str:
    """Call Ollama's chat endpoint and return the assistant content."""

    options: dict[str, Any] = {
        "temperature": payload.temperature,
        "num_predict": payload.max_tokens,
    }
    if payload.seed is not None:
        options["seed"] = payload.seed

    request_body = {
        "model": payload.model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    timeout = _resolve_timeout(timeout_sec=timeout_sec, connect_timeout_sec=connect_timeout_sec)

    try:
        async with AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=request_body)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise NetworkError(
            "Ollama request timed out.",
            backend="ollama",
            model=payload.model,
            response_status=504,
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderError(
            "Ollama returned an error response.",
            backend="ollama",
            model=payload.model,
            http_status=exc.response.status_code,
            response_status=502,
            details={"body_snippet": sanitize_text(exc.response.text)},
        ) from exc
    except httpx.RequestError as exc:
        raise ServiceUnreachableError(
            "Ollama is unavailable.",
            backend="ollama",
            model=payload.model,
        ) from exc

    response_payload = response.json()
    return str(response_payload.get("message", {}).get("content", "")).strip()


async def call_groq(
    payload: GenerateSingleRequest,
    *,
    messages: list[dict[str, str]],
    timeout_sec: float | None = None,
    connect_timeout_sec: float | None = None,
) -> str:
    """Call Groq's OpenAI-compatible chat endpoint and return the response content."""

    if not settings.groq_api_key.strip():
        raise NotConfiguredError(
            "Groq backend is not configured.",
            backend="groq",
            model=payload.model,
        )

    request_body: dict[str, Any] = {
        "model": payload.model,
        "messages": messages,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }
    if payload.seed is not None:
        request_body["seed"] = payload.seed
    timeout = _resolve_timeout(timeout_sec=timeout_sec, connect_timeout_sec=connect_timeout_sec)

    try:
        async with AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise NetworkError(
            "Groq request timed out.",
            backend="groq",
            model=payload.model,
            response_status=504,
        ) from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        details = {"body_snippet": sanitize_text(exc.response.text)}
        if status_code == 429:
            raise ProviderRateLimitedError(
                "Groq rate limited the request.",
                backend="groq",
                model=payload.model,
                http_status=status_code,
                retry_after_ms=parse_retry_after_ms(exc.response),
                details=details,
            ) from exc
        raise ProviderError(
            "Groq returned an error response.",
            backend="groq",
            model=payload.model,
            http_status=status_code,
            response_status=502,
            details=details,
        ) from exc
    except httpx.RequestError as exc:
        raise ServiceUnreachableError(
            "Groq is unavailable.",
            backend="groq",
            model=payload.model,
        ) from exc

    response_payload = response.json()
    choices = response_payload.get("choices", [])
    if not choices:
        raise ProviderError(
            "Groq returned an empty response.",
            backend="groq",
            model=payload.model,
            response_status=502,
        )
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


async def call_backend(
    payload: GenerateSingleRequest,
    *,
    messages: list[dict[str, str]],
    timeout_sec: float | None = None,
    connect_timeout_sec: float | None = None,
) -> str:
    """Call the configured backend for one generation attempt."""

    if payload.backend == "ollama":
        return await call_ollama(
            payload,
            messages=messages,
            timeout_sec=timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
        )
    return await call_groq(
        payload,
        messages=messages,
        timeout_sec=timeout_sec,
        connect_timeout_sec=connect_timeout_sec,
    )


def normalize_generated_output(payload: GenerateSingleRequest, output: GeneratedOutput) -> GeneratedOutput:
    """Apply final output-shape normalization before returning response payloads."""

    spec = output_spec(payload)
    capped_items = output.items[: output_item_ceiling(payload)]

    if spec.format == "pros_cons":
        structured = output.structured_output or ProsConsStructuredOutput()
        items_per_section = spec.structure.items or 4
        pros = structured.pros[:items_per_section]
        cons = structured.cons[:items_per_section]
        return GeneratedOutput(
            items=capped_items,
            raw_text=None,
            structured_output=ProsConsStructuredOutput(pros=pros, cons=cons),
        )

    raw_text = output.raw_text
    if raw_text is None and capped_items:
        raw_text = "\n".join(capped_items)
    return GeneratedOutput(
        items=capped_items,
        raw_text=raw_text,
        structured_output=None,
    )


async def generate_items(payload: GenerateSingleRequest) -> GeneratedOutput:
    """Generate content items for one request using the selected backend."""

    if payload.backend == "ollama" and is_embedding_model(payload.model):
        raise ValidationServiceError(
            "Embedding model cannot be used for chat generation",
            backend="ollama",
            model=payload.model,
        )

    base_messages = build_messages(payload)
    content = await call_backend(payload, messages=base_messages)
    output = parse_items(
        content,
        payload=payload,
    )
    issues = validate_items(payload=payload, content=content, output=output)

    if issues:
        reminder = build_retry_reminder(payload, issues)
        retry_messages = append_retry_reminder(base_messages, reminder)
        retry_content = await call_backend(payload, messages=retry_messages)
        retry_output = parse_items(
            retry_content,
            payload=payload,
        )
        retry_issues = validate_items(payload=payload, content=retry_content, output=retry_output)
        content = retry_content
        output = retry_output
        issues = retry_issues

    if issues:
        output = apply_last_resort_fixes(payload, output)
        issues = validate_items(
            payload=payload,
            content="",
            output=output,
            check_content_for_json=False,
        )

    hard_issues = [issue for issue in issues if issue not in SOFT_VALIDATION_ISSUES]
    if hard_issues:
        raise ProviderError(
            "Model output did not satisfy output constraints.",
            backend=payload.backend,
            model=payload.model,
            response_status=502,
            details={
                "issues": issues,
                "hard_issues": hard_issues,
                "requested_count": output_item_floor(payload),
                "received_count": len(output.items),
                "contains_json_like_text": is_json_leakage(content),
            },
        )

    if issues:
        logger.warning(
            "output_constraints_soft_warning backend=%s model=%s issues=%s",
            payload.backend,
            payload.model,
            issues,
        )

    return normalize_generated_output(payload, output)
