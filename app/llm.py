"""LLM backend helpers for stateless content generation."""

from __future__ import annotations

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
from app.schemas import GenerateSingleRequest, OutputFormat
from src.prompts.phrase_prompt import build_messages

logger = logging.getLogger(__name__)
AsyncClient = httpx.AsyncClient

LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:\d{1,3}[\)\].:-]\s*|[-*•]\s*)")
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


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


def dedupe_items(items: list[str], *, count: int) -> list[str]:
    """Deduplicate while preserving order and cap to the requested count."""

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) == count:
            return deduped
    return deduped


def parse_items(
    content: str,
    *,
    count: int,
    output_format: OutputFormat,
) -> list[str]:
    """Parse model text into phrase items with JSON leakage safeguards."""

    payload = extract_json_fragment(content)
    raw_items: list[Any] = []
    if isinstance(payload, dict):
        candidate = payload.get("items")
        if not isinstance(candidate, list):
            candidate = payload.get("phrases")
        if isinstance(candidate, list):
            raw_items = candidate
    elif isinstance(payload, list):
        raw_items = payload

    remove_numbering = output_format == "lines"
    parsed: list[str] = []
    for raw_item in raw_items:
        item = strip_list_prefix(normalize_item(raw_item), remove_numbering=remove_numbering)
        if not item or looks_like_json_fragment(item):
            continue
        parsed.append(item)

    for item in extract_quoted_items(content, count=count):
        parsed.append(strip_list_prefix(item, remove_numbering=remove_numbering))

    for raw_line in content.splitlines():
        line = strip_list_prefix(raw_line, remove_numbering=remove_numbering)
        if not line or looks_like_json_fragment(line):
            continue
        parsed.append(line)

    return dedupe_items(parsed, count=count)


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
    items: list[str],
    check_content_for_json: bool = True,
) -> list[str]:
    """Return validation issue codes for one generation attempt."""

    issues: list[str] = []
    if len(items) < payload.count:
        issues.append("insufficient_items")
    if payload.emoji_policy == "none" and any(contains_emoji(item) for item in items):
        issues.append("emoji_not_allowed")
    if any(word_count(item) > payload.max_words for item in items):
        issues.append("max_words_exceeded")
    if any(looks_like_json_fragment(item) for item in items):
        issues.append("json_leakage")
    if check_content_for_json and is_json_leakage(content):
        issues.append("json_leakage")
    return sorted(set(issues))


def build_retry_reminder(payload: GenerateSingleRequest, issues: list[str]) -> str:
    """Build a strict one-time retry reminder from validation issues."""

    reminders: list[str] = [f"Return exactly {payload.count} phrases."]
    if "json_leakage" in issues:
        reminders.append("Output ONLY phrases, no JSON.")
    if "insufficient_items" in issues:
        reminders.append("Do not omit any phrase; return all requested phrases.")
    if "emoji_not_allowed" in issues:
        reminders.append("Do not use emojis.")
    if "max_words_exceeded" in issues:
        reminders.append(f"Keep every phrase at or under {payload.max_words} words.")
    if payload.output_format == "lines":
        reminders.append("Use plain lines without numbering.")
    else:
        reminders.append("Use numbered lines only, one phrase per line.")
    return " ".join(reminders)


def append_retry_reminder(messages: list[dict[str, str]], reminder: str) -> list[dict[str, str]]:
    """Append one strict reminder turn for the retry attempt."""

    return [*messages, {"role": "user", "content": f"STRICT REMINDER: {reminder}"}]


def apply_last_resort_fixes(payload: GenerateSingleRequest, items: list[str]) -> list[str]:
    """Apply best-effort post-processing after retry before hard failure."""

    remove_numbering = payload.output_format == "lines"
    cleaned: list[str] = []
    for item in items:
        candidate = strip_list_prefix(item, remove_numbering=remove_numbering)
        if payload.emoji_policy == "none":
            candidate = remove_emojis(candidate)
        candidate = trim_to_word_limit(candidate, payload.max_words)
        candidate = " ".join(candidate.split()).strip()
        if not candidate or looks_like_json_fragment(candidate):
            continue
        cleaned.append(candidate)
    return dedupe_items(cleaned, count=payload.count)


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


async def call_ollama(
    payload: GenerateSingleRequest,
    *,
    messages: list[dict[str, str]],
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

    try:
        async with AsyncClient(timeout=httpx.Timeout(settings.request_timeout_sec)) as client:
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

    try:
        async with AsyncClient(timeout=httpx.Timeout(settings.request_timeout_sec)) as client:
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
) -> str:
    """Call the configured backend for one generation attempt."""

    if payload.backend == "ollama":
        return await call_ollama(payload, messages=messages)
    return await call_groq(payload, messages=messages)


async def generate_items(payload: GenerateSingleRequest) -> list[str]:
    """Generate content items for one request using the selected backend."""

    if payload.backend == "ollama" and is_embedding_model(payload.model):
        raise ValidationServiceError(
            "Embedding model cannot be used for chat generation",
            backend="ollama",
            model=payload.model,
        )

    base_messages = build_messages(payload)
    content = await call_backend(payload, messages=base_messages)
    items = parse_items(
        content,
        count=payload.count,
        output_format=payload.output_format,
    )
    issues = validate_items(payload=payload, content=content, items=items)

    if issues:
        reminder = build_retry_reminder(payload, issues)
        retry_messages = append_retry_reminder(base_messages, reminder)
        retry_content = await call_backend(payload, messages=retry_messages)
        retry_items = parse_items(
            retry_content,
            count=payload.count,
            output_format=payload.output_format,
        )
        retry_issues = validate_items(payload=payload, content=retry_content, items=retry_items)
        content = retry_content
        items = retry_items
        issues = retry_issues

    if issues:
        items = apply_last_resort_fixes(payload, items)
        issues = validate_items(
            payload=payload,
            content="",
            items=items,
            check_content_for_json=False,
        )

    if issues:
        raise ProviderError(
            "Model output did not satisfy output constraints.",
            backend=payload.backend,
            model=payload.model,
            response_status=502,
            details={
                "issues": issues,
                "requested_count": payload.count,
                "received_count": len(items),
                "contains_json_like_text": is_json_leakage(content),
            },
        )

    return items[: payload.count]
