"""LLM backend helpers for stateless content generation."""

from __future__ import annotations

from json import JSONDecodeError
import json
import logging
from typing import Any

import httpx

from app.config import settings
from app.schemas import GenerateSingleRequest

logger = logging.getLogger(__name__)
AsyncClient = httpx.AsyncClient

SYSTEM_PROMPT = (
    "You are a content generation assistant. Follow the tone sliders exactly, keep the writing "
    "natural and non-cheesy, and return JSON only."
)


class UpstreamServiceError(Exception):
    """Represents an upstream backend or dependency failure."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def is_embedding_model(model_name: str) -> bool:
    """Return whether a model name refers to an embedding-only model."""

    normalized = model_name.strip().lower()
    if normalized in {name.lower() for name in settings.ollama_embedding_models}:
        return True
    return any(normalized.startswith(prefix.lower()) for prefix in settings.ollama_embedding_prefixes)


def build_user_prompt(payload: GenerateSingleRequest) -> str:
    """Construct a compact user prompt from the request body."""

    keywords = ", ".join(payload.prompt_keywords) if payload.prompt_keywords else "none"
    return (
        f"Theme: {payload.theme_name}\n"
        f"Visual style: {payload.visual_style}\n"
        f"Tone sliders: funny={payload.tone_funny_pct}/100, emotional={payload.tone_emotion_pct}/100\n"
        f"Keywords: {keywords}\n"
        f"Generate exactly {payload.count} distinct items.\n"
        "Each item should be 8 to 20 words, usable as short content copy, and feel clean rather than cheesy.\n"
        'Return valid JSON only in this shape: {"items": ["...", "..."]}'
    )


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


def normalize_item(value: Any) -> str:
    """Convert one parsed model output item into display text."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    return str(value).strip()


def parse_items(content: str, *, count: int) -> list[str]:
    """Parse a JSON-first model response into a list of generated items."""

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

    items: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        item = normalize_item(raw_item)
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
        if len(items) == count:
            return items

    for raw_line in content.splitlines():
        line = raw_line.strip().lstrip("-*0123456789. ").strip()
        if not line or line in seen:
            continue
        seen.add(line)
        items.append(line)
        if len(items) == count:
            return items

    return items


def build_messages(payload: GenerateSingleRequest) -> list[dict[str, str]]:
    """Create the system and user messages shared by all chat backends."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(payload)},
    ]


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


async def fetch_ollama_tags(*, timeout_sec: float | None = None) -> list[str]:
    """Fetch model names from the local Ollama instance."""

    timeout = httpx.Timeout(timeout_sec or settings.request_timeout_sec)
    url = f"{settings.ollama_url.rstrip('/')}/api/tags"

    try:
        async with AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("Ollama request timed out.", status_code=504) from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"Ollama returned HTTP {exc.response.status_code}.",
            status_code=502,
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamServiceError("Ollama is unavailable.", status_code=503) from exc

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
    except UpstreamServiceError:
        return False
    return True


async def call_ollama(payload: GenerateSingleRequest) -> str:
    """Call Ollama's chat endpoint and return the assistant content."""

    options: dict[str, Any] = {
        "temperature": payload.temperature,
        "num_predict": payload.max_tokens,
    }
    if payload.seed is not None:
        options["seed"] = payload.seed

    request_body = {
        "model": payload.model,
        "messages": build_messages(payload),
        "stream": False,
        "format": "json",
        "options": options,
    }
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"

    try:
        async with AsyncClient(timeout=httpx.Timeout(settings.request_timeout_sec)) as client:
            response = await client.post(url, json=request_body)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise UpstreamServiceError("Ollama request timed out.", status_code=504) from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"Ollama returned HTTP {exc.response.status_code}.",
            status_code=502,
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamServiceError("Ollama is unavailable.", status_code=503) from exc

    response_payload = response.json()
    return str(response_payload.get("message", {}).get("content", "")).strip()


async def call_groq(payload: GenerateSingleRequest) -> str:
    """Call Groq's OpenAI-compatible chat endpoint and return the response content."""

    if not settings.groq_api_key.strip():
        raise UpstreamServiceError("Groq backend is not configured.", status_code=503)

    request_body: dict[str, Any] = {
        "model": payload.model,
        "messages": build_messages(payload),
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
        raise UpstreamServiceError("Groq request timed out.", status_code=504) from exc
    except httpx.HTTPStatusError as exc:
        raise UpstreamServiceError(
            f"Groq returned HTTP {exc.response.status_code}.",
            status_code=502,
        ) from exc
    except httpx.RequestError as exc:
        raise UpstreamServiceError("Groq is unavailable.", status_code=503) from exc

    response_payload = response.json()
    choices = response_payload.get("choices", [])
    if not choices:
        raise UpstreamServiceError("Groq returned an empty response.", status_code=502)

    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


async def generate_items(payload: GenerateSingleRequest) -> list[str]:
    """Generate content items for one request using the selected backend."""

    if payload.backend == "ollama":
        if is_embedding_model(payload.model):
            raise ValueError("Embedding model cannot be used for chat generation")
        content = await call_ollama(payload)
    else:
        content = await call_groq(payload)

    items = parse_items(content, count=payload.count)
    if not items:
        logger.warning("backend=%s model=%s returned no parseable items", payload.backend, payload.model)
        raise UpstreamServiceError("Upstream model returned no items.", status_code=502)
    return items
