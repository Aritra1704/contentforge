# LLM Comparator

Stateless FastAPI service for lightweight content generation across local Ollama chat models and an optional Groq backend. It is designed to be called by other apps such as TinySe or ecard factory, one request per backend/model, with explicit busy handling instead of any DB-backed queue.

## Features

- Stateless API with no SQLite or persistent storage
- Ollama-first generation flow for `mistral:7b`, `qwen2.5:7b-instruct`, and `llama3.1:8b`
- Layered prompt structure: SYSTEM + GUIDELINES + USER TASK
- Embedding-model rejection for chat requests
- In-memory concurrency guard with `429 busy` responses and `Retry-After`
- Structured JSON errors with request IDs and optional trace IDs
- Request lifecycle logs to stdout with stack traces on failures
- Discovery and health endpoints for callers that need to probe readiness

## Setup

Use Python 3.12 or 3.13.

```bash
cd llm-comparator
python3.13 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
source venv/bin/activate
python -m uvicorn app.main:app --reload
```

## Config

Relevant env vars are defined in `.env.example`:

- `OLLAMA_URL`
- `OLLAMA_CHAT_MODELS`
- `OLLAMA_EMBEDDING_MODELS`
- `MAX_CONCURRENT_JOBS`
- `MAX_QUEUE`
- `BUSY_RETRY_AFTER_MS`
- `REQUEST_TIMEOUT_SEC`
- `LOG_LEVEL`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `JUDGE_ENABLED`
- `JUDGE_PROVIDER` (default: `groq`)
- `JUDGE_MODEL` (default: `llama-3.3-70b-versatile`)
- `JUDGE_TIMEOUT_SEC`
- `JUDGE_CONNECT_TIMEOUT_SEC`
- `JUDGE_FALLBACK_TO_BASELINE`

### Recommended judge setup

Keep generation targets unchanged, and use Groq as the quality judge backend:

```env
JUDGE_ENABLED=true
JUDGE_PROVIDER=groq
JUDGE_MODEL=llama-3.3-70b-versatile
JUDGE_TIMEOUT_SEC=120
JUDGE_CONNECT_TIMEOUT_SEC=10
JUDGE_FALLBACK_TO_BASELINE=true
GROQ_API_KEY=your_groq_api_key
```

This setup avoids long local judge runtimes while preserving the same generation flow.

## Endpoints

- `GET /health`
- `GET /models`
- `POST /generate/single`
- `POST /generate/compare-models`
- `POST /generation/compare-models`

### Create content

Primary endpoint:

```text
POST /generate/single
```

Request body:

```json
{
  "theme_name": "Warm Wishes",
  "tone_funny_pct": 20,
  "tone_emotion_pct": 70,
  "prompt_keywords": ["family", "gratitude"],
  "visual_style": "soft watercolor",
  "backend": "ollama",
  "model": "qwen2.5:7b-instruct",
  "count": 3,
  "max_tokens": 300,
  "max_words": 16,
  "emoji_policy": "none",
  "tone_style": "conversational",
  "audience": "general",
  "avoid_cliches": true,
  "avoid_phrases": [
    "new week",
    "rise and shine",
    "inner strength",
    "you got this",
    "make it happen",
    "shine bright",
    "positive vibes"
  ],
  "output_format": "numbered",
  "temperature": 0.8,
  "trace_id": "tinyse-run-001"
}
```

Successful response:

```json
{
  "ok": true,
  "backend": "ollama",
  "model": "qwen2.5:7b-instruct",
  "items": ["...", "...", "..."],
  "meta": {
    "latency_ms": 1234,
    "request_id": "generated-request-id",
    "trace_id": "tinyse-run-001",
    "busy": false,
    "applied_settings": {
      "max_words": 16,
      "emoji_policy": "none",
      "tone_style": "conversational",
      "avoid_cliches": true
    }
  },
  "errors": null
}
```

Every response includes `X-Request-Id`. Example:

```bash
curl -i -X POST http://127.0.0.1:8000/generate/single \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Warm Wishes",
    "tone_funny_pct": 20,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["family", "gratitude"],
    "visual_style": "soft watercolor",
    "backend": "ollama",
    "model": "qwen2.5:7b-instruct",
    "count": 3,
    "max_tokens": 300,
    "max_words": 14,
    "emoji_policy": "light",
    "tone_style": "witty",
    "audience": "young professionals",
    "avoid_cliches": true,
    "avoid_phrases": ["new week", "you got this"],
    "output_format": "lines",
    "temperature": 0.8,
    "trace_id": "tinyse-run-001"
  }'
```

Look for:

```text
X-Request-Id: <request-id>
```

### Compare multiple models

Use this when you want one request to test multiple backend/model pairs and still receive per-model error objects:

```bash
curl -X POST http://127.0.0.1:8000/generation/compare-models \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Warm Wishes",
    "tone_funny_pct": 20,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["family", "gratitude"],
    "visual_style": "soft watercolor",
    "targets": [
      {"backend": "ollama", "model": "qwen2.5:7b-instruct"},
      {"backend": "groq", "model": "llama-3.3-70b-versatile"}
    ],
    "count": 3,
    "max_tokens": 300,
    "temperature": 0.8,
    "trace_id": "tinyse-compare-001"
  }'
```

If one target fails, the endpoint still returns `results[]` with a structured `error` object for that target.

### Health

```bash
curl http://127.0.0.1:8000/health
```

### Model discovery

```bash
curl http://127.0.0.1:8000/models
```

### Generate with Qwen

```bash
curl -X POST http://127.0.0.1:8000/generate/single \
  -H "Content-Type: application/json" \
  -d '{
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
    "trace_id": "tinyse-run-001"
  }'
```

### Generate with Mistral

```bash
curl -X POST http://127.0.0.1:8000/generate/single \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Warm Wishes",
    "tone_funny_pct": 20,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["family", "gratitude"],
    "visual_style": "soft watercolor",
    "backend": "ollama",
    "model": "mistral:7b",
    "count": 3,
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

### Generate with Llama 3.1

```bash
curl -X POST http://127.0.0.1:8000/generate/single \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Warm Wishes",
    "tone_funny_pct": 20,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["family", "gratitude"],
    "visual_style": "soft watercolor",
    "backend": "ollama",
    "model": "llama3.1:8b",
    "count": 3,
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

### Busy response example

If all job slots are occupied, the service returns `429`:

```json
{
  "ok": false,
  "error": {
    "error_type": "busy",
    "message": "The service is busy. Retry later.",
    "backend": "ollama",
    "model": "qwen2.5:7b-instruct",
    "http_status": 429,
    "retry_after_ms": 2000
  },
  "meta": {
    "request_id": "generated-request-id",
    "trace_id": "tinyse-run-001",
    "busy": true
  }
}
```

It also includes the header `Retry-After: 2`.

### Failure response format

All failures use the same JSON shape:

```json
{
  "ok": false,
  "error": {
    "error_type": "not_configured",
    "message": "Groq backend is not configured.",
    "backend": "groq",
    "model": "llama-3.3-70b-versatile",
    "http_status": 503
  },
  "meta": {
    "request_id": "generated-request-id",
    "trace_id": "tinyse-run-001",
    "busy": false
  }
}
```

Possible `error_type` values include:

- `not_configured`
- `rate_limited`
- `busy`
- `provider_error`
- `network_error`
- `service_unreachable`
- `validation_error`
- `internal_error`

## Logs

Logs are written to stdout.

- Each request logs `request_started` and `request_completed`
- Failures log `request_failed` with `request_id`, `trace_id`, `backend`, and `model`
- Stack traces are included at `ERROR` level for failures
- Busy rejections log `request_busy` at `INFO`

## Minimal test plan

Run the automated tests:

```bash
pytest tests/ -v
```

Manual curl checks:

1. Call `POST /generate/single` with `model = "qwen2.5:7b-instruct"`.
2. Call `POST /generate/single` with `model = "mistral:7b"`.
3. Call `POST /generate/single` with `model = "llama3.1:8b"`.
4. Call `POST /generate/single` with `max_words = 5` and confirm each item has at most 5 words.
5. Call `POST /generate/single` with `emoji_policy = "none"` and confirm output has no emojis.
6. Call `POST /generate/single` with `model = "nomic-embed-text:latest"` and confirm `400`.
7. Call `POST /generate/single` with `backend = "groq"` and no `GROQ_API_KEY`, then confirm `503` with `error.error_type = "not_configured"`.
8. Fire two requests quickly and confirm the second returns `429` with `Retry-After: 2`.
9. Call `POST /generation/compare-models` with one valid Ollama target and one failing Groq target, then confirm the failing target returns a structured `error` object in `results[]`.
