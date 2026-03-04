# LLM Comparator

Stateless FastAPI service for lightweight content generation across local Ollama chat models and an optional Groq backend. It is designed to be called by other apps such as TinySe or ecard factory, one request per backend/model, with explicit busy handling instead of any DB-backed queue.

## Features

- Stateless API with no SQLite or persistent storage
- Ollama-first generation flow for `mistral:7b`, `qwen2.5:7b-instruct`, and `llama3.1:8b`
- Embedding-model rejection for chat requests
- In-memory concurrency guard with `429 busy` responses and `Retry-After`
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

Relevant env vars are defined in [.env.example](/Users/aritrarpal/Documents/workspace_biz/llm-comparator/.env.example):

- `OLLAMA_URL`
- `OLLAMA_CHAT_MODELS`
- `OLLAMA_EMBEDDING_MODELS`
- `MAX_CONCURRENT_JOBS`
- `MAX_QUEUE`
- `BUSY_RETRY_AFTER_MS`
- `REQUEST_TIMEOUT_SEC`
- `GROQ_API_KEY`
- `GROQ_MODEL`

## Endpoints

- `GET /health`
- `GET /models`
- `POST /generate/single`

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
    "busy": false
  }
}
```

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
  "error": "busy",
  "retry_after_ms": 2000,
  "meta": {
    "busy": true
  }
}
```

It also includes the header `Retry-After: 2`.

## Minimal test plan

Run the automated tests:

```bash
pytest tests/ -v
```

Manual curl checks:

1. Call `POST /generate/single` with `model = "qwen2.5:7b-instruct"`.
2. Call `POST /generate/single` with `model = "mistral:7b"`.
3. Call `POST /generate/single` with `model = "llama3.1:8b"`.
4. Call `POST /generate/single` with `model = "nomic-embed-text:latest"` and confirm `400`.
5. Fire two requests quickly and confirm the second returns `429` with `Retry-After: 2`.
