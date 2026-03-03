# LLM Comparator

Standalone FastAPI application for running the same ecard prompt against multiple LLM backends and storing the outputs for side-by-side comparison.

## Features

- FastAPI API with async SQLite persistence
- Concurrent comparison across configurable Groq and Ollama backends
- Stored results with run summaries, per-run detail, CSV export, and aggregate stats
- No PostgreSQL or external migrations required

## Setup

```bash
cd llm-comparator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env` if you want the Groq backend enabled. Without it, `/generate/compare` defaults to the configured Ollama backends for local runs.

## Run

```bash
uvicorn app.main:app --reload
```

The app creates `llm_comparator.db` automatically on startup.

## Test

```bash
pytest tests/ -v
```

## Endpoints

### Root

```bash
curl http://127.0.0.1:8000/
```

### Compare multiple backends

```bash
curl -X POST http://127.0.0.1:8000/generate/compare \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Motivational Monday",
    "tone_funny_pct": 30,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["strength", "monday", "energy"],
    "visual_style": "minimal sunrise",
    "count": 3
  }'
```

### Run one backend only

```bash
curl -X POST http://127.0.0.1:8000/generate/single \
  -H "Content-Type: application/json" \
  -d '{
    "theme_name": "Warm Wishes",
    "tone_funny_pct": 20,
    "tone_emotion_pct": 70,
    "prompt_keywords": ["family", "gratitude"],
    "visual_style": "soft watercolor",
    "backend": "groq",
    "count": 3
  }'
```

### List stored runs

```bash
curl http://127.0.0.1:8000/results/runs
```

### View one run

```bash
curl http://127.0.0.1:8000/results/runs/<run_id>
```

### Export one run as CSV

```bash
curl -OJ http://127.0.0.1:8000/results/runs/<run_id>/export
```

### Aggregate stats

```bash
curl http://127.0.0.1:8000/results/stats
```
