#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${QUALITY_MEMORY_DSN:-}" ]]; then
  echo "ERROR: QUALITY_MEMORY_DSN is not set."
  exit 1
fi

DB_DSN="$QUALITY_MEMORY_DSN"
SCHEMA_NAME="${QUALITY_MEMORY_SCHEMA:-contentforge}"
DROP_OLD_DB="${DROP_OLD_QUALITY_MEMORY_DB:-true}"
OLD_DB_NAME="${OLD_QUALITY_MEMORY_DB_NAME:-contentforge}"

echo "Target DSN is configured."
echo "Target schema: ${SCHEMA_NAME}"

echo "Applying schema + quality_runs table..."
psql "$DB_DSN" -v ON_ERROR_STOP=1 <<SQL
CREATE SCHEMA IF NOT EXISTS "${SCHEMA_NAME}";

CREATE TABLE IF NOT EXISTS "${SCHEMA_NAME}".quality_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    theme_name TEXT NOT NULL,
    keywords TEXT[] NOT NULL DEFAULT '{}',
    tone_config JSONB NOT NULL,
    output_spec JSONB NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    output_text TEXT NOT NULL,
    quality_score_json JSONB NOT NULL,
    judge_json JSONB NULL,
    detected_cliches TEXT[] NOT NULL DEFAULT '{}',
    repetition_flags TEXT[] NOT NULL DEFAULT '{}',
    json_leak_flag BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_quality_runs_created_at ON "${SCHEMA_NAME}".quality_runs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quality_runs_theme_name ON "${SCHEMA_NAME}".quality_runs (LOWER(theme_name));
CREATE INDEX IF NOT EXISTS idx_quality_runs_keywords ON "${SCHEMA_NAME}".quality_runs USING GIN (keywords);
SQL

echo "Running insert smoke test (transaction will rollback)..."
psql "$DB_DSN" -v ON_ERROR_STOP=1 <<SQL
BEGIN;
INSERT INTO "${SCHEMA_NAME}".quality_runs (
    run_id,
    theme_name,
    keywords,
    tone_config,
    output_spec,
    backend,
    model,
    output_text,
    quality_score_json,
    judge_json,
    detected_cliches,
    repetition_flags,
    json_leak_flag
)
VALUES (
    'quality-memory-smoke',
    'Warm Wishes',
    ARRAY['family', 'gratitude'],
    '{"tone_funny_pct":20,"tone_emotion_pct":70,"tone_style":"conversational","emoji_policy":"none","audience":"general","avoid_cliches":true}'::jsonb,
    '{"format":"one_liner","structure":{"items":3}}'::jsonb,
    'ollama',
    'qwen2.5:7b-instruct',
    'Quality memory smoke-test output.',
    '{"total":88}'::jsonb,
    NULL,
    ARRAY['shine bright'],
    ARRAY['wishing you'],
    FALSE
);

SELECT run_id, theme_name, backend, model, json_leak_flag
FROM "${SCHEMA_NAME}".quality_runs
WHERE run_id = 'quality-memory-smoke';
ROLLBACK;
SQL

if [[ "${DROP_OLD_DB}" == "true" ]]; then
  echo "Dropping old standalone database (if exists): ${OLD_DB_NAME}"
  dropdb --if-exists --force "${OLD_DB_NAME}" || true
fi

echo "Quality memory setup and insert verification completed."
