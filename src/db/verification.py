"""Startup/runtime verification helpers for required Postgres tables."""

from __future__ import annotations

import logging
from typing import Any

from src.db.constants import DB_SCHEMA, QUALITY_RUNS_TABLE, QUALITY_RUNS_TABLE_NAME

logger = logging.getLogger(__name__)

CHECK_QUALITY_RUNS_TABLE_SQL = """
SELECT 1
FROM information_schema.tables
WHERE table_schema = %s
  AND table_name = %s
LIMIT 1
"""


def verify_quality_runs_table_exists(cursor: Any) -> None:
    """Raise RuntimeError when the required quality-runs table is missing."""

    cursor.execute(CHECK_QUALITY_RUNS_TABLE_SQL, (DB_SCHEMA, QUALITY_RUNS_TABLE_NAME))
    exists = cursor.fetchone() is not None
    if exists:
        return

    logger.critical(
        "quality_memory_table_missing schema=%s table=%s",
        DB_SCHEMA,
        QUALITY_RUNS_TABLE_NAME,
    )
    raise RuntimeError(f"Required table missing: {QUALITY_RUNS_TABLE}")

