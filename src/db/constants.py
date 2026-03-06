"""Database naming constants used by runtime Postgres access."""

# Single source of truth for DB schema/table names. Update here if schema changes in future.
DB_SCHEMA = "contentforge"
QUALITY_RUNS_TABLE_NAME = "quality_runs"
QUALITY_RUNS_TABLE = f"{DB_SCHEMA}.{QUALITY_RUNS_TABLE_NAME}"

