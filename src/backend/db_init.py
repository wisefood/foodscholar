"""Database initialization — create schema and tables if they don't exist."""
import logging
from sqlalchemy import text

from backend.postgres import Base, PostgresConnectionSingleton
from models.db import SCHEMA  # noqa: F401 — also registers models on Base.metadata

logger = logging.getLogger(__name__)


def _apply_schema_updates(conn) -> None:
    """Apply idempotent schema updates for existing deployments."""
    statements = [
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS helpfulness VARCHAR(24)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS target_answer VARCHAR(16)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS feedback_mode VARCHAR(24)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ALTER COLUMN preferred_answer DROP NOT NULL
        """,
        f"""
        UPDATE {SCHEMA}.qa_feedback
        SET target_answer = 'overall'
        WHERE target_answer IS NULL
        """,
        f"""
        UPDATE {SCHEMA}.qa_feedback
        SET feedback_mode = 'ab_preference'
        WHERE feedback_mode IS NULL
          AND preferred_answer IN ('a', 'b', 'neither', 'both')
        """,
        f"""
        UPDATE {SCHEMA}.qa_feedback
        SET feedback_mode = 'general'
        WHERE feedback_mode IS NULL
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ALTER COLUMN target_answer SET DEFAULT 'overall'
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ALTER COLUMN target_answer SET NOT NULL
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ALTER COLUMN feedback_mode SET DEFAULT 'general'
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ALTER COLUMN feedback_mode SET NOT NULL
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_requests
        ADD COLUMN IF NOT EXISTS pipeline_meta JSONB
        """,
        # Platform correlation id (the gateway's X-Request-Id), and identity on
        # feedback rows. Both are nullable: every row written before this
        # migration predates the correlation id, and direct callers may still
        # arrive without one.
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_requests
        ADD COLUMN IF NOT EXISTS correlation_id VARCHAR(64)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_qa_requests_correlation_id
        ON {SCHEMA}.qa_requests (correlation_id)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS member_id VARCHAR(255)
        """,
        f"""
        ALTER TABLE IF EXISTS {SCHEMA}.qa_feedback
        ADD COLUMN IF NOT EXISTS correlation_id VARCHAR(64)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_qa_feedback_user_id
        ON {SCHEMA}.qa_feedback (user_id)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS ix_qa_feedback_correlation_id
        ON {SCHEMA}.qa_feedback (correlation_id)
        """,
    ]

    for stmt in statements:
        conn.execute(text(stmt))


def init_db() -> None:
    """Create the schema and all tables (no-op if they already exist)."""
    engine = PostgresConnectionSingleton.get_sync_engine()

    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        Base.metadata.create_all(bind=conn)
        _apply_schema_updates(conn)
        conn.commit()

    logger.info(
        "Database schema '%s' and tables verified / created (with updates).",
        SCHEMA,
    )
