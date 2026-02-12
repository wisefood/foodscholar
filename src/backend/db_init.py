"""Database initialization — create schema and tables if they don't exist."""
import logging
from sqlalchemy import text

from backend.postgres import Base, PostgresConnectionSingleton
from models.db import SCHEMA  # noqa: F401 — also registers models on Base.metadata

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create the schema and all tables (no-op if they already exist)."""
    engine = PostgresConnectionSingleton.get_sync_engine()

    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        conn.commit()

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema '%s' and tables verified / created.", SCHEMA)
