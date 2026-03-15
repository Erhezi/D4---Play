"""Database service — pyodbc connection and parameterized query execution."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pyodbc

from ai_export_builder.config import settings

logger = logging.getLogger(__name__)


def get_connection() -> pyodbc.Connection:
    """Open a new pyodbc connection using the configured connection string."""
    return pyodbc.connect(settings.connection_string, timeout=settings.db_timeout)


def execute_query(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Execute a parameterized query and return results as a DataFrame.

    All values MUST be passed via *params* (? placeholders) — never
    interpolated into *sql*.
    """
    params = params or []
    logger.info("Executing query (%d params)", len(params))
    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn, params=params)
        logger.info("Query returned %d rows", len(df))
        return df
    finally:
        conn.close()
