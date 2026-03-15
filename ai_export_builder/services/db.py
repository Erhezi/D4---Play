"""Database service — pyodbc connection and parameterized query execution."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyodbc

from ai_export_builder.config import settings

if TYPE_CHECKING:
    from ai_export_builder.services.registry_loader import Registry

logger = logging.getLogger(__name__)


def get_connection() -> pyodbc.Connection:
    """Open a new pyodbc connection using the default configured connection string."""
    return pyodbc.connect(settings.connection_string, timeout=settings.db_timeout)


def get_connection_for_view(view_id: str, registry: "Registry") -> pyodbc.Connection:
    """Open a pyodbc connection routed to the database that owns *view_id*.

    The connection string is read from the environment variable specified in
    connection.yaml (``connection_string_env``).
    """
    conn_str = registry.get_connection_string(view_id)
    return pyodbc.connect(conn_str, timeout=settings.db_timeout)


def execute_query(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Execute a parameterized query against the default connection.

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


def execute_query_for_view(
    view_id: str,
    sql: str,
    registry: "Registry",
    params: list[Any] | None = None,
) -> pd.DataFrame:
    """Execute a parameterized query routed to the database that owns *view_id*.

    All values MUST be passed via *params* (? placeholders) — never
    interpolated into *sql*.
    """
    params = params or []
    logger.info(
        "Executing query for view '%s' [db: %s] (%d params)",
        view_id,
        registry.get_database_key(view_id),
        len(params),
    )
    conn = get_connection_for_view(view_id, registry)
    try:
        df = pd.read_sql(sql, conn, params=params)
        logger.info("Query returned %d rows", len(df))
        return df
    finally:
        conn.close()
