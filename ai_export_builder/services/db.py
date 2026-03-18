"""Database service — SQLAlchemy engine and parameterized query execution."""

from __future__ import annotations

import logging
import urllib.parse
from typing import TYPE_CHECKING, Any

import pandas as pd
from sqlalchemy import create_engine, text

from ai_export_builder.config import settings

if TYPE_CHECKING:
    from ai_export_builder.services.registry_loader import Registry

logger = logging.getLogger(__name__)


def _build_engine_url(conn_str: str) -> str:
    """Convert an ODBC connection string to a SQLAlchemy URL."""
    return f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn_str)}"


def _positional_to_named(sql: str, params: list[Any]) -> tuple[str, dict[str, Any]]:
    """Replace positional ``?`` placeholders with ``:p0, :p1, …`` named params.

    Returns (new_sql, params_dict).
    """
    named: dict[str, Any] = {}
    idx = 0
    parts: list[str] = []
    for ch in sql:
        if ch == "?":
            key = f"p{idx}"
            parts.append(f":{key}")
            named[key] = params[idx] if idx < len(params) else None
            idx += 1
        else:
            parts.append(ch)
    return "".join(parts), named


def execute_query(sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Execute a parameterized query against the default connection.

    All values MUST be passed via *params* (? placeholders) — never
    interpolated into *sql*.
    """
    params = params or []
    logger.info("Executing query (%d params)", len(params))
    engine = create_engine(_build_engine_url(settings.connection_string))
    named_sql, named_params = _positional_to_named(sql, params)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(named_sql), conn, params=named_params)
        logger.info("Query returned %d rows", len(df))
        return df
    finally:
        engine.dispose()


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
    conn_str = registry.get_connection_string(view_id)
    engine = create_engine(_build_engine_url(conn_str))
    named_sql, named_params = _positional_to_named(sql, params)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(named_sql), conn, params=named_params)
        logger.info("Query returned %d rows", len(df))
        return df
    finally:
        engine.dispose()
