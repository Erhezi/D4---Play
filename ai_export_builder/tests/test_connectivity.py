"""Diagnostic tests — confirm LLM and DB connectivity with registered views.

Run with:
    python -m pytest ai_export_builder/tests/test_connectivity.py -v

These tests hit real external services (OpenAI API, SQL Server databases).
They are marked with ``pytest.mark.integration`` so you can skip them in CI
via ``pytest -m "not integration"``.
"""
from __future__ import annotations

import os
import logging

import pytest

from ai_export_builder.config import settings
from ai_export_builder.services.openai_client import build_openai_http_client
from ai_export_builder.services.registry_loader import load_registry

logger = logging.getLogger(__name__)
_registry = load_registry()


# ======================================================================
# 1.  LLM connectivity
# ======================================================================


@pytest.mark.integration
class TestLLMConnectivity:
    """Verify the OpenAI API is reachable and responds."""

    def test_api_key_is_configured(self):
        """OPENAI_API_KEY must be present (either env var or .env)."""
        assert settings.openai_api_key, (
            "OPENAI_API_KEY is not set. "
            "Export it as an environment variable or add it to .env."
        )

    def test_openai_simple_response(self):
        """Send a trivial prompt and confirm we get text back."""
        from openai import OpenAI

        with build_openai_http_client() as http_client:
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                http_client=http_client,
            )
            response = client.responses.create(
                model=settings.openai_model,
                input="Reply with the single word: CONNECTED",
            )
        assert response.output_text, "Empty response from OpenAI"
        logger.info("LLM responded: %s", response.output_text[:120])

    def test_openai_json_mode(self):
        """Confirm JSON-mode works (same path as parse_intent)."""
        from openai import OpenAI

        with build_openai_http_client() as http_client:
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                http_client=http_client,
            )
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {"role": "system", "content": "Reply with a JSON object: {\"status\": \"ok\"}"},
                    {"role": "user", "content": "ping"},
                ],
                text={"format": {"type": "json_object"}},
            )
        import json
        data = json.loads(response.output_text)
        assert "status" in data, f"Unexpected JSON: {data}"
        logger.info("LLM JSON-mode response: %s", data)


# ======================================================================
# 2.  Database connectivity per registered view
# ======================================================================


def _view_ids() -> list[str]:
    """Return all view IDs from the registry for parametrized tests."""
    return _registry.all_view_ids()


def _has_connection_env(view_id: str) -> bool:
    """Return True if the env var for this view's database is set."""
    config = _registry.get_connection_config(view_id)
    if not config:
        return False
    env_var = config.get("connection_string_env", "")
    return bool(os.environ.get(env_var))


@pytest.mark.integration
class TestDatabaseConnectivity:
    """Verify each registered view is reachable and queryable."""

    @pytest.mark.parametrize("view_id", _view_ids())
    def test_connection_env_var_documented(self, view_id: str):
        """Each view must map to a database with a connection_string_env entry."""
        config = _registry.get_connection_config(view_id)
        assert config is not None, (
            f"View '{view_id}' has no connection config in connection.yaml"
        )
        env_var = config.get("connection_string_env")
        assert env_var, (
            f"View '{view_id}' connection config is missing 'connection_string_env'"
        )
        logger.info(
            "View '%s' → database '%s' → env var '%s' (set: %s)",
            view_id,
            _registry.get_database_key(view_id),
            env_var,
            bool(os.environ.get(env_var)),
        )

    @pytest.mark.parametrize("view_id", _view_ids())
    def test_view_is_queryable(self, view_id: str):
        """Connect to the DB and SELECT TOP 1 from the view using its schema."""
        if not _has_connection_env(view_id):
            config = _registry.get_connection_config(view_id)
            env_var = config.get("connection_string_env", "?")
            pytest.skip(
                f"Env var '{env_var}' not set — cannot connect to "
                f"database for view '{view_id}'"
            )

        import pyodbc
        import pandas as pd

        view_meta = _registry.get_view_meta(view_id)
        schema = view_meta.get("view_schema", "dbo")
        conn_str = _registry.get_connection_string(view_id)

        conn = pyodbc.connect(conn_str, timeout=settings.db_timeout)
        try:
            sql = f"SELECT TOP 1 * FROM {schema}.{view_id}"
            df = pd.read_sql(sql, conn)
            assert len(df) >= 0, "Query should return at least zero rows"
            logger.info(
                "View '%s' (%s.%s) — returned %d cols: %s",
                view_id,
                schema,
                view_id,
                len(df.columns),
                list(df.columns),
            )
        finally:
            conn.close()

    @pytest.mark.parametrize("view_id", _view_ids())
    def test_registered_columns_exist_in_view(self, view_id: str):
        """Every column declared in registry_views.yaml must exist in the actual DB view."""
        if not _has_connection_env(view_id):
            config = _registry.get_connection_config(view_id)
            env_var = config.get("connection_string_env", "?")
            pytest.skip(
                f"Env var '{env_var}' not set — cannot connect to "
                f"database for view '{view_id}'"
            )

        import pyodbc
        import pandas as pd

        view_meta = _registry.get_view_meta(view_id)
        schema = view_meta.get("view_schema", "dbo")
        conn_str = _registry.get_connection_string(view_id)

        conn = pyodbc.connect(conn_str, timeout=settings.db_timeout)
        try:
            # Fetch just the column names (no rows needed)
            sql = f"SELECT TOP 0 * FROM {schema}.{view_id}"
            df = pd.read_sql(sql, conn)
            actual_columns = {c.upper() for c in df.columns}
        finally:
            conn.close()

        registered_columns = _registry.get_all_columns(view_id)
        missing = [
            c for c in registered_columns if c.upper() not in actual_columns
        ]
        assert not missing, (
            f"View '{view_id}' is missing registered columns: {missing}. "
            f"Actual columns: {sorted(actual_columns)}"
        )
        logger.info(
            "View '%s': all %d registered columns found in DB",
            view_id,
            len(registered_columns),
        )
