"""LangGraph node: generate SQL, inject RLS, execute query, return DataFrame."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState
from ai_export_builder.services.db import execute_query_for_view
from ai_export_builder.services.registry_loader import load_registry
from ai_export_builder.services.sql_builder import build_query

logger = logging.getLogger(__name__)

_registry = load_registry()


def node_execute_export(state: ExportState) -> dict[str, Any]:
    """Build parameterized SQL from the validated intent, execute it, return results."""
    intent = state.get("intent")
    if intent is None:
        return {
            "status": "failed",
            "error_message": "Cannot execute — no valid intent.",
            "result_df": None,
            "result_row_count": 0,
        }

    profile = state.get("user_profile", {})
    facilities = profile.get("facilities", settings.user_facilities)

    logger.info(
        "node_execute_export: building SQL for view '%s' (%d cols, %d filters)",
        intent.selected_view,
        len(intent.columns),
        len(intent.filters),
    )

    try:
        sql, params = build_query(
            intent,
            user_facilities=facilities,
            max_rows=settings.max_export_rows,
        )
        logger.info("node_execute_export: SQL built (%d params)", len(params))

        df = execute_query_for_view(
            view_id=intent.selected_view,
            sql=sql,
            registry=_registry,
            params=params,
        )

        row_count = len(df)
        logger.info("node_execute_export: query returned %d rows", row_count)

        return {
            "status": "completed",
            "result_df": df,
            "result_row_count": row_count,
            "error_message": "",
        }
    except Exception as exc:
        logger.error("node_execute_export: execution failed — %s", exc)
        return {
            "status": "failed",
            "error_message": str(exc),
            "result_df": None,
            "result_row_count": 0,
        }
