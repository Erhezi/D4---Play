"""LangGraph node: fetch a 20-row preview and aggregation summary."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState
from ai_export_builder.services.db import execute_query_for_view
from ai_export_builder.services.registry_loader import load_registry
from ai_export_builder.services.sql_builder import build_aggregation_query, build_query

logger = logging.getLogger(__name__)

_registry = load_registry()

PREVIEW_ROW_LIMIT = 20


def node_hydrate_preview(state: ExportState) -> dict[str, Any]:
    """Build a 20-row preview and run the aggregation summary query.

    Returns preview_data (list of dicts) and aggregation_summary
    (row_count + sum_check column totals) for display on the
    verification card.
    """
    intent = state.get("intent")
    if intent is None:
        return {
            "preview_data": None,
            "aggregation_summary": None,
            "status": "failed",
        }

    profile = state.get("user_profile", {})
    facilities = profile.get("facilities", settings.user_facilities)

    preview_data: list[dict[str, Any]] | None = None
    aggregation_summary: dict[str, Any] | None = None

    # --- 20-row preview ---
    try:
        sql, params = build_query(
            intent,
            user_facilities=facilities,
            max_rows=PREVIEW_ROW_LIMIT,
        )
        logger.info("node_hydrate_preview: fetching %d-row preview", PREVIEW_ROW_LIMIT)
        df = execute_query_for_view(
            view_id=intent.selected_view,
            sql=sql,
            registry=_registry,
            params=params,
        )
        preview_data = df.to_dict(orient="records")
        logger.info("node_hydrate_preview: preview returned %d rows", len(preview_data))
    except Exception as exc:
        logger.error("node_hydrate_preview: preview query failed — %s", exc)
        # Non-fatal: continue without preview

    # --- Aggregation summary ---
    try:
        sum_check_cols = _registry.get_sum_check_columns(intent.selected_view)
        agg_sql, agg_params = build_aggregation_query(
            intent,
            sum_check_columns=sum_check_cols,
            user_facilities=facilities,
        )
        logger.info("node_hydrate_preview: running aggregation query")
        agg_df = execute_query_for_view(
            view_id=intent.selected_view,
            sql=agg_sql,
            registry=_registry,
            params=agg_params,
        )
        if len(agg_df) > 0:
            row = agg_df.iloc[0]
            sums: dict[str, float] = {}
            for col in sum_check_cols:
                safe_alias = f"total_{col.replace(' ', '_')}"
                if safe_alias in row.index:
                    sums[col] = float(row[safe_alias]) if row[safe_alias] is not None else 0.0
            aggregation_summary = {
                "row_count": int(row["row_count"]),
                "sums": sums,
            }
            logger.info(
                "node_hydrate_preview: aggregation — %d rows, sums=%s",
                aggregation_summary["row_count"], sums,
            )
    except Exception as exc:
        logger.error("node_hydrate_preview: aggregation query failed — %s", exc)
        # Non-fatal: continue without aggregation

    return {
        "preview_data": preview_data,
        "aggregation_summary": aggregation_summary,
        "status": "previewing",
    }
