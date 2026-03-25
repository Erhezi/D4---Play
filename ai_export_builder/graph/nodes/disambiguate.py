"""LangGraph node: run SELECT DISTINCT previews for LIKE/eq filters on text↔ID paired columns."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import FilterOperator
from ai_export_builder.services.db import execute_query_for_view
from ai_export_builder.services.registry_loader import load_registry
from ai_export_builder.services.sql_builder import build_disambiguation_query

logger = logging.getLogger(__name__)

_registry = load_registry()


def node_disambiguate(state: ExportState) -> dict[str, Any]:
    """For each LIKE or eq filter on a companion-paired text column, run a
    SELECT DISTINCT preview so the user can confirm which entities to include.

    Sets ``disambiguation_needed=True`` when at least one preview has results.
    """
    intent = state.get("intent")
    if intent is None:
        return {"disambiguation_needed": False, "disambiguation_results": []}

    profile = state.get("user_profile", {})
    facilities = profile.get("facilities", settings.user_facilities)

    disambiguable = _registry.get_disambiguable_columns(intent.selected_view)
    # disambiguable: {text_col: id_col}

    results: list[dict[str, Any]] = []

    for f in intent.filters:
        if f.column not in disambiguable:
            continue
        if f.operator not in (FilterOperator.like, FilterOperator.eq):
            continue

        text_col = f.column
        id_col = disambiguable[text_col]

        # Collect the values to disambiguate (may be a list for multi-value LIKE)
        raw_values: list[str]
        if isinstance(f.value, list):
            raw_values = [str(v) for v in f.value]
        else:
            raw_values = [str(f.value)]

        all_matches: list[dict[str, str]] = []
        for raw_val in raw_values:
            like_value = raw_val.strip("%")
            like_value = f"%{like_value}%"

            try:
                sql, params = build_disambiguation_query(
                    view_id=intent.selected_view,
                    text_col=text_col,
                    id_col=id_col,
                    like_value=like_value,
                    user_facilities=facilities,
                )
                logger.info(
                    "node_disambiguate: previewing %s LIKE %s (%d params)",
                    text_col, like_value, len(params),
                )
                df = execute_query_for_view(
                    view_id=intent.selected_view,
                    sql=sql,
                    registry=_registry,
                    params=params,
                )
                matches = [
                    {"text": row[text_col], "id": row[id_col]}
                    for _, row in df.iterrows()
                ]
                all_matches.extend(matches)
                logger.info(
                    "node_disambiguate: %s LIKE %s returned %d distinct matches",
                    text_col, like_value, len(matches),
                )
            except Exception as exc:
                logger.error("node_disambiguate: disambiguation query failed — %s", exc)
                continue

        # Deduplicate matches by (text, id) pair
        seen: set[tuple[str, str]] = set()
        unique_matches: list[dict[str, str]] = []
        for m in all_matches:
            key = (str(m["text"]), str(m["id"]))
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        results.append({
            "column": text_col,
            "companion": id_col,
            "original_operator": f.operator.value,
            "original_value": f.value,
            "matches": unique_matches,
        })

    needed = any(r["matches"] for r in results)
    return {
        "disambiguation_needed": needed,
        "disambiguation_results": results,
        "status": "disambiguating" if needed else state.get("status", "parsing"),
    }
