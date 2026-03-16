"""LangGraph node: deterministic validation of ExportIntent against the registry."""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterOperator
from ai_export_builder.services.registry_loader import Registry, load_registry

logger = logging.getLogger(__name__)

_registry = load_registry()

_VALID_OPERATORS = {op.value for op in FilterOperator}


def node_validate_intent(state: ExportState) -> dict[str, Any]:
    """Validate intent against registry. Returns validation_errors list (empty = pass)."""
    intent = state.get("intent")
    errors: list[str] = []

    if intent is None:
        errors.append("No intent produced by the parser.")
        return {"validation_errors": errors, "status": "failed"}

    # 1. View exists
    if not _registry.view_exists(intent.selected_view):
        errors.append(
            f"View '{intent.selected_view}' is not in the registry. "
            f"Valid views: {_registry.all_view_ids()}"
        )
        # Cannot validate columns/filters if view is unknown
        return {"validation_errors": errors}

    valid_columns = set(_registry.get_all_columns(intent.selected_view))

    # 2. All requested columns exist in the view
    for col in intent.columns:
        if col not in valid_columns:
            errors.append(
                f"Column '{col}' does not exist in view '{intent.selected_view}'. "
                f"Valid columns: {sorted(valid_columns)}"
            )

    # 3. Filter columns exist and operators are valid
    for i, f in enumerate(intent.filters):
        if f.column not in valid_columns:
            errors.append(
                f"Filter #{i+1}: column '{f.column}' does not exist in view "
                f"'{intent.selected_view}'."
            )
        if f.operator.value not in _VALID_OPERATORS:
            errors.append(
                f"Filter #{i+1}: operator '{f.operator.value}' is not allowed. "
                f"Valid operators: {sorted(_VALID_OPERATORS)}"
            )

        # 4. Date values well-formed (for between operator)
        if f.operator == FilterOperator.between:
            values = f.value if isinstance(f.value, list) else [f.value]
            if len(values) != 2:
                errors.append(
                    f"Filter #{i+1}: 'between' requires exactly 2 values, got {len(values)}."
                )
            for v in values:
                if not _is_valid_date_string(v):
                    # Not necessarily an error — could be numeric between
                    col_meta = _registry.get_column_meta(intent.selected_view, f.column)
                    if col_meta and col_meta.get("type") == "date":
                        errors.append(
                            f"Filter #{i+1}: value '{v}' is not a valid date (expected YYYY-MM-DD)."
                        )

        # 5. 'in' operator should have a list value
        if f.operator == FilterOperator.in_:
            if not isinstance(f.value, list):
                errors.append(
                    f"Filter #{i+1}: 'in' operator expects a list of values."
                )

    if errors:
        logger.warning("node_validate_intent: %d errors found", len(errors))
        for e in errors:
            logger.warning("  - %s", e)
    else:
        logger.info("node_validate_intent: intent passed validation")
        # Resolve columns: basic groups + LLM-selected + filter cols + companions
        _resolve_columns(intent, _registry)

    return {"validation_errors": errors}


def _resolve_columns(intent: ExportIntent, registry: Registry) -> None:
    """Deterministically expand intent.columns to include basic group columns,
    filter columns, and companion (text↔ID) columns. Mutates intent in place."""
    view_id = intent.selected_view
    basic_cols = registry.get_basic_columns(view_id)

    # Ordered set: basic first, then LLM-selected, then filter-added
    seen: set[str] = set()
    resolved: list[str] = []

    def _add(col: str) -> None:
        if col not in seen:
            seen.add(col)
            resolved.append(col)

    # 1. Always include basic group columns
    for c in basic_cols:
        _add(c)

    # 2. Add LLM-selected columns (may include enrichment cols)
    for c in intent.columns:
        _add(c)

    # 3. Add filter columns to output
    for f in intent.filters:
        _add(f.column)

    # 4. Add companion columns for all currently selected columns
    snapshot = list(resolved)
    for c in snapshot:
        companion = registry.get_companion_column(view_id, c)
        if companion:
            _add(companion)

    intent.columns = resolved
    logger.info("_resolve_columns: resolved %d columns for %s", len(resolved), view_id)


def _is_valid_date_string(val: str) -> bool:
    """Check if a string looks like a valid ISO date (YYYY-MM-DD)."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(val).strip()))
