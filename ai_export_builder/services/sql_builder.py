"""Build parameterized SQL from an ExportIntent using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ai_export_builder.config import settings
from ai_export_builder.models.intent import ExportIntent, FilterOperator, SortItem

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,          # SQL, not HTML
    keep_trailing_newline=True,
)

_OP_MAP: dict[FilterOperator, str] = {
    FilterOperator.eq: "=",
    FilterOperator.neq: "<>",
    FilterOperator.gt: ">",
    FilterOperator.gte: ">=",
    FilterOperator.lt: "<",
    FilterOperator.lte: "<=",
}


def build_query(
    intent: ExportIntent,
    user_facilities: list[str] | None = None,
    max_rows: int | None = None,
) -> tuple[str, list[Any]]:
    """Return (sql_string, params_list) with ? placeholders for pyodbc.

    All values are parameterized — nothing is string-interpolated into the SQL.
    """
    facilities = user_facilities or settings.user_facilities
    row_limit = max_rows or settings.max_export_rows

    params: list[Any] = []
    filter_contexts: list[dict[str, Any]] = []

    for f in intent.filters:
        ctx: dict[str, Any] = {
            "column": f.column,
            "operator": f.operator.value,
        }
        if f.operator == FilterOperator.between:
            values = f.value if isinstance(f.value, list) else [f.value, f.value]
            params.extend(values[:2])
        elif f.operator == FilterOperator.in_:
            values = f.value if isinstance(f.value, list) else [f.value]
            ctx["placeholders"] = ", ".join("?" for _ in values)
            params.extend(values)
        elif f.operator == FilterOperator.like:
            if isinstance(f.value, list):
                # Multi-value LIKE → within-field OR group
                ctx["or_group"] = True
                ctx["placeholders_count"] = len(f.value)
                for v in f.value:
                    raw = str(v).strip("%")
                    params.append(f"%{raw}%")
            else:
                raw = f.value if isinstance(f.value, str) else str(f.value)
                raw = raw.strip("%")
                params.append(f"%{raw}%")
        else:
            ctx["sql_op"] = _OP_MAP.get(f.operator, "=")
            params.append(f.value)

        filter_contexts.append(ctx)

    # Build sort context from intent.sort_by
    sort_contexts: list[dict[str, str]] = []
    if intent.sort_by:
        view_columns = {c.lower() for c in intent.columns}
        for s in intent.sort_by:
            # Only include sort columns that are in the selected columns
            if s.column.lower() in view_columns or s.column in intent.columns:
                sort_contexts.append({"column": s.column, "direction": s.direction})

    # RLS
    rls_all = "ALL" in facilities
    rls_placeholders = ""
    if not rls_all:
        rls_placeholders = ", ".join("?" for _ in facilities)
        params.extend(facilities)

    # Row limit (always last param)
    params.append(row_limit)

    template = _env.get_template("select_query.sql.j2")
    sql = template.render(
        view=intent.selected_view,
        columns=intent.columns,
        filters=filter_contexts,
        sort_columns=sort_contexts,
        rls_column="FacilityCode",
        rls_values=facilities,
        rls_all=rls_all,
        rls_placeholders=rls_placeholders,
    )

    return sql.strip(), params


def build_disambiguation_query(
    view_id: str,
    text_col: str,
    id_col: str,
    like_value: str,
    user_facilities: list[str] | None = None,
    max_rows: int = 50,
) -> tuple[str, list[Any]]:
    """Return (sql, params) for a SELECT DISTINCT preview of text↔ID pairs.

    Used to show the user which entities match a LIKE filter before the
    full export runs.  All values are parameterized.
    """
    facilities = user_facilities or settings.user_facilities
    params: list[Any] = [like_value]

    rls_clause = ""
    if "ALL" not in facilities:
        rls_placeholders = ", ".join("?" for _ in facilities)
        rls_clause = f"AND [FacilityCode] IN ({rls_placeholders})"
        params.extend(facilities)

    sql = (
        f"SELECT DISTINCT TOP {int(max_rows)} [{text_col}], [{id_col}]\n"
        f"FROM [{view_id}]\n"
        f"WHERE [{text_col}] LIKE ?\n"
        f"{rls_clause}\n"
        f"ORDER BY [{text_col}]"
    )
    return sql.strip(), params


def build_aggregation_query(
    intent: ExportIntent,
    sum_check_columns: list[str],
    user_facilities: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Return (sql, params) for an aggregation summary query.

    Generates ``SELECT COUNT(*) AS row_count, SUM([col]) AS total_col ...``
    using the same WHERE filters as the main export but with no row limit.
    The *sum_check_columns* are the measure columns marked ``sum_check: true``
    in the registry.
    """
    facilities = user_facilities or settings.user_facilities
    params: list[Any] = []
    filter_clauses: list[str] = []

    for f in intent.filters:
        if f.operator == FilterOperator.between:
            values = f.value if isinstance(f.value, list) else [f.value, f.value]
            filter_clauses.append(f"AND [{f.column}] BETWEEN ? AND ?")
            params.extend(values[:2])
        elif f.operator == FilterOperator.in_:
            values = f.value if isinstance(f.value, list) else [f.value]
            placeholders = ", ".join("?" for _ in values)
            filter_clauses.append(f"AND [{f.column}] IN ({placeholders})")
            params.extend(values)
        elif f.operator == FilterOperator.like:
            if isinstance(f.value, list):
                or_parts = " OR ".join(f"[{f.column}] LIKE ?" for _ in f.value)
                filter_clauses.append(f"AND ({or_parts})")
                for v in f.value:
                    raw = str(v).strip("%")
                    params.append(f"%{raw}%")
            else:
                raw = f.value if isinstance(f.value, str) else str(f.value)
                raw = raw.strip("%")
                filter_clauses.append(f"AND [{f.column}] LIKE ?")
                params.append(f"%{raw}%")
        else:
            sql_op = _OP_MAP.get(f.operator, "=")
            filter_clauses.append(f"AND [{f.column}] {sql_op} ?")
            params.append(f.value)

    # RLS
    if "ALL" not in facilities:
        rls_placeholders = ", ".join("?" for _ in facilities)
        filter_clauses.append(f"AND [FacilityCode] IN ({rls_placeholders})")
        params.extend(facilities)

    # Build SELECT list
    select_parts = ["COUNT(*) AS row_count"]
    for col in sum_check_columns:
        safe_alias = col.replace(" ", "_")
        select_parts.append(f"SUM([{col}]) AS total_{safe_alias}")

    select_list = ", ".join(select_parts)
    where_block = "\n    ".join(filter_clauses)

    sql = (
        f"SELECT {select_list}\n"
        f"FROM [{intent.selected_view}]\n"
        f"WHERE 1=1\n"
        f"    {where_block}"
    ).strip()

    return sql, params
