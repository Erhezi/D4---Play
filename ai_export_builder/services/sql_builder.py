"""Build parameterized SQL from an ExportIntent using Jinja2 templates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ai_export_builder.config import settings
from ai_export_builder.models.intent import ExportIntent, FilterOperator

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
            params.append(f.value)
        else:
            ctx["sql_op"] = _OP_MAP.get(f.operator, "=")
            params.append(f.value)

        filter_contexts.append(ctx)

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
        rls_column="FacilityCode",
        rls_values=facilities,
        rls_all=rls_all,
        rls_placeholders=rls_placeholders,
    )

    return sql.strip(), params
