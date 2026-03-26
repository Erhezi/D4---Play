"""Template-based intent summarizer — produces plain-English descriptions of ExportIntent."""

from __future__ import annotations

from ai_export_builder.models.intent import ExportIntent, FilterOperator
from ai_export_builder.services.registry_loader import Registry


def summarize_intent(intent: ExportIntent, registry: Registry) -> str:
    """Return a one-sentence plain-English summary of an ExportIntent."""
    # View name
    view_meta = registry.get_view_meta(intent.selected_view)
    view_name = (view_meta or {}).get("display_name", intent.selected_view)

    # Filter descriptions
    filter_parts: list[str] = []
    for f in intent.filters:
        col_meta = registry.get_column_meta(intent.selected_view, f.column)
        col_label = (col_meta or {}).get("label", f.column)
        if f.operator == FilterOperator.like:
            vals = f.value if isinstance(f.value, list) else [f.value]
            filter_parts.append(f"{col_label} matching {_quote_vals(vals)}")
        elif f.operator == FilterOperator.between:
            vals = f.value if isinstance(f.value, list) else [f.value]
            if len(vals) >= 2:
                filter_parts.append(f"{col_label} from {vals[0]} to {vals[1]}")
            else:
                filter_parts.append(f"{col_label} = {vals[0]}")
        elif f.operator == FilterOperator.in_:
            vals = f.value if isinstance(f.value, list) else [f.value]
            filter_parts.append(f"{col_label} in {_quote_vals(vals)}")
        elif f.operator in (FilterOperator.eq,):
            filter_parts.append(f"{col_label} = {_quote_val(f.value)}")
        else:
            filter_parts.append(f"{col_label} {f.operator.value} {_quote_val(f.value)}")

    # Column enrichment description
    enrichment_parts: list[str] = []
    if intent.columns:
        # Group by field groups if possible
        field_groups = (view_meta or {}).get("field_groups", [])
        enrichment_group_names: list[str] = []
        ungrouped_cols: list[str] = list(intent.columns)
        for fg in field_groups:
            if fg.get("group_type") in ("basic", "core"):
                continue
            group_cols = set(fg.get("columns_included", []))
            if group_cols & set(intent.columns):
                enrichment_group_names.append(fg.get("group_name", "additional columns"))
                for c in group_cols:
                    if c in ungrouped_cols:
                        ungrouped_cols.remove(c)
        if enrichment_group_names:
            enrichment_parts.append(", ".join(enrichment_group_names))

    # Assemble
    parts: list[str] = [f"**{view_name}**"]
    if filter_parts:
        parts.append("filtered by " + ", ".join(filter_parts))
    if enrichment_parts:
        parts.append("including " + " and ".join(enrichment_parts))

    return " — ".join(parts)


def _quote_val(v) -> str:
    if isinstance(v, list):
        return ", ".join(f'"{x}"' for x in v)
    return f'"{v}"'


def _quote_vals(vals: list) -> str:
    if len(vals) == 1:
        return f'"{vals[0]}"'
    return ", ".join(f'"{v}"' for v in vals)
