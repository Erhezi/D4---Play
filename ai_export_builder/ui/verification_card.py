"""Streamlit verification card — editable intent review before SQL execution."""
from __future__ import annotations

from typing import Any

import streamlit as st

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.registry_loader import Registry


def render_verification_card(
    intent: ExportIntent,
    registry: Registry,
    validation_errors: list[str] | None = None,
) -> tuple[bool, ExportIntent | None]:
    """Render the verification card and return (confirmed, edited_intent | None).

    Returns ``(True, edited_intent)`` when the user clicks *Confirm*.
    Returns ``(False, edited_intent)`` when the user clicks *Edit & Resubmit*.
    Returns ``(False, None)`` while the user hasn't acted yet.
    """
    st.subheader("📋 Verify Your Export")

    # --- Validation warnings ---
    errors = validation_errors or []
    if errors:
        for err in errors:
            st.error(err)

    if intent.warnings:
        for w in intent.warnings:
            st.warning(w)

    # --- View info ---
    view_meta = registry.get_view_meta(intent.selected_view)
    display_name = (view_meta or {}).get("display_name", intent.selected_view)
    st.markdown(f"**View:** {display_name} (`{intent.selected_view}`)")

    # --- Column selection (grouped by field groups, companions shown together) ---
    st.markdown("**Columns:**")

    basic_cols = set(registry.get_basic_columns(intent.selected_view))
    all_columns = registry.get_all_columns(intent.selected_view)
    # Track which columns have a companion so we can render them on the same line
    companion_rendered: set[str] = set()
    selected_columns: list[str] = []

    # Group: Basic columns (always included, non-removable)
    field_groups = (view_meta or {}).get("field_groups", [])
    for fg in field_groups:
        group_name = fg.get("group_name", "Columns")
        group_type = fg.get("group_type", "")
        group_cols = fg.get("columns_included", [])
        is_basic = group_type == "basic"

        st.markdown(f"*{group_name}* {'(always included)' if is_basic else ''}")

        cols_per_row = 3
        for i in range(0, len(group_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_widget in enumerate(row_cols):
                idx = i + j
                if idx >= len(group_cols):
                    break
                col_name = group_cols[idx]
                if col_name in companion_rendered:
                    continue
                col_meta = registry.get_column_meta(intent.selected_view, col_name) or {}
                label = col_meta.get("label", col_name)

                # Show companion pair together
                companion = registry.get_companion_column(intent.selected_view, col_name)
                if companion and companion in group_cols:
                    comp_meta = registry.get_column_meta(intent.selected_view, companion) or {}
                    comp_label = comp_meta.get("label", companion)
                    display_label = f"{label} ({col_name}) + {comp_label} ({companion})"
                    companion_rendered.add(companion)
                else:
                    display_label = f"{label} ({col_name})"

                checked = col_name in intent.columns
                if is_basic:
                    col_widget.checkbox(display_label, value=True, disabled=True, key=f"col_{col_name}")
                    selected_columns.append(col_name)
                    if companion and companion in group_cols:
                        selected_columns.append(companion)
                else:
                    if col_widget.checkbox(display_label, value=checked, key=f"col_{col_name}"):
                        selected_columns.append(col_name)
                        if companion and companion in group_cols:
                            selected_columns.append(companion)

    # Any remaining columns not in any field group
    grouped_cols = set()
    for fg in field_groups:
        grouped_cols.update(fg.get("columns_included", []))
    ungrouped = [c for c in all_columns if c not in grouped_cols and c not in companion_rendered]

    if ungrouped:
        st.markdown("*Other Columns*")
        cols_per_row = 3
        for i in range(0, len(ungrouped), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_widget in enumerate(row_cols):
                idx = i + j
                if idx >= len(ungrouped):
                    break
                col_name = ungrouped[idx]
                col_meta = registry.get_column_meta(intent.selected_view, col_name) or {}
                label = col_meta.get("label", col_name)
                companion = registry.get_companion_column(intent.selected_view, col_name)
                if companion:
                    comp_meta = registry.get_column_meta(intent.selected_view, companion) or {}
                    display_label = f"{label} ({col_name}) + {comp_meta.get('label', companion)} ({companion})"
                else:
                    display_label = f"{label} ({col_name})"
                checked = col_name in intent.columns
                if col_widget.checkbox(display_label, value=checked, key=f"col_{col_name}"):
                    selected_columns.append(col_name)
                    if companion:
                        selected_columns.append(companion)

    # --- Filters (editable rows) ---
    st.markdown("**Filters:**")
    edited_filters: list[FilterItem] = []

    if not intent.filters:
        st.info("No filters applied.")
    else:
        for i, f in enumerate(intent.filters):
            with st.expander(f"Filter {i + 1}: {f.column} {f.operator.value} {f.value}", expanded=True):
                fcol1, fcol2, fcol3 = st.columns([2, 1, 3])

                col_options = all_columns if all_columns else [f.column]
                col_idx = col_options.index(f.column) if f.column in col_options else 0
                new_col = fcol1.selectbox(
                    "Column", col_options, index=col_idx, key=f"filter_col_{i}"
                )

                op_options = [op.value for op in FilterOperator]
                op_idx = op_options.index(f.operator.value) if f.operator.value in op_options else 0
                new_op = fcol2.selectbox(
                    "Operator", op_options, index=op_idx, key=f"filter_op_{i}"
                )

                current_val = (
                    ", ".join(f.value) if isinstance(f.value, list) else str(f.value)
                )
                # Strip '%' wrappers for LIKE filters — they are added automatically in the query
                if f.operator == FilterOperator.like:
                    current_val = current_val.strip("%")
                new_val_str = fcol3.text_input(
                    "Value(s)", value=current_val, key=f"filter_val_{i}",
                    help="For 'between': start, end. For 'in': comma-separated values. For 'like': search term (% added automatically).",
                )

                # Parse the value back
                new_op_enum = FilterOperator(new_op)
                if new_op_enum in (FilterOperator.between, FilterOperator.in_):
                    parsed_val: str | list[str] = [
                        v.strip() for v in new_val_str.split(",") if v.strip()
                    ]
                else:
                    parsed_val = new_val_str.strip()

                edited_filters.append(
                    FilterItem(column=new_col, operator=new_op_enum, value=parsed_val)
                )

    # --- Action buttons ---
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    edited_intent = ExportIntent(
        selected_view=intent.selected_view,
        columns=selected_columns or intent.columns,
        filters=edited_filters,
        warnings=intent.warnings,
    )

    if btn_col1.button("✅ Confirm Export", type="primary", key="btn_confirm"):
        return True, edited_intent

    if btn_col2.button("✏️ Edit & Resubmit", key="btn_edit"):
        return False, edited_intent

    return False, None
