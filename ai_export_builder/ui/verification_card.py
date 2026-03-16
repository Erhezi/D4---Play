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

    # --- Column selection (checkboxes) ---
    st.markdown("**Columns:**")
    all_columns = registry.get_all_columns(intent.selected_view)
    selected_columns: list[str] = []

    cols_per_row = 3
    for i in range(0, len(all_columns), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_widget in enumerate(row_cols):
            idx = i + j
            if idx >= len(all_columns):
                break
            col_name = all_columns[idx]
            col_meta = registry.get_column_meta(intent.selected_view, col_name) or {}
            label = col_meta.get("label", col_name)
            checked = col_name in intent.columns
            if col_widget.checkbox(f"{label} ({col_name})", value=checked, key=f"col_{col_name}"):
                selected_columns.append(col_name)

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
                new_val_str = fcol3.text_input(
                    "Value(s)", value=current_val, key=f"filter_val_{i}",
                    help="For 'between': start, end. For 'in': comma-separated values.",
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
