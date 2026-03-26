"""Streamlit verification card — editable intent review before SQL execution."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import streamlit as st

from ai_export_builder.models.intent import OPERATOR_LABELS, ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.registry_loader import Registry
from ai_export_builder.services.sql_builder import build_query
from ai_export_builder.config import settings

# Build display-label list and reverse lookup for the operator selectbox
_OP_DISPLAY = [(op, OPERATOR_LABELS.get(op.value, op.value)) for op in FilterOperator]
_LABEL_TO_OP = {label: op for op, label in _OP_DISPLAY}


def _is_date_column(registry: Registry, view_id: str, column: str) -> bool:
    """Check if a column is a date type in the registry."""
    meta = registry.get_column_meta(view_id, column)
    return (meta or {}).get("type") in ("date", "datetime")


def _parse_date(val: str) -> date | None:
    """Try to parse an ISO date string, return None on failure."""
    try:
        return date.fromisoformat(val.strip())
    except (ValueError, AttributeError):
        return None


def render_verification_card(
    intent: ExportIntent,
    registry: Registry,
    validation_errors: list[str] | None = None,
    preview_data: list[dict[str, Any]] | None = None,
    aggregation_summary: dict[str, Any] | None = None,
) -> tuple[bool | str, ExportIntent | None]:
    """Render the verification card and return (action, edited_intent | None).

    Returns ``(True, edited_intent)`` when the user clicks *Confirm Export*.
    Returns ``('refine', edited_intent)`` when the user clicks *Refine*.
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

    # --- Column selection (grouped by field groups with checkboxes) ---
    st.markdown("**Columns:**")

    all_columns = registry.get_all_columns(intent.selected_view)
    companion_rendered: set[str] = set()
    selected_columns: list[str] = []

    field_groups = (view_meta or {}).get("field_groups", [])
    for fg in field_groups:
        group_name = fg.get("group_name", "Columns")
        group_type = fg.get("group_type", "")
        group_cols = fg.get("columns_included", [])
        is_basic = group_type in ("basic", "core")

        # Build items: (display_label, [col_names])
        group_items: list[tuple[str, list[str]]] = []
        seen_in_group: set[str] = set()

        for col_name in group_cols:
            if col_name in seen_in_group or col_name in companion_rendered:
                continue
            col_meta = registry.get_column_meta(intent.selected_view, col_name) or {}
            label = col_meta.get("label", col_name)

            companion = registry.get_companion_column(intent.selected_view, col_name)
            if companion and companion in group_cols:
                comp_meta = registry.get_column_meta(intent.selected_view, companion) or {}
                display = f"{label} + {comp_meta.get('label', companion)}"
                cols_list = [col_name, companion]
                seen_in_group.add(companion)
                companion_rendered.add(companion)
            else:
                display = label
                cols_list = [col_name]

            group_items.append((display, cols_list))
            seen_in_group.add(col_name)

        total_count = len(group_items)
        chk_prefix = f"chk_{intent.selected_view}_{group_name.replace(' ', '_')}"

        # Compute selection count from session state (if present) or intent defaults
        sel_count = 0
        for idx, (_display, cols) in enumerate(group_items):
            chk_key = f"{chk_prefix}_{idx}"
            if is_basic:
                sel_count += 1
            elif chk_key in st.session_state:
                if st.session_state[chk_key]:
                    sel_count += 1
            elif any(c in intent.columns for c in cols):
                sel_count += 1

        if is_basic:
            expander_label = f"🔒 {group_name} (always included) [{total_count}/{total_count}]"
        else:
            expander_label = f"{group_name} [{sel_count}/{total_count}]"

        # Keep expander open if user has interacted with any checkbox in this group
        has_interaction = any(
            f"{chk_prefix}_{i}" in st.session_state for i in range(total_count)
        )

        with st.expander(expander_label, expanded=has_interaction):
            for idx, (display, cols) in enumerate(group_items):
                chk_key = f"{chk_prefix}_{idx}"
                if is_basic:
                    st.checkbox(display, value=True, disabled=True, key=chk_key)
                    selected_columns.extend(cols)
                else:
                    default_val = any(c in intent.columns for c in cols)
                    if st.checkbox(display, value=default_val, key=chk_key):
                        selected_columns.extend(cols)

    # Any remaining columns not in any field group
    grouped_cols: set[str] = set()
    for fg in field_groups:
        grouped_cols.update(fg.get("columns_included", []))
    ungrouped = [c for c in all_columns if c not in grouped_cols and c not in companion_rendered]

    if ungrouped:
        ug_items: list[tuple[str, list[str]]] = []
        ug_seen: set[str] = set()

        for col_name in ungrouped:
            if col_name in ug_seen:
                continue
            col_meta = registry.get_column_meta(intent.selected_view, col_name) or {}
            label = col_meta.get("label", col_name)
            companion = registry.get_companion_column(intent.selected_view, col_name)
            if companion and companion in ungrouped and companion not in ug_seen:
                comp_meta = registry.get_column_meta(intent.selected_view, companion) or {}
                display = f"{label} + {comp_meta.get('label', companion)}"
                cols_list = [col_name, companion]
                ug_seen.add(companion)
            else:
                display = label
                cols_list = [col_name]
            ug_items.append((display, cols_list))
            ug_seen.add(col_name)

        ug_prefix = f"chk_{intent.selected_view}_Other"
        ug_sel = 0
        for idx, (_display, cols) in enumerate(ug_items):
            chk_key = f"{ug_prefix}_{idx}"
            if chk_key in st.session_state:
                if st.session_state[chk_key]:
                    ug_sel += 1
            elif any(c in intent.columns for c in cols):
                ug_sel += 1

        ug_has_interaction = any(
            f"{ug_prefix}_{i}" in st.session_state for i in range(len(ug_items))
        )

        with st.expander(f"Other Columns [{ug_sel}/{len(ug_items)}]", expanded=ug_has_interaction):
            for idx, (display, cols) in enumerate(ug_items):
                chk_key = f"{ug_prefix}_{idx}"
                default_val = any(c in intent.columns for c in cols)
                if st.checkbox(display, value=default_val, key=chk_key):
                    selected_columns.extend(cols)

    # --- Filters (editable rows) ---
    st.markdown("**Filters:**")
    edited_filters: list[FilterItem] = []

    if not intent.filters:
        st.info("No filters applied.")
    else:
        for i, f in enumerate(intent.filters):
            display_val = ", ".join(f.value) if isinstance(f.value, list) else str(f.value)
            with st.expander(f"Filter {i + 1}: {f.column} {f.operator.label} {display_val}", expanded=True):
                fcol1, fcol2, fcol3 = st.columns([2, 1, 3])

                col_options = all_columns if all_columns else [f.column]
                col_idx = col_options.index(f.column) if f.column in col_options else 0
                new_col = fcol1.selectbox(
                    "Column", col_options, index=col_idx, key=f"filter_col_{i}"
                )

                op_labels = [label for _, label in _OP_DISPLAY]
                current_label = f.operator.label
                op_idx = op_labels.index(current_label) if current_label in op_labels else 0
                new_op_label = fcol2.selectbox(
                    "Operator", op_labels, index=op_idx, key=f"filter_op_{i}"
                )
                new_op_enum = _LABEL_TO_OP.get(new_op_label, f.operator)

                # --- Value input: calendar picker for date columns, text for others ---
                is_date = _is_date_column(registry, intent.selected_view, new_col)

                if is_date and new_op_enum == FilterOperator.between:
                    # Date between → two calendar pickers
                    values = f.value if isinstance(f.value, list) else [f.value, f.value]
                    d1, d2 = fcol3.columns(2)
                    start_date = d1.date_input(
                        "Start",
                        value=_parse_date(values[0]) or date.today(),
                        key=f"filter_date_start_{i}",
                    )
                    end_date = d2.date_input(
                        "End",
                        value=_parse_date(values[1] if len(values) > 1 else values[0]) or date.today(),
                        key=f"filter_date_end_{i}",
                    )
                    parsed_val: str | list[str] = [start_date.isoformat(), end_date.isoformat()]

                elif is_date and new_op_enum not in (FilterOperator.in_, FilterOperator.like):
                    # Date with single-value operator (eq, gt, lt, etc.) → one calendar picker
                    current_str = f.value if isinstance(f.value, str) else (f.value[0] if f.value else "")
                    picked = fcol3.date_input(
                        "Date",
                        value=_parse_date(current_str) or date.today(),
                        key=f"filter_date_{i}",
                    )
                    parsed_val = picked.isoformat()

                else:
                    # Non-date or LIKE/IN: text input
                    current_val = (
                        ", ".join(f.value) if isinstance(f.value, list) else str(f.value)
                    )
                    if f.operator == FilterOperator.like:
                        current_val = current_val.strip("%")
                    new_val_str = fcol3.text_input(
                        "Value(s)", value=current_val, key=f"filter_val_{i}",
                        help="For 'between': start, end.  For 'is one of': comma-separated.  For 'contains': search term (wildcards added automatically).",
                    )

                    if new_op_enum in (FilterOperator.between, FilterOperator.in_):
                        parsed_val = [
                            v.strip() for v in new_val_str.split(",") if v.strip()
                        ]
                    else:
                        parsed_val = new_val_str.strip()

                edited_filters.append(
                    FilterItem(column=new_col, operator=new_op_enum, value=parsed_val)
                )

    # --- Build edited intent from current selections ---
    edited_intent = intent.model_copy(update={
        "columns": selected_columns or intent.columns,
        "filters": edited_filters,
    })

    # --- Generated SQL (collapsed) ---
    try:
        _sql, _params = build_query(edited_intent, settings.user_facilities)
        with st.expander("🔍 Generated SQL", expanded=False):
            st.code(_sql, language="sql")
            if _params:
                st.caption(f"Parameters: {list(_params)}")
    except Exception:
        pass  # Non-critical — don't block the card

    # --- Preview & Action buttons ---
    preview_active = st.session_state.get("preview_active", False)

    if preview_active:
        # Show preview data
        if preview_data:
            with st.expander("📊 Data Preview (first 20 rows)", expanded=True):
                import pandas as pd
                st.dataframe(pd.DataFrame(preview_data), width='stretch')

        # Show aggregation summary
        if aggregation_summary:
            st.markdown("**Summary:**")
            metric_cols = st.columns(1 + len(aggregation_summary.get("sums", {})))
            metric_cols[0].metric("Total Rows", f"{aggregation_summary['row_count']:,}")
            for idx, (col, total) in enumerate(aggregation_summary.get("sums", {}).items(), start=1):
                if idx < len(metric_cols):
                    metric_cols[idx].metric(f"Total {col}", f"${total:,.2f}")

        # Show action buttons: Re-Preview and Confirm; refinement via chat
        btn_col1, btn_col2, _ = st.columns([1, 1, 3])

        if btn_col1.button("👁 Re-Preview", key="btn_preview"):
            return "preview", edited_intent

        if btn_col2.button("✅ Confirm Export", type="primary", key="btn_confirm"):
            return True, edited_intent

        st.caption("💬 Want to change something? Just type in the chat below — e.g. *\"add a date filter for last quarter\"*")
    else:
        # Show Preview button only (no Confirm/Refine until preview is done)
        if st.button("👁 Preview", type="primary", key="btn_preview"):
            return "preview", edited_intent

    return False, None
