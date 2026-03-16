"""Streamlit disambiguation card — preview LIKE/eq matches for text↔ID paired columns."""
from __future__ import annotations

from typing import Any

import streamlit as st

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator


def render_disambiguation_card(
    intent: ExportIntent,
    disambiguation_results: list[dict[str, Any]],
) -> tuple[bool, ExportIntent | None]:
    """Show SELECT DISTINCT previews and let the user pick specific entities.

    Returns ``(True, updated_intent)`` when the user clicks *Confirm Selection*.
    Returns ``(False, None)`` while waiting for user action.
    """
    st.subheader("🔍 Confirm Matching Entities")
    st.markdown(
        "The following entities matched your search. "
        "Select the ones you want to include in the export, "
        "or click **Keep Partial Match** to use the original filter as-is."
    )

    selections: dict[str, list[str]] = {}  # column -> selected ID values

    for result in disambiguation_results:
        text_col = result["column"]
        id_col = result["companion"]
        matches = result["matches"]
        original_value = result["original_value"]

        st.markdown(f"**{text_col}** matching `{original_value}`:")

        if not matches:
            st.warning(
                f"No matches found for '{original_value}'. "
                "Try adjusting your search term."
            )
            new_value = st.text_input(
                f"Adjust search for {text_col}:",
                value=original_value if isinstance(original_value, str) else str(original_value),
                key=f"disambig_adjust_{text_col}",
            )
            selections[text_col] = []
            continue

        selected_ids: list[str] = []
        for i, match in enumerate(matches):
            label = f"{match['text']}  (ID: {match['id']})"
            if st.checkbox(label, value=True, key=f"disambig_{text_col}_{i}"):
                selected_ids.append(match["id"])

        selections[text_col] = selected_ids
        st.markdown(f"*{len(selected_ids)} of {len(matches)} selected*")
        st.divider()

    # --- Action buttons ---
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])

    if btn_col1.button("✅ Confirm Selection", type="primary", key="btn_disambig_confirm"):
        updated_intent = _apply_disambiguation(intent, disambiguation_results, selections)
        return True, updated_intent

    if btn_col2.button("📝 Keep Partial Match", key="btn_disambig_keep"):
        # Keep original LIKE filters unchanged
        return True, intent

    return False, None


def _apply_disambiguation(
    intent: ExportIntent,
    disambiguation_results: list[dict[str, Any]],
    selections: dict[str, list[str]],
) -> ExportIntent:
    """Replace LIKE/eq filters on text columns with IN filters on ID columns
    using the user's selected values."""
    # Build lookup: text_col -> result metadata
    result_lookup = {r["column"]: r for r in disambiguation_results}

    new_filters: list[FilterItem] = []
    for f in intent.filters:
        if f.column in result_lookup and f.column in selections:
            selected_ids = selections[f.column]
            id_col = result_lookup[f.column]["companion"]
            if selected_ids:
                if len(selected_ids) == 1:
                    new_filters.append(
                        FilterItem(column=id_col, operator=FilterOperator.eq, value=selected_ids[0])
                    )
                else:
                    new_filters.append(
                        FilterItem(column=id_col, operator=FilterOperator.in_, value=selected_ids)
                    )
            # If no IDs selected, drop the filter entirely
        else:
            new_filters.append(f)

    return ExportIntent(
        selected_view=intent.selected_view,
        columns=intent.columns,
        filters=new_filters,
        warnings=intent.warnings,
    )
