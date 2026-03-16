"""Streamlit entry point for the AI-Assisted Export Builder."""
from __future__ import annotations

import uuid
from datetime import date

import pandas as pd
import streamlit as st

from ai_export_builder.config import settings
from ai_export_builder.graph.nodes.execute_export import node_execute_export
from ai_export_builder.graph.state import ExportState, TemporalContext, UserProfile
from ai_export_builder.graph.workflow import compile_graph
from ai_export_builder.models.intent import ExportIntent
from ai_export_builder.services.audit import log_export_request
from ai_export_builder.services.rate_limiter import rate_limiter
from ai_export_builder.services.registry_loader import load_registry
from ai_export_builder.services.sql_builder import build_query
from ai_export_builder.ui.chat import (
    add_message,
    get_user_input,
    init_session_state,
    render_chat_history,
)
from ai_export_builder.ui.disambiguation_card import render_disambiguation_card
from ai_export_builder.ui.verification_card import render_verification_card

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Export Builder",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialise
# ---------------------------------------------------------------------------
init_session_state()

if "registry" not in st.session_state:
    st.session_state.registry = load_registry()

if "compiled_graph" not in st.session_state:
    st.session_state.compiled_graph = compile_graph()

registry = st.session_state.registry
compiled_graph = st.session_state.compiled_graph

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 AI Export Builder")
    st.divider()
    st.markdown(f"**User:** {settings.test_user_name}")
    st.markdown(f"**User ID:** {settings.test_user_id}")
    remaining = rate_limiter.remaining(settings.test_user_id)
    st.markdown(f"**Requests remaining today:** {remaining} / {settings.daily_request_limit}")

    if st.session_state.get("result_df") is not None:
        st.divider()
        st.markdown("### Download")
        df: pd.DataFrame = st.session_state.result_df
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download CSV ({len(df)} rows)",
            data=csv_data,
            file_name="export.csv",
            mime="text/csv",
        )

    st.divider()
    if st.button("🗑️ Clear conversation"):
        for key in ["messages", "graph_state", "thread_id",
                    "awaiting_confirmation", "awaiting_disambiguation", "result_df"]:
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("AI-Assisted Export Builder")

render_chat_history()

# ---------------------------------------------------------------------------
# Disambiguation card (HITL breakpoint — appears before verification)
# ---------------------------------------------------------------------------
if st.session_state.get("awaiting_disambiguation"):
    graph_state: dict = st.session_state.graph_state
    intent: ExportIntent | None = graph_state.get("intent")
    disambiguation_results = graph_state.get("disambiguation_results", [])

    if intent is not None and disambiguation_results:
        confirmed, updated_intent = render_disambiguation_card(
            intent, disambiguation_results
        )

        if confirmed and updated_intent is not None:
            st.session_state.awaiting_disambiguation = False
            graph_state["intent"] = updated_intent
            graph_state["disambiguation_needed"] = False
            st.session_state.graph_state = graph_state

            add_message(
                "assistant",
                "Entity selection confirmed. Please review the full export details below.",
            )
            st.session_state.awaiting_confirmation = True
            st.rerun()

# ---------------------------------------------------------------------------
# Verification card (HITL breakpoint)
# ---------------------------------------------------------------------------
if st.session_state.get("awaiting_confirmation"):
    graph_state: dict = st.session_state.graph_state
    intent: ExportIntent | None = graph_state.get("intent")
    validation_errors = graph_state.get("validation_errors", [])

    if intent is not None:
        confirmed, edited_intent = render_verification_card(
            intent, registry, validation_errors=validation_errors
        )

        if confirmed and edited_intent is not None:
            # User confirmed — execute the query directly
            st.session_state.awaiting_confirmation = False
            graph_state["intent"] = edited_intent
            st.session_state.graph_state = graph_state

            add_message("assistant", "Running export query…")

            try:
                exec_result = node_execute_export(graph_state)
                graph_state.update(exec_result)
                st.session_state.graph_state = graph_state
                status = graph_state.get("status", "unknown")

                if status == "completed":
                    result_df = graph_state.get("result_df")
                    row_count = graph_state.get("result_row_count", 0)
                    st.session_state.result_df = result_df

                    add_message(
                        "assistant",
                        f"✅ Export complete! **{row_count}** rows returned. "
                        "Use the download button in the sidebar to get your CSV.",
                    )

                    # Show preview
                    if result_df is not None and len(result_df) > 0:
                        st.dataframe(result_df.head(100), use_container_width=True)

                    # Audit log
                    sql, _ = build_query(edited_intent, settings.user_facilities)
                    log_export_request(
                        user_id=settings.test_user_id,
                        prompt=graph_state.get("user_query", ""),
                        intent=edited_intent,
                        sql=sql,
                        row_count=row_count,
                        status="completed",
                    )
                else:
                    err = graph_state.get("error_message", "Unknown error")
                    add_message("assistant", f"❌ Export failed: {err}")
                    log_export_request(
                        user_id=settings.test_user_id,
                        prompt=graph_state.get("user_query", ""),
                        intent=edited_intent,
                        sql=None,
                        row_count=None,
                        status="failed",
                        error=err,
                    )
            except Exception as exc:
                add_message("assistant", f"❌ Error during execution: {exc}")
                log_export_request(
                    user_id=settings.test_user_id,
                    prompt=graph_state.get("user_query", ""),
                    intent=edited_intent,
                    sql=None,
                    row_count=None,
                    status="failed",
                    error=str(exc),
                )
            st.rerun()

        elif edited_intent is not None and not confirmed:
            # User chose "Edit & Resubmit"
            st.session_state.awaiting_confirmation = False
            resubmit_text = (
                f"Edited intent for view `{edited_intent.selected_view}` "
                f"with {len(edited_intent.columns)} columns and "
                f"{len(edited_intent.filters)} filters."
            )
            add_message("user", resubmit_text)
            # Re-run the full graph with the edited intent as the new query context
            st.session_state.graph_state = None
            st.rerun()

# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------
user_input = get_user_input()

if user_input:
    # Rate-limit check
    if not rate_limiter.check(settings.test_user_id):
        add_message(
            "assistant",
            f"⚠️ You've reached the daily limit of **{settings.daily_request_limit}** requests. "
            "Please try again tomorrow.",
        )
    else:
        add_message("user", user_input)
        add_message("assistant", "🔍 Parsing your request…")

        # Build initial state
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id

        initial_state: ExportState = {
            "user_query": user_input,
            "intent": None,
            "validation_errors": [],
            "status": "parsing",
            "retry_count": 0,
            "disambiguation_needed": False,
            "disambiguation_results": [],
            "temporal_context": TemporalContext(
                current_date=date.today().isoformat(),
                fiscal_year_start_month=settings.fiscal_year_start_month,
            ),
            "user_profile": UserProfile(
                user_id=settings.test_user_id,
                user_name=settings.test_user_name,
                facilities=settings.user_facilities,
            ),
            "result_df": None,
            "result_row_count": 0,
            "error_message": "",
        }

        rate_limiter.increment(settings.test_user_id)

        try:
            thread_cfg = {"configurable": {"thread_id": thread_id}}
            graph_state = dict(initial_state)

            for event in compiled_graph.stream(
                initial_state,
                config=thread_cfg,
            ):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        graph_state.update(node_output)

            st.session_state.graph_state = graph_state
            status = graph_state.get("status", "unknown")

            if status == "pending_disambiguation":
                # Graph paused at disambiguation HITL breakpoint
                intent = graph_state.get("intent")
                results = graph_state.get("disambiguation_results", [])
                total_matches = sum(len(r.get("matches", [])) for r in results)
                if intent and results:
                    add_message(
                        "assistant",
                        f"🔍 Found **{total_matches}** matching entities. "
                        "Please review and confirm which ones to include.",
                    )
                st.session_state.awaiting_disambiguation = True
                st.rerun()

            elif status == "pending_approval":
                # Graph paused at HITL breakpoint
                intent = graph_state.get("intent")
                if intent:
                    add_message(
                        "assistant",
                        f"I've parsed your request. Please review the export details below for "
                        f"**{intent.selected_view}** with **{len(intent.columns)}** columns "
                        f"and **{len(intent.filters)}** filters.",
                    )
                st.session_state.awaiting_confirmation = True
                st.rerun()

            elif status == "failed":
                errors = graph_state.get("validation_errors", [])
                err_msg = graph_state.get("error_message", "")
                detail = "\n".join(f"- {e}" for e in errors) if errors else err_msg
                add_message("assistant", f"❌ Unable to process your request:\n{detail}")
                log_export_request(
                    user_id=settings.test_user_id,
                    prompt=user_input,
                    intent=graph_state.get("intent"),
                    sql=None,
                    row_count=None,
                    status="failed",
                    error=detail,
                )

        except Exception as exc:
            add_message("assistant", f"❌ An unexpected error occurred: {exc}")
            log_export_request(
                user_id=settings.test_user_id,
                prompt=user_input,
                intent=None,
                sql=None,
                row_count=None,
                status="failed",
                error=str(exc),
            )
