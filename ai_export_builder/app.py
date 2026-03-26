"""Streamlit entry point for the AI-Assisted Export Builder."""
from __future__ import annotations

import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from ai_export_builder.services.intent_summarizer import summarize_intent
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
# Progressive status labels for streaming graph events
# ---------------------------------------------------------------------------
_NODE_STATUS_LABELS: dict[str, str] = {
    "guardrail": "🛡️ Checking request safety…",
    "meta_responder": "📖 Looking up information…",
    "orchestrator": "🧠 Understanding your request…",
    "parse_intent": "🧠 Interpreting your request…",
    "validate_intent": "✅ Validating against data registry…",
    "increment_retry": "🔄 Adjusting interpretation…",
    "disambiguate": "🔍 Looking up matching entities…",
    "hydrate_preview": "📊 Preparing data preview…",
    "human_review": "📋 Ready for your review",
    "disambiguation_review": "📋 Entities need confirmation",
}

_PROCESSING_TIPS: list[str] = [
    "💡 Tip: You can filter by vendor, date range, GL account, cost center, and more.",
    "💡 Tip: After the export, you can download results as CSV from the sidebar.",
    "💡 Tip: Ask *\"what data is available?\"* to see all queryable views.",
    "💡 Tip: You can refine your export at any time by typing in the chat.",
    "💡 Tip: Ask *\"what does [field name] mean?\"* to learn about any column.",
]

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

    # Show refinement round if in refinement loop
    _gs_sidebar = st.session_state.get("graph_state") or {}
    _ref_count = _gs_sidebar.get("refinement_count", 0)
    if _ref_count > 0:
        st.markdown(f"**Refinement round:** {_ref_count} / {settings.max_refinement_rounds}")

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

    # Show SQL button — visible whenever an intent has been parsed
    _gs = st.session_state.get("graph_state") or {}
    _intent = _gs.get("intent")
    if _intent is not None:
        st.divider()
        if st.button("🔍 Show SQL"):
            st.session_state.show_sql = not st.session_state.get("show_sql", False)
        if st.session_state.get("show_sql"):
            try:
                _sql, _params = build_query(_intent, settings.user_facilities)
                st.markdown("**Generated SQL**")
                st.code(_sql, language="sql")
                if _params:
                    st.markdown("**Parameters**")
                    st.json(list(_params))
            except Exception as _exc:
                st.warning(f"Could not render SQL: {_exc}")

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
            st.session_state.preview_active = False
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
        action, edited_intent = render_verification_card(
            intent,
            registry,
            validation_errors=validation_errors,
            preview_data=graph_state.get("preview_data"),
            aggregation_summary=graph_state.get("aggregation_summary"),
        )

        if action == "preview" and edited_intent is not None:
            # Run preview query with current column/filter edits
            graph_state["intent"] = edited_intent
            graph_state["preview_data"] = None
            graph_state["aggregation_summary"] = None
            from ai_export_builder.graph.nodes.hydrate_preview import node_hydrate_preview
            try:
                preview_result = node_hydrate_preview(graph_state)
                graph_state.update(preview_result)
            except Exception as exc:
                add_message("assistant", f"⚠️ Preview generation failed: {exc}")
            st.session_state.graph_state = graph_state
            st.session_state.preview_active = True
            st.rerun()

        if action is True and edited_intent is not None:
            # User confirmed — execute the query directly
            st.session_state.awaiting_confirmation = False
            st.session_state.preview_active = False
            graph_state["intent"] = edited_intent
            graph_state["previous_intent"] = None  # Clear refinement context
            graph_state["refinement_count"] = 0
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
                        st.dataframe(result_df.head(100), width='stretch')

                    # Follow-up suggestion buttons
                    st.markdown("**What's next?**")
                    _fu_cols = st.columns(3)
                    if _fu_cols[0].button("📅 Different date range", key="fu_date"):
                        st.session_state._prefill_query = "Change the date range"
                        graph_state["previous_intent"] = edited_intent
                        st.session_state.graph_state = graph_state
                        st.rerun()
                    if _fu_cols[1].button("📊 Add more columns", key="fu_cols"):
                        st.session_state._prefill_query = "Add more columns to the export"
                        graph_state["previous_intent"] = edited_intent
                        st.session_state.graph_state = graph_state
                        st.rerun()
                    if _fu_cols[2].button("🔄 Different filters", key="fu_filter"):
                        st.session_state._prefill_query = "Change the filters"
                        graph_state["previous_intent"] = edited_intent
                        st.session_state.graph_state = graph_state
                        st.rerun()

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

# ---------------------------------------------------------------------------
# Welcome message (first visit only)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    # Collect sample questions from registry
    _sample_questions: list[str] = []
    for _vid in registry.all_view_ids():
        _vmeta = registry.get_view_meta(_vid)
        for sq in (_vmeta or {}).get("samples_of_valid_queries", [])[:2]:
            _sample_questions.append(sq)

    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Welcome to the AI Export Builder!** I can help you pull data from our "
            "procurement, invoicing, and savings databases.\n\n"
            "Just describe the data you need in plain English. For example:"
        )
        # Render clickable sample question buttons
        _btn_cols = st.columns(min(len(_sample_questions), 3))
        for _i, _sq in enumerate(_sample_questions[:3]):
            if _btn_cols[_i % 3].button(f"📝 {_sq}", key=f"sample_q_{_i}"):
                st.session_state._prefill_query = _sq
                st.rerun()
        st.markdown(
            "\nYou can also ask:\n"
            '- *"What data is available?"*\n'
            '- *"What does VendorName mean?"*'
        )

# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------

# Check for prefilled query from sample question buttons
_prefill = st.session_state.pop("_prefill_query", None)
user_input = _prefill or get_user_input()

if user_input:
    # If awaiting_confirmation, auto-trigger refinement context
    existing_gs = st.session_state.get("graph_state") or {}
    if st.session_state.get("awaiting_confirmation"):
        # Set up refinement from the current verified intent
        current_intent = existing_gs.get("intent")
        if current_intent is not None:
            existing_gs["previous_intent"] = current_intent
            st.session_state.graph_state = existing_gs
        st.session_state.awaiting_confirmation = False

    # Reset UI states for new/refinement request
    st.session_state.preview_active = False
    st.session_state.awaiting_refinement_input = False
    for key in list(st.session_state.keys()):
        if key.startswith("grp_") or key.startswith("chk_"):
            del st.session_state[key]

    # Rate-limit check
    if not rate_limiter.check(settings.test_user_id):
        add_message(
            "assistant",
            f"⚠️ You've reached the daily limit of **{settings.daily_request_limit}** requests. "
            "Please try again tomorrow.",
        )
    else:
        add_message("user", user_input)

        # Build initial state — carry over refinement context if available
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id

        existing_gs = st.session_state.get("graph_state") or {}
        previous_intent = existing_gs.get("previous_intent")
        refinement_count = existing_gs.get("refinement_count", 0)
        prev_disambiguation = existing_gs.get("disambiguation_results", [])
        if previous_intent is not None:
            # Carry original query from the first request
            original_user_query = existing_gs.get("original_user_query", "")
        else:
            prev_disambiguation = []  # Don't carry stale results for fresh requests
            original_user_query = user_input  # This IS the original request

        initial_state: ExportState = {
            "user_query": user_input,
            "original_user_query": original_user_query,
            "intent": None,
            "validation_errors": [],
            "status": "parsing",
            "retry_count": 0,
            "refinement_count": refinement_count,
            "previous_intent": previous_intent,
            "disambiguation_results": prev_disambiguation,
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

            # Progressive status feedback
            with st.status("Processing your request…", expanded=True) as status_container:
                st.caption(random.choice(_PROCESSING_TIPS))

                for event in compiled_graph.stream(
                    initial_state,
                    config=thread_cfg,
                ):
                    for node_name, node_output in event.items():
                        if isinstance(node_output, dict):
                            graph_state.update(node_output)
                        # Update status label
                        label = _NODE_STATUS_LABELS.get(node_name)
                        if label:
                            status_container.update(label=label)
                            st.write(label)

                status_container.update(label="Done!", state="complete")

            st.session_state.graph_state = graph_state
            status = graph_state.get("status", "unknown")

            # --- Meta-query response ---
            if status == "meta_response":
                meta_text = graph_state.get("meta_response", "")
                add_message("assistant", meta_text)
                # Don't clear graph state — allow follow-up questions
                st.rerun()

            # --- Guidance needed (proactive clarification) ---
            elif status == "needs_guidance":
                guidance = graph_state.get("guidance_question", "Could you tell me more about what data you need?")
                add_message("assistant", f"🤔 {guidance}")
                # Keep partial intent as previous_intent for the next round
                partial = graph_state.get("intent")
                if partial:
                    graph_state["previous_intent"] = partial
                st.session_state.graph_state = graph_state
                st.rerun()

            elif status == "pending_approval":
                # Graph paused at HITL breakpoint
                intent = graph_state.get("intent")
                if intent:
                    summary = summarize_intent(intent, registry)
                    ref_cnt = graph_state.get("refinement_count", 0)
                    if ref_cnt > 1:
                        add_message(
                            "assistant",
                            f"Updated! {summary}\n\nPlease review below.",
                        )
                    else:
                        add_message(
                            "assistant",
                            f"I've parsed your request: {summary}\n\n"
                            "Please review the export details below.",
                        )
                st.session_state.preview_active = False
                st.session_state.awaiting_confirmation = True
                st.rerun()

            elif status == "disambiguating":
                # Graph paused at disambiguation HITL breakpoint
                add_message(
                    "assistant",
                    "I found some ambiguous terms in your request. "
                    "Please review the matching entities below to confirm your selection.",
                )
                st.session_state.awaiting_disambiguation = True
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
                # Clear state so next input is treated as a fresh request
                st.session_state.graph_state = None

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
            # Clear state so next input is treated as a fresh request
            st.session_state.graph_state = None
