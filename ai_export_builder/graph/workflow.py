"""LangGraph workflow definition for the Export Builder.

Graph flow:
  parse_intent → validate_intent → (errors & retries < 2 → loop back)
                                 → (errors & retries >= 2 → human_review)
                                 → (no errors → human_review)
  human_review (interrupt) → execute_export
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ai_export_builder.graph.nodes.execute_export import node_execute_export
from ai_export_builder.graph.nodes.parse_intent import node_parse_intent
from ai_export_builder.graph.nodes.validate_intent import node_validate_intent
from ai_export_builder.graph.state import ExportState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ------------------------------------------------------------------
# Routing functions
# ------------------------------------------------------------------

def _after_validate(state: ExportState) -> str:
    """Decide the next step after validation.

    - If no errors → go to human_review (HITL breakpoint).
    - If errors and retries remain → loop back to parse_intent.
    - If errors and retries exhausted → surface to human_review with errors.
    """
    errors = state.get("validation_errors", [])
    retry = state.get("retry_count", 0)

    if not errors:
        logger.info("Validation passed — routing to human_review")
        return "human_review"

    if retry < MAX_RETRIES:
        logger.info("Validation failed (retry %d/%d) — looping back to parse_intent",
                     retry, MAX_RETRIES)
        return "retry_parse"

    logger.warning("Validation failed after %d retries — surfacing errors to human_review",
                   MAX_RETRIES)
    return "human_review"


def _increment_retry(state: ExportState) -> dict[str, Any]:
    """Bump retry counter before re-entering parse_intent."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def _mark_pending_approval(state: ExportState) -> dict[str, Any]:
    """Set status to pending_approval before HITL breakpoint."""
    return {"status": "pending_approval"}


def _mark_executing(state: ExportState) -> dict[str, Any]:
    """Set status to executing before the SQL runs."""
    return {"status": "executing"}


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph workflow."""
    graph = StateGraph(ExportState)

    # Nodes
    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("validate_intent", node_validate_intent)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("human_review", _mark_pending_approval)
    graph.add_node("pre_execute", _mark_executing)
    graph.add_node("execute_export", node_execute_export)

    # Edges
    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "validate_intent")

    # Conditional routing after validation
    graph.add_conditional_edges(
        "validate_intent",
        _after_validate,
        {
            "human_review": "human_review",
            "retry_parse": "increment_retry",
        },
    )

    graph.add_edge("increment_retry", "parse_intent")

    # After human_review (HITL interrupt happens here) → execute
    graph.add_edge("human_review", "pre_execute")
    graph.add_edge("pre_execute", "execute_export")
    graph.add_edge("execute_export", END)

    return graph


def compile_graph():
    """Build and compile the graph with an interrupt before human_review's output is consumed.

    The interrupt happens BEFORE 'pre_execute', meaning the graph pauses
    after `human_review` sets status='pending_approval'. The Streamlit UI
    will show the verification card, then resume the graph to continue
    into execute_export.
    """
    graph = build_graph()
    return graph.compile(interrupt_before=["pre_execute"])
