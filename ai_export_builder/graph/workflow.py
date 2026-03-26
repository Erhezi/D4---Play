"""LangGraph workflow definition for the Export Builder.

Graph flow:
  guardrail → orchestrator → (guardrail blocked → END)
                            → (refinement limit → reset_signal → END)
                            → parse_intent → validate_intent
                              → (errors & retries < 2 → loop back)
                              → (errors & retries >= 2 → human_review)
                              → (no errors → disambiguate)
  disambiguate → (disambiguation_needed → disambiguation_review (HITL))
              → hydrate_preview → human_review
  human_review (interrupt) → execute_export
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ai_export_builder.graph.nodes.disambiguate import node_disambiguate
from ai_export_builder.graph.nodes.execute_export import node_execute_export
from ai_export_builder.graph.nodes.guardrail import node_guardrail
from ai_export_builder.graph.nodes.hydrate_preview import node_hydrate_preview
from ai_export_builder.graph.nodes.meta_responder import node_meta_responder
from ai_export_builder.graph.nodes.orchestrator import node_orchestrator
from ai_export_builder.graph.nodes.parse_intent import node_parse_intent
from ai_export_builder.graph.nodes.reset_signal import node_reset_signal
from ai_export_builder.graph.nodes.validate_intent import node_validate_intent
from ai_export_builder.graph.state import ExportState

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ------------------------------------------------------------------
# Routing functions
# ------------------------------------------------------------------

def _after_guardrail(state: ExportState) -> str:
    """Route after guardrail: meta-queries go to meta_responder, everything else to orchestrator."""
    classification = state.get("guardrail_classification", "")
    if classification.startswith("meta:"):
        return "meta_responder"
    return "orchestrator"


def _after_orchestrator(state: ExportState) -> str:
    """Route after orchestrator: parse, reset, or end (if guardrail blocked)."""
    status = state.get("status", "")
    if status == "failed":
        # Guardrail blocked — go to END
        return "end"
    if status == "reset":
        return "reset_signal"
    return "parse_intent"


def _after_validate(state: ExportState) -> str:
    """Decide the next step after validation.

    - If status is needs_guidance → route to end for HITL clarification.
    - If no errors → go to disambiguate.
    - If errors and retries remain → loop back to parse_intent.
    - If errors and retries exhausted → surface to human_review with errors.
    """
    if state.get("status") == "needs_guidance":
        logger.info("Guidance needed — routing to end for HITL clarification")
        return "end"

    errors = state.get("validation_errors", [])
    retry = state.get("retry_count", 0)

    if not errors:
        logger.info("Validation passed — routing to disambiguate")
        return "disambiguate"

    if retry < MAX_RETRIES:
        logger.info("Validation failed (retry %d/%d) — looping back to parse_intent",
                     retry, MAX_RETRIES)
        return "retry_parse"

    logger.warning("Validation failed after %d retries — surfacing errors to human_review",
                   MAX_RETRIES)
    return "human_review"


def _after_disambiguate(state: ExportState) -> str:
    """Route after disambiguation: if previews are available go to HITL review,
    otherwise skip straight to hydrate_preview."""
    if state.get("disambiguation_needed"):
        logger.info("Disambiguation needed — routing to disambiguation_review")
        return "disambiguation_review"
    logger.info("No disambiguation needed — routing to hydrate_preview")
    return "hydrate_preview"


def _increment_retry(state: ExportState) -> dict[str, Any]:
    """Bump retry counter before re-entering parse_intent."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def _mark_pending_approval(state: ExportState) -> dict[str, Any]:
    """Set status to pending_approval before HITL breakpoint."""
    return {"status": "pending_approval"}


def _mark_pending_disambiguation(state: ExportState) -> dict[str, Any]:
    """Set status to pending_disambiguation before HITL breakpoint."""
    return {"status": "pending_disambiguation"}


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
    graph.add_node("guardrail", node_guardrail)
    graph.add_node("meta_responder", node_meta_responder)
    graph.add_node("orchestrator", node_orchestrator)
    graph.add_node("parse_intent", node_parse_intent)
    graph.add_node("validate_intent", node_validate_intent)
    graph.add_node("increment_retry", _increment_retry)
    graph.add_node("disambiguate", node_disambiguate)
    graph.add_node("disambiguation_review", _mark_pending_disambiguation)
    graph.add_node("hydrate_preview", node_hydrate_preview)
    graph.add_node("human_review", _mark_pending_approval)
    graph.add_node("pre_execute", _mark_executing)
    graph.add_node("execute_export", node_execute_export)
    graph.add_node("reset_signal", node_reset_signal)

    # Entry: guardrail → conditional routing (meta_responder or orchestrator)
    graph.set_entry_point("guardrail")
    graph.add_conditional_edges(
        "guardrail",
        _after_guardrail,
        {
            "meta_responder": "meta_responder",
            "orchestrator": "orchestrator",
        },
    )
    graph.add_edge("meta_responder", END)

    # Orchestrator routing
    graph.add_conditional_edges(
        "orchestrator",
        _after_orchestrator,
        {
            "parse_intent": "parse_intent",
            "reset_signal": "reset_signal",
            "end": END,
        },
    )

    graph.add_edge("reset_signal", END)

    # parse_intent → validate_intent
    graph.add_edge("parse_intent", "validate_intent")

    # Conditional routing after validation
    graph.add_conditional_edges(
        "validate_intent",
        _after_validate,
        {
            "disambiguate": "disambiguate",
            "human_review": "human_review",
            "retry_parse": "increment_retry",
            "end": END,
        },
    )

    graph.add_edge("increment_retry", "parse_intent")

    # Conditional routing after disambiguation
    graph.add_conditional_edges(
        "disambiguate",
        _after_disambiguate,
        {
            "disambiguation_review": "disambiguation_review",
            "hydrate_preview": "hydrate_preview",
        },
    )

    # After disambiguation_review (HITL interrupt) → hydrate_preview
    graph.add_edge("disambiguation_review", "hydrate_preview")

    # After hydrate_preview → human_review
    graph.add_edge("hydrate_preview", "human_review")

    # After human_review (HITL interrupt happens here) → execute
    graph.add_edge("human_review", "pre_execute")
    graph.add_edge("pre_execute", "execute_export")
    graph.add_edge("execute_export", END)

    return graph


def compile_graph():
    """Build and compile the graph with interrupts before disambiguation and execution.

    Two HITL interrupt points:
    1. Before 'disambiguation_review' output is consumed — shows the SELECT DISTINCT
       preview so users can confirm which entities to include.
    2. Before 'pre_execute' — shows the verification card for final column/filter review.
    """
    graph = build_graph()
    return graph.compile(interrupt_before=["disambiguation_review", "pre_execute"])
