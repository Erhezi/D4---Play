"""LangGraph node: route-and-count orchestrator for the refinement loop."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState

logger = logging.getLogger(__name__)


def node_orchestrator(state: ExportState) -> dict[str, Any]:
    """Decide whether to continue to parse_intent or to reset.

    Routing (done via conditional edges downstream):
    - If ``guardrail_passed`` is False → the graph will route to END
      (error already set by guardrail).
    - If ``refinement_count`` exceeds ``max_refinement_rounds`` → route to
      ``node_reset_signal``.
    - Otherwise → increment counter and route to ``node_parse_intent``.

    On the **first** invocation (no ``previous_intent``), the counter starts
    at 1.  On subsequent refinements it increments.
    """
    guardrail_passed = state.get("guardrail_passed", True)

    if not guardrail_passed:
        logger.info("node_orchestrator: guardrail blocked — routing to END")
        return {"status": "failed"}

    previous_intent = state.get("previous_intent")
    current_count = state.get("refinement_count", 0)
    max_rounds = settings.max_refinement_rounds

    if previous_intent is not None:
        # This is a refinement round
        new_count = current_count + 1
    else:
        # First run
        new_count = 1

    if new_count > max_rounds:
        logger.info(
            "node_orchestrator: refinement limit reached (%d/%d) — routing to reset",
            new_count, max_rounds,
        )
        return {
            "refinement_count": new_count,
            "status": "reset",
        }

    logger.info(
        "node_orchestrator: proceeding to parse_intent (round %d/%d)",
        new_count, max_rounds,
    )
    return {
        "refinement_count": new_count,
        "status": "orchestrating",
    }
