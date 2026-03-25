"""LangGraph node: terminal node when the refinement limit is exceeded."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState

logger = logging.getLogger(__name__)


def node_reset_signal(state: ExportState) -> dict[str, Any]:
    """Return a user-friendly message when the maximum refinement rounds are exhausted."""
    max_rounds = settings.max_refinement_rounds
    logger.info("node_reset_signal: refinement limit (%d) reached", max_rounds)
    return {
        "status": "reset",
        "validation_errors": [
            f"You've reached the maximum of {max_rounds} refinements for this session. "
            "Please start a new conversation for best accuracy."
        ],
    }
