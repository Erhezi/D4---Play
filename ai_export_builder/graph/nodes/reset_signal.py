"""LangGraph node: terminal node when the refinement limit is exceeded."""

from __future__ import annotations

import logging
from typing import Any

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState

logger = logging.getLogger(__name__)


def node_reset_signal(state: ExportState) -> dict[str, Any]:
    """Return a user-friendly suggestion when the maximum refinement rounds are exhausted."""
    max_rounds = settings.max_refinement_rounds
    logger.info("node_reset_signal: refinement limit (%d) reached", max_rounds)
    return {
        "status": "failed",
        "validation_errors": [
            f"You've gone through {max_rounds} refinement rounds. "
            "For best accuracy, I'd recommend starting a new conversation. "
            "You can click **Clear conversation** in the sidebar to start fresh, "
            "or keep refining if you'd like."
        ],
    }
