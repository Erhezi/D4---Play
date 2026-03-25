"""Tests for the reset_signal node."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_export_builder.graph.nodes.reset_signal import node_reset_signal
from ai_export_builder.graph.state import ExportState


def _make_state(**overrides) -> ExportState:
    base: dict = {
        "user_query": "Show me PO lines",
        "intent": None,
        "validation_errors": [],
        "status": "reset",
        "retry_count": 0,
        "refinement_count": 4,
        "guardrail_passed": True,
        "previous_intent": None,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }
    base.update(overrides)
    return base


class TestResetSignal:
    def test_returns_reset_status(self):
        result = node_reset_signal(_make_state())
        assert result["status"] == "reset"

    def test_returns_user_friendly_message(self):
        result = node_reset_signal(_make_state())
        errors = result["validation_errors"]
        assert len(errors) == 1
        assert "maximum" in errors[0].lower()
        assert "new conversation" in errors[0].lower()

    @patch("ai_export_builder.graph.nodes.reset_signal.settings")
    def test_message_includes_configured_rounds(self, mock_settings):
        mock_settings.max_refinement_rounds = 5
        result = node_reset_signal(_make_state())
        assert "5" in result["validation_errors"][0]

    @patch("ai_export_builder.graph.nodes.reset_signal.settings")
    def test_default_rounds_in_message(self, mock_settings):
        mock_settings.max_refinement_rounds = 3
        result = node_reset_signal(_make_state())
        assert "3" in result["validation_errors"][0]
