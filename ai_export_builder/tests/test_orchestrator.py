"""Tests for the orchestrator node — routing logic and refinement counting."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_export_builder.graph.nodes.orchestrator import node_orchestrator
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent


def _make_state(**overrides) -> ExportState:
    base: dict = {
        "user_query": "Show me PO lines",
        "intent": None,
        "validation_errors": [],
        "status": "orchestrating",
        "retry_count": 0,
        "refinement_count": 0,
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


def _make_intent(**overrides) -> ExportIntent:
    defaults = {
        "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        "columns": ["VendorName"],
        "filters": [],
    }
    defaults.update(overrides)
    return ExportIntent(**defaults)


class TestOrchestratorFirstRun:
    def test_first_run_sets_count_to_1(self):
        result = node_orchestrator(_make_state())
        assert result["refinement_count"] == 1
        assert result["status"] == "orchestrating"

    def test_first_run_with_no_previous_intent(self):
        result = node_orchestrator(_make_state(previous_intent=None))
        assert result["refinement_count"] == 1


class TestOrchestratorRefinement:
    def test_second_round_increments_count(self):
        state = _make_state(
            previous_intent=_make_intent(),
            refinement_count=1,
        )
        result = node_orchestrator(state)
        assert result["refinement_count"] == 2
        assert result["status"] == "orchestrating"

    def test_third_round_increments_count(self):
        state = _make_state(
            previous_intent=_make_intent(),
            refinement_count=2,
        )
        result = node_orchestrator(state)
        assert result["refinement_count"] == 3
        assert result["status"] == "orchestrating"

    @patch("ai_export_builder.graph.nodes.orchestrator.settings")
    def test_exceeds_max_rounds_triggers_reset(self, mock_settings):
        mock_settings.max_refinement_rounds = 3
        state = _make_state(
            previous_intent=_make_intent(),
            refinement_count=3,
        )
        result = node_orchestrator(state)
        assert result["refinement_count"] == 4
        assert result["status"] == "reset"

    @patch("ai_export_builder.graph.nodes.orchestrator.settings")
    def test_custom_max_rounds(self, mock_settings):
        mock_settings.max_refinement_rounds = 5
        state = _make_state(
            previous_intent=_make_intent(),
            refinement_count=4,
        )
        result = node_orchestrator(state)
        assert result["refinement_count"] == 5
        assert result["status"] == "orchestrating"


class TestOrchestratorGuardrailBlocked:
    def test_guardrail_blocked_routes_to_failed(self):
        state = _make_state(guardrail_passed=False)
        result = node_orchestrator(state)
        assert result["status"] == "failed"

    def test_guardrail_blocked_does_not_increment_count(self):
        state = _make_state(guardrail_passed=False, refinement_count=1)
        result = node_orchestrator(state)
        assert "refinement_count" not in result
