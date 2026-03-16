"""Integration test — mock OpenAI, run the full LangGraph workflow.

Verifies:
1. Graph pauses at pending_approval (HITL breakpoint).
2. After resuming, the execute_export node runs and produces results.
3. Retry loop works when validation fails.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_export_builder.graph.state import ExportState, TemporalContext, UserProfile
from ai_export_builder.graph.workflow import build_graph, compile_graph
from ai_export_builder.models.intent import ExportIntent


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_GOOD_LLM_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "CalculateExtendedAmount"],
    "filters": [
        {"column": "VendorName", "operator": "eq", "value": "Medline"},
    ],
    "warnings": [],
})

_BAD_LLM_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "InvalidColumnXYZ"],
    "filters": [],
    "warnings": [],
})


def _initial_state(query: str = "Export all Medline glove spend") -> ExportState:
    return {
        "user_query": query,
        "intent": None,
        "validation_errors": [],
        "status": "parsing",
        "retry_count": 0,
        "temporal_context": TemporalContext(
            current_date="2026-03-15",
            fiscal_year_start_month=1,
        ),
        "user_profile": UserProfile(
            user_id="test_user",
            user_name="Test User",
            facilities=["ALL"],
        ),
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }


def _mock_llm_response(content: str):
    """Create a mock OpenAI client whose responses.create() returns *content*."""
    mock_cls = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = content
    mock_cls.return_value.responses.create.return_value = mock_response
    return mock_cls


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestFullWorkflowPausesAtApproval:
    """Test that the graph pauses at the HITL breakpoint with a valid intent."""

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_graph_pauses_at_pending_approval(self, mock_openai_cls):
        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        app = compile_graph()
        state = _initial_state()
        thread_cfg = {"configurable": {"thread_id": "test-thread-1"}}

        # Collect state from the stream
        collected: dict = dict(state)
        for event in app.stream(state, config=thread_cfg):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    collected.update(node_output)

        # The graph should have paused at pending_approval
        assert collected["status"] == "pending_approval"
        assert collected["intent"] is not None
        assert collected["intent"].selected_view == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert collected["validation_errors"] == []


class TestRetryLoop:
    """Verify that validation failures trigger retries before surfacing errors."""

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_invalid_then_valid_retries(self, mock_openai_cls):
        """First call returns invalid columns, second returns valid ones."""
        bad_resp = MagicMock()
        bad_resp.output_text = _BAD_LLM_RESPONSE
        good_resp = MagicMock()
        good_resp.output_text = _GOOD_LLM_RESPONSE

        mock_instance = MagicMock()
        mock_instance.responses.create.side_effect = [bad_resp, good_resp]
        mock_openai_cls.return_value = mock_instance

        app = compile_graph()
        state = _initial_state()
        thread_cfg = {"configurable": {"thread_id": "test-thread-2"}}

        collected: dict = dict(state)
        for event in app.stream(state, config=thread_cfg):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    collected.update(node_output)

        # After retry, it should reach pending_approval with the valid intent
        assert collected["status"] == "pending_approval"
        assert collected["validation_errors"] == []
        assert collected["retry_count"] == 1


class TestExecuteExportWithMockDB:
    """Test that after confirmation, the execute node runs SQL via the DB service."""

    @patch("ai_export_builder.graph.nodes.execute_export.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_full_flow_with_mock_db(self, mock_openai_cls, mock_exec_query):
        """Run the full flow: parse → validate → (pause) → execute with mock DB.

        Phase 1 uses the LangGraph stream (pauses at pending_approval).
        Phase 2 calls node_execute_export directly (mimics the Streamlit app
        which manages HITL via session state, not LangGraph checkpointing).
        """
        from ai_export_builder.graph.nodes.execute_export import node_execute_export

        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        # Mock DB returns a small DataFrame
        mock_exec_query.return_value = pd.DataFrame({
            "VendorName": ["Medline", "Medline"],
            "CalculateExtendedAmount": [1500.0, 2300.0],
        })

        # --- Phase 1: run until HITL pause ---
        app = compile_graph()
        state = _initial_state()
        thread_cfg = {"configurable": {"thread_id": "test-thread-3"}}

        collected: dict = dict(state)
        for event in app.stream(state, config=thread_cfg):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    collected.update(node_output)

        assert collected["status"] == "pending_approval"
        assert collected["intent"] is not None

        # --- Phase 2: directly call execute_export (same as Streamlit app) ---
        exec_result = node_execute_export(collected)
        collected.update(exec_result)

        assert collected["status"] == "completed"
        assert collected["result_row_count"] == 2
        assert collected["result_df"] is not None
        assert len(collected["result_df"]) == 2
        mock_exec_query.assert_called_once()
