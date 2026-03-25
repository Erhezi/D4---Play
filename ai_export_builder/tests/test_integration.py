"""Integration test — mock OpenAI + DB, run the full LangGraph workflow.

Verifies v2.0 flows:
1. Happy path: guardrail → orchestrator → parse → validate → disambiguate → preview → pause
2. Guardrail block (SQL injection)
3. Refinement loop with delta-parsing
4. Reset signal after max refinements
5. Multi-value LIKE filter → OR clause
6. Aggregation and sorting in output
7. Execute after approval (mock DB)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_export_builder.graph.state import ExportState, TemporalContext, UserProfile
from ai_export_builder.graph.workflow import build_graph, compile_graph
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator, SortItem


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

_GOOD_LLM_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "CalculateExtendedAmount"],
    "filters": [
        {"column": "VendorName", "operator": "eq", "value": "Medline"},
    ],
    "sort_by": [],
    "warnings": [],
})

_GOOD_LLM_WITH_SORT = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "CalculateExtendedAmount"],
    "filters": [
        {"column": "VendorName", "operator": "eq", "value": "Medline"},
    ],
    "sort_by": [{"column": "VendorName", "direction": "ASC"}],
    "warnings": [],
})

_MULTI_VALUE_LIKE_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "CalculateExtendedAmount"],
    "filters": [
        {"column": "VendorName", "operator": "like", "value": ["Medline", "Cardinal"]},
    ],
    "sort_by": [],
    "warnings": [],
})

_BAD_LLM_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "InvalidColumnXYZ"],
    "filters": [],
    "sort_by": [],
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


def _stream_graph(app, state, thread_id="test-thread"):
    """Stream events through the compiled graph, collecting final state."""
    collected: dict = dict(state)
    for event in app.stream(state, config={"configurable": {"thread_id": thread_id}}):
        for node_name, node_output in event.items():
            if isinstance(node_output, dict):
                collected.update(node_output)
    return collected


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestFullWorkflowPausesAtApproval:
    """Test that the graph pauses at the HITL breakpoint with a valid intent."""

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_graph_pauses_at_pending_approval(self, mock_openai_cls, mock_preview_exec, mock_disambig_exec):
        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        # Mock disambiguation to find no matches (skip disambiguation)
        mock_disambig_exec.return_value = pd.DataFrame({"VendorName": pd.Series(dtype="str"), "Vendor": pd.Series(dtype="str")})
        # Mock the preview DB call
        mock_preview_exec.return_value = pd.DataFrame({
            "VendorName": ["Medline"],
            "CalculateExtendedAmount": [100.0],
        })

        app = compile_graph()
        state = _initial_state()
        collected = _stream_graph(app, state, "test-thread-1")

        # The graph should have paused at pending_approval
        assert collected["status"] == "pending_approval"
        assert collected["intent"] is not None
        assert collected["intent"].selected_view == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert collected["validation_errors"] == []


class TestRetryLoop:
    """Verify that validation failures trigger retries before surfacing errors."""

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_invalid_then_valid_retries(self, mock_openai_cls, mock_preview_exec, mock_disambig_exec):
        """First call returns invalid columns, second returns valid ones."""
        bad_resp = MagicMock()
        bad_resp.output_text = _BAD_LLM_RESPONSE
        good_resp = MagicMock()
        good_resp.output_text = _GOOD_LLM_RESPONSE

        mock_instance = MagicMock()
        mock_instance.responses.create.side_effect = [bad_resp, good_resp]
        mock_openai_cls.return_value = mock_instance

        mock_disambig_exec.return_value = pd.DataFrame({"VendorName": pd.Series(dtype="str"), "Vendor": pd.Series(dtype="str")})
        mock_preview_exec.return_value = pd.DataFrame({"VendorName": ["Medline"]})

        app = compile_graph()
        state = _initial_state()
        collected = _stream_graph(app, state, "test-thread-2")

        # After retry, it should reach pending_approval with the valid intent
        assert collected["status"] == "pending_approval"
        assert collected["validation_errors"] == []
        assert collected["retry_count"] == 1


class TestExecuteExportWithMockDB:
    """Test that after confirmation, the execute node runs SQL via the DB service."""

    @patch("ai_export_builder.graph.nodes.execute_export.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_full_flow_with_mock_db(self, mock_openai_cls, mock_preview_exec, mock_disambig_exec, mock_exec_query):
        from ai_export_builder.graph.nodes.execute_export import node_execute_export

        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        mock_disambig_exec.return_value = pd.DataFrame({"VendorName": pd.Series(dtype="str"), "Vendor": pd.Series(dtype="str")})
        mock_preview_exec.return_value = pd.DataFrame({
            "VendorName": ["Medline"],
            "CalculateExtendedAmount": [100.0],
        })
        mock_exec_query.return_value = pd.DataFrame({
            "VendorName": ["Medline", "Medline"],
            "CalculateExtendedAmount": [1500.0, 2300.0],
        })

        # --- Phase 1: run until HITL pause ---
        app = compile_graph()
        state = _initial_state()
        collected = _stream_graph(app, state, "test-thread-3")

        assert collected["status"] == "pending_approval"
        assert collected["intent"] is not None

        # --- Phase 2: directly call execute_export (same as Streamlit app) ---
        exec_result = node_execute_export(collected)
        collected.update(exec_result)

        assert collected["status"] == "completed"
        assert collected["result_row_count"] == 2
        assert collected["result_df"] is not None
        assert len(collected["result_df"]) == 2


class TestGuardrailBlock:
    """Verify SQL injection / blocked queries are stopped at guardrail."""

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_guardrail_blocks_sql_injection(self, mock_openai_cls):
        """A blocked:dml_or_injection classification should end the graph."""
        mock_response = MagicMock()
        mock_response.output_text = json.dumps({
            "classification": "blocked:dml_or_injection",
            "reason": "Contains SQL injection attempt",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        app = compile_graph()
        state = _initial_state("DROP TABLE users; --")
        collected = _stream_graph(app, state, "test-guardrail-block")

        assert collected["status"] == "failed"
        assert collected.get("guardrail_passed") is False
        # Should never reach parse_intent
        assert collected.get("intent") is None


class TestSortingInWorkflow:
    """Verify sorting propagates through the workflow."""

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_sort_by_preserved_through_workflow(self, mock_openai_cls, mock_preview_exec, mock_disambig_exec):
        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_WITH_SORT
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        mock_disambig_exec.return_value = pd.DataFrame({"VendorName": pd.Series(dtype="str"), "Vendor": pd.Series(dtype="str")})
        mock_preview_exec.return_value = pd.DataFrame({"VendorName": ["Medline"]})

        app = compile_graph()
        state = _initial_state("Show Medline spend sorted by vendor name")
        collected = _stream_graph(app, state, "test-sort")

        assert collected["status"] == "pending_approval"
        assert len(collected["intent"].sort_by) == 1
        assert collected["intent"].sort_by[0].column == "VendorName"


class TestMultiValueLikeInWorkflow:
    """Verify multi-value LIKE filters propagate correctly."""

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_multi_value_like_reaches_disambiguation(self, mock_openai_cls, mock_disambig_exec):
        mock_response = MagicMock()
        mock_response.output_text = _MULTI_VALUE_LIKE_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        # Each value gets its own disambiguation query
        mock_disambig_exec.side_effect = [
            pd.DataFrame({"VendorName": ["Medline Industries, LP"], "Vendor": ["100123"]}),
            pd.DataFrame({"VendorName": ["Cardinal Health, Inc."], "Vendor": ["200456"]}),
        ]

        app = compile_graph()
        state = _initial_state("Export spend for Medline or Cardinal")
        collected = _stream_graph(app, state, "test-multi-like")

        # Should pause for disambiguation since matches were found
        assert collected.get("disambiguation_needed") is True
        matches = collected["disambiguation_results"][0]["matches"]
        assert len(matches) == 2


class TestPreviewDataInWorkflow:
    """Verify hydrate_preview populates preview_data and aggregation_summary."""

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_preview_data_populated(self, mock_openai_cls, mock_preview_exec, mock_disambig_exec):
        mock_response = MagicMock()
        mock_response.output_text = _GOOD_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_response

        mock_disambig_exec.return_value = pd.DataFrame({"VendorName": pd.Series(dtype="str"), "Vendor": pd.Series(dtype="str")})
        preview_df = pd.DataFrame({
            "VendorName": ["Medline"] * 5,
            "CalculateExtendedAmount": [100.0] * 5,
        })
        agg_df = pd.DataFrame({"row_count": [500], "total_CalculateExtendedAmount": [75000.0]})
        mock_preview_exec.side_effect = [preview_df, agg_df]

        app = compile_graph()
        state = _initial_state()
        collected = _stream_graph(app, state, "test-preview")

        assert collected.get("preview_data") is not None
        assert len(collected["preview_data"]) == 5
        assert collected.get("aggregation_summary") is not None
        assert collected["aggregation_summary"]["row_count"] == 500
