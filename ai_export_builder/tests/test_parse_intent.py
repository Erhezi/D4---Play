"""Tests for the parse_intent node — mock OpenAI, verify structured output."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_export_builder.graph.nodes.parse_intent import node_parse_intent
from ai_export_builder.graph.state import ExportState


def _make_state(query: str = "Show me all Stryker purchases last month") -> ExportState:
    return {
        "user_query": query,
        "intent": None,
        "validation_errors": [],
        "status": "parsing",
        "retry_count": 0,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }


_MOCK_LLM_RESPONSE = json.dumps({
    "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
    "columns": ["VendorName", "CalculateExtendedAmount", "POReleaseDate"],
    "filters": [
        {"column": "VendorName", "operator": "eq", "value": "Stryker"},
        {"column": "POReleaseDate", "operator": "between", "value": ["2026-02-01", "2026-02-28"]},
    ],
    "warnings": [],
})


class TestParseIntent:
    @patch("ai_export_builder.graph.nodes.parse_intent.ChatOpenAI")
    def test_successful_parse(self, mock_chat_cls):
        mock_response = MagicMock()
        mock_response.content = _MOCK_LLM_RESPONSE
        mock_chat_cls.return_value.invoke.return_value = mock_response

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is not None
        assert result["intent"].selected_view == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert "VendorName" in result["intent"].columns
        assert len(result["intent"].filters) == 2
        assert result["validation_errors"] == []

    @patch("ai_export_builder.graph.nodes.parse_intent.ChatOpenAI")
    def test_llm_failure_returns_error(self, mock_chat_cls):
        mock_chat_cls.return_value.invoke.side_effect = RuntimeError("API unavailable")

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is None
        assert len(result["validation_errors"]) > 0
        assert "LLM parsing error" in result["validation_errors"][0]
        assert result["status"] == "failed"

    @patch("ai_export_builder.graph.nodes.parse_intent.ChatOpenAI")
    def test_invalid_json_returns_error(self, mock_chat_cls):
        mock_response = MagicMock()
        mock_response.content = "not json at all"
        mock_chat_cls.return_value.invoke.return_value = mock_response

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is None
        assert len(result["validation_errors"]) > 0

    @patch("ai_export_builder.graph.nodes.parse_intent.ChatOpenAI")
    def test_retry_includes_validation_feedback(self, mock_chat_cls):
        mock_response = MagicMock()
        mock_response.content = _MOCK_LLM_RESPONSE
        mock_chat_cls.return_value.invoke.return_value = mock_response

        state = _make_state()
        state["validation_errors"] = ["Column 'FakeCol' does not exist"]
        state["retry_count"] = 1

        result = node_parse_intent(state)
        # Check that the LLM was called (it should receive feedback in the user message)
        call_args = mock_chat_cls.return_value.invoke.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "FakeCol" in user_msg
