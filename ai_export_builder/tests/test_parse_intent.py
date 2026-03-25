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


def _mock_openai_response(content: str):
    """Build a mock OpenAI client whose responses.create() returns *content*."""
    mock_cls = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = content
    mock_cls.return_value.responses.create.return_value = mock_resp
    return mock_cls


class TestParseIntent:
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_successful_parse(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is not None
        assert result["intent"].selected_view == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert "VendorName" in result["intent"].columns
        assert len(result["intent"].filters) == 2
        assert result["validation_errors"] == []

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_llm_failure_returns_error(self, mock_openai_cls):
        mock_openai_cls.return_value.responses.create.side_effect = RuntimeError("API unavailable")

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is None
        assert len(result["validation_errors"]) > 0
        assert "LLM parsing error" in result["validation_errors"][0]
        assert result["status"] == "failed"

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_invalid_json_returns_error(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = "not json at all"
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is None
        assert len(result["validation_errors"]) > 0

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_retry_includes_validation_feedback(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        state["validation_errors"] = ["Column 'FakeCol' does not exist"]
        state["retry_count"] = 1

        result = node_parse_intent(state)
        # Check that the LLM was called (it should receive feedback in the user message)
        call_args = mock_openai_cls.return_value.responses.create.call_args
        messages = call_args[1]["input"]
        user_msg = messages[1]["content"]
        assert "FakeCol" in user_msg

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_certificate_error_returns_actionable_message(self, mock_openai_cls):
        mock_openai_cls.return_value.responses.create.side_effect = RuntimeError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
        )

        state = _make_state()
        result = node_parse_intent(state)

        assert result["intent"] is None
        assert len(result["validation_errors"]) == 1
        assert "OPENAI_CA_BUNDLE" in result["validation_errors"][0]

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_successful_parse_sets_previous_intent(self, mock_openai_cls):
        """After a successful parse, previous_intent should be set."""
        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        result = node_parse_intent(state)

        assert result["previous_intent"] is not None
        assert result["previous_intent"].selected_view == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"


class TestDeltaParsing:
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_delta_context_included_in_prompt(self, mock_openai_cls):
        """When previous_intent exists, it should be included in the user message."""
        from ai_export_builder.models.intent import ExportIntent

        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        prev_intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
        )
        state = _make_state("Also add the amount column")
        state["previous_intent"] = prev_intent

        node_parse_intent(state)

        call_args = mock_openai_cls.return_value.responses.create.call_args
        messages = call_args[1]["input"]
        user_msg = messages[1]["content"]
        assert "CURRENT EXPORT STATE" in user_msg
        assert "REFINEMENT REQUEST" in user_msg
        assert "VendorName" in user_msg

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_no_delta_context_on_first_run(self, mock_openai_cls):
        """Without previous_intent, no delta context should be in the prompt."""
        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        node_parse_intent(state)

        call_args = mock_openai_cls.return_value.responses.create.call_args
        messages = call_args[1]["input"]
        user_msg = messages[1]["content"]
        assert "CURRENT EXPORT STATE" not in user_msg

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_sort_by_in_json_schema_prompt(self, mock_openai_cls):
        """The system prompt should include sort_by in the JSON schema."""
        mock_resp = MagicMock()
        mock_resp.output_text = _MOCK_LLM_RESPONSE
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state()
        node_parse_intent(state)

        call_args = mock_openai_cls.return_value.responses.create.call_args
        messages = call_args[1]["input"]
        system_msg = messages[0]["content"]
        assert "sort_by" in system_msg
