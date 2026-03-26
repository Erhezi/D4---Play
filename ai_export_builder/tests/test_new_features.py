"""Tests for the new guardrail meta-classification and parse_intent guidance_needed features."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_export_builder.graph.nodes.guardrail import node_guardrail
from ai_export_builder.graph.nodes.parse_intent import node_parse_intent
from ai_export_builder.graph.state import ExportState


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_guardrail_state(query: str = "Show me all Stryker PO lines") -> ExportState:
    return {
        "user_query": query,
        "intent": None,
        "validation_errors": [],
        "status": "guarding",
        "retry_count": 0,
        "refinement_count": 0,
        "guardrail_passed": False,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }


def _make_parse_state(query: str = "Give me some data") -> ExportState:
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


def _mock_openai_response(content: str):
    mock_cls = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = content
    mock_cls.return_value.responses.create.return_value = mock_resp
    return mock_cls


# ---------------------------------------------------------------------------
# Guardrail: meta-classification
# ---------------------------------------------------------------------------

class TestGuardrailMetaClassification:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_meta_capabilities_passes_guardrail(self, mock_openai_cls):
        """Meta:capabilities queries should pass the guardrail."""
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "meta:capabilities",
            "reason": "User asking about tool capabilities",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_guardrail_state("What can I ask?")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is True
        assert result["guardrail_classification"] == "meta:capabilities"
        assert result["status"] == "guarding"

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_meta_field_info_passes_guardrail(self, mock_openai_cls):
        """Meta:field_info queries should pass the guardrail."""
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "meta:field_info",
            "reason": "User asking about a field",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_guardrail_state("What does VendorName mean?")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is True
        assert result["guardrail_classification"] == "meta:field_info"

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_allowed_has_classification(self, mock_openai_cls):
        """Allowed queries should also include guardrail_classification."""
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "allowed",
            "reason": "Valid data query",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_guardrail_state("Show me Stryker PO lines")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is True
        assert result["guardrail_classification"] == "allowed"

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_blocked_has_classification(self, mock_openai_cls):
        """Blocked queries should include the classification label."""
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:out_of_scope",
            "reason": "Weather question",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_guardrail_state("What's the weather?")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False
        assert result["guardrail_classification"] == "blocked:out_of_scope"

    def test_refinement_gets_allowed_classification(self):
        """Refinements should auto-pass with 'allowed' classification."""
        from ai_export_builder.models.intent import ExportIntent
        state = _make_guardrail_state("Add VendorNumber to the export")
        state["previous_intent"] = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=[], filters=[], sort_by=[], warnings=[],
        )
        result = node_guardrail(state)

        assert result["guardrail_passed"] is True
        assert result["guardrail_classification"] == "allowed"


# ---------------------------------------------------------------------------
# Parse intent: guidance_needed
# ---------------------------------------------------------------------------

class TestParseIntentGuidance:
    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_guidance_needed_returns_needs_guidance_status(self, mock_openai_cls):
        """When LLM returns guidance_needed, node should set status to needs_guidance."""
        mock_openai_cls_inst = _mock_openai_response(json.dumps({
            "guidance_needed": True,
            "guidance_question": "Which dataset would you like: Purchase Orders, Invoices, or Savings?",
            "selected_view": "",
            "columns": [],
            "filters": [],
            "sort_by": [],
            "warnings": [],
        }))
        mock_openai_cls.return_value = mock_openai_cls_inst.return_value

        state = _make_parse_state("I need some data")
        result = node_parse_intent(state)

        assert result["status"] == "needs_guidance"
        assert "guidance_question" in result
        assert len(result["guidance_question"]) > 0

    @patch("ai_export_builder.graph.nodes.parse_intent.OpenAI")
    def test_normal_parse_no_guidance(self, mock_openai_cls):
        """When LLM returns a normal intent (no guidance_needed), status should be parsing."""
        mock_openai_cls_inst = _mock_openai_response(json.dumps({
            "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            "columns": [],
            "filters": [
                {"column": "VendorName", "operator": "eq", "value": "Stryker"},
            ],
            "sort_by": [],
            "warnings": [],
        }))
        mock_openai_cls.return_value = mock_openai_cls_inst.return_value

        state = _make_parse_state("Show me Stryker PO lines")
        result = node_parse_intent(state)

        assert result.get("status") != "needs_guidance"
        assert result["intent"] is not None
