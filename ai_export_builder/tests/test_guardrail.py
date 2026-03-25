"""Tests for the guardrail node — mock OpenAI, verify classification."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ai_export_builder.graph.nodes.guardrail import (
    _build_error_response,
    _build_few_shot_section,
    _build_system_prompt,
    node_guardrail,
)
from ai_export_builder.graph.state import ExportState


def _make_state(query: str = "Show me all Stryker PO lines") -> ExportState:
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


def _mock_classification(classification: str, reason: str = "test reason"):
    """Return a mock OpenAI class whose responses.create() returns a classification JSON."""
    content = json.dumps({"classification": classification, "reason": reason})
    mock_cls = MagicMock()
    mock_resp = MagicMock()
    mock_resp.output_text = content
    mock_cls.return_value.responses.create.return_value = mock_resp
    return mock_cls


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestGuardrailAllowed:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_allowed_query_passes(self, mock_openai_cls):
        mock_openai_cls.side_effect = _mock_classification("allowed").__class__
        # Simpler: just wire up directly
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({"classification": "allowed", "reason": "valid export"})
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("Show me all Stryker PO lines from last month")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is True
        assert result["status"] == "guarding"

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_allowed_no_validation_errors(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({"classification": "allowed", "reason": "ok"})
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        result = node_guardrail(_make_state())
        assert "validation_errors" not in result


class TestGuardrailBlockedDML:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_dml_blocked(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:dml_or_injection",
            "reason": "User wants to delete data",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("Delete all invoices from last year")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False
        assert result["status"] == "failed"
        assert len(result["validation_errors"]) == 1
        assert "read-only" in result["validation_errors"][0].lower() or \
               "cannot" in result["validation_errors"][0].lower()

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_injection_blocked(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:dml_or_injection",
            "reason": "SQL injection attempt",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("'; DROP TABLE vendors; --")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False
        assert result["status"] == "failed"


class TestGuardrailBlockedPHI:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_phi_blocked(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:phi_pii",
            "reason": "User asks for patient data",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("Export patient MRNs associated with surgical supplies")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False
        assert result["status"] == "failed"
        assert "patient health information" in result["validation_errors"][0].lower() or \
               "phi" in result["validation_errors"][0].lower() or \
               "pii" in result["validation_errors"][0].lower()

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_pii_blocked(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:phi_pii",
            "reason": "SSN request",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("Give me the SSNs of all vendors")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False


class TestGuardrailBlockedOutOfScope:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_out_of_scope_blocked(self, mock_openai_cls):
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({
            "classification": "blocked:out_of_scope",
            "reason": "Weather question",
        })
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        state = _make_state("What's the weather today?")
        result = node_guardrail(state)

        assert result["guardrail_passed"] is False
        assert result["status"] == "failed"
        assert "what you can query" in result["validation_errors"][0].lower() or \
               "here" in result["validation_errors"][0].lower()


# ---------------------------------------------------------------------------
# Error / fallback tests
# ---------------------------------------------------------------------------


class TestGuardrailFallback:
    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_llm_failure_defaults_to_passthrough(self, mock_openai_cls):
        """If the LLM call fails, the guardrail should pass through."""
        mock_openai_cls.return_value.responses.create.side_effect = RuntimeError("API down")

        result = node_guardrail(_make_state())

        assert result["guardrail_passed"] is True
        assert result["status"] == "guarding"

    @patch("ai_export_builder.graph.nodes.guardrail.settings")
    def test_no_api_key_defaults_to_passthrough(self, mock_settings):
        """Without an API key, guardrail passes through."""
        mock_settings.openai_api_key = ""

        result = node_guardrail(_make_state())

        assert result["guardrail_passed"] is True

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_malformed_json_defaults_to_passthrough(self, mock_openai_cls):
        """If LLM returns invalid JSON, guardrail passes through."""
        mock_resp = MagicMock()
        mock_resp.output_text = "not json"
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        result = node_guardrail(_make_state())

        assert result["guardrail_passed"] is True

    @patch("ai_export_builder.graph.nodes.guardrail.OpenAI")
    def test_missing_classification_key_defaults_to_allowed(self, mock_openai_cls):
        """If LLM returns JSON without 'classification', defaults to allowed."""
        mock_resp = MagicMock()
        mock_resp.output_text = json.dumps({"reason": "no classification key"})
        mock_openai_cls.return_value.responses.create.return_value = mock_resp

        result = node_guardrail(_make_state())

        assert result["guardrail_passed"] is True


# ---------------------------------------------------------------------------
# Error message and prompt construction tests
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_blocked_message_includes_topics(self):
        msg = _build_error_response("blocked:dml_or_injection")
        assert "what you can query" in msg.lower() or "here" in msg.lower()

    def test_blocked_message_includes_user_message(self):
        msg = _build_error_response("blocked:phi_pii")
        # Should contain the user_message from common_invalid_queries.yaml
        assert "procurement" in msg.lower() or "patient" in msg.lower()

    def test_unknown_category_gives_generic_message(self):
        msg = _build_error_response("blocked:unknown_category")
        assert "cannot be processed" in msg.lower()


class TestPromptConstruction:
    def test_system_prompt_contains_topics(self):
        prompt = _build_system_prompt()
        assert "Available Data Topics" in prompt
        # Should reference at least one view
        assert "PO" in prompt or "AP" in prompt or "Savings" in prompt

    def test_system_prompt_contains_few_shot(self):
        prompt = _build_system_prompt()
        assert "blocked:" in prompt

    def test_few_shot_section_non_empty(self):
        section = _build_few_shot_section()
        assert len(section) > 0
        assert "blocked:" in section
