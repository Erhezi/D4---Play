"""Tests for the meta_responder node — verify capability and field-info responses."""
from __future__ import annotations

import pytest

from ai_export_builder.graph.nodes.meta_responder import (
    _build_capabilities_response,
    _build_view_columns_response,
    _find_field_info,
    _search_field,
    node_meta_responder,
)
from ai_export_builder.graph.state import ExportState


def _make_state(
    classification: str = "meta:capabilities",
    query: str = "What can I ask?",
) -> ExportState:
    return {
        "user_query": query,
        "intent": None,
        "validation_errors": [],
        "status": "guarding",
        "retry_count": 0,
        "refinement_count": 0,
        "guardrail_passed": True,
        "guardrail_classification": classification,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }


# ---------------------------------------------------------------------------
# Node-level tests
# ---------------------------------------------------------------------------

class TestNodeMetaResponder:
    def test_capabilities_returns_meta_response_status(self):
        state = _make_state("meta:capabilities", "What can I ask?")
        result = node_meta_responder(state)
        assert result["status"] == "meta_response"
        assert "meta_response" in result
        assert len(result["meta_response"]) > 0

    def test_field_info_returns_meta_response_status(self):
        state = _make_state("meta:field_info", "What does VendorName mean?")
        result = node_meta_responder(state)
        assert result["status"] == "meta_response"
        assert "meta_response" in result

    def test_unknown_classification_falls_back_to_capabilities(self):
        state = _make_state("meta:unknown", "Help")
        result = node_meta_responder(state)
        assert result["status"] == "meta_response"
        assert "what I can help" in result["meta_response"].lower() or \
               "Here's what" in result["meta_response"]


# ---------------------------------------------------------------------------
# Capabilities response
# ---------------------------------------------------------------------------

class TestCapabilitiesResponse:
    def test_lists_at_least_one_view(self):
        resp = _build_capabilities_response()
        # Should mention at least one of our known views
        assert "PO" in resp or "Invoice" in resp or "Savings" in resp

    def test_mentions_field_queries(self):
        resp = _build_capabilities_response()
        assert "field" in resp.lower() or "column" in resp.lower()

    def test_includes_sample_questions(self):
        resp = _build_capabilities_response()
        # Sample questions are italicized
        assert "*" in resp


# ---------------------------------------------------------------------------
# Field search
# ---------------------------------------------------------------------------

class TestSearchField:
    def test_known_field_returns_description(self):
        result = _search_field("VendorName")
        # Should find VendorName in at least one view
        assert result is not None
        assert "VendorName" in result

    def test_unknown_field_returns_not_found(self):
        result = _search_field("NonExistentColumnXYZ123")
        assert result is not None
        assert "couldn't find" in result.lower()


# ---------------------------------------------------------------------------
# View columns listing
# ---------------------------------------------------------------------------

class TestViewColumnsResponse:
    def test_po_view_lists_columns(self):
        result = _build_view_columns_response("PO")
        assert result is not None
        # Should list at least one column
        assert "**" in result  # Bold column labels

    def test_unknown_view_returns_error(self):
        result = _build_view_columns_response("NonExistentView")
        assert "couldn't find" in result.lower()


# ---------------------------------------------------------------------------
# Find field info (regex extraction)
# ---------------------------------------------------------------------------

class TestFindFieldInfo:
    def test_what_does_x_mean(self):
        result = _find_field_info("what does VendorName mean?")
        assert result is not None
        assert "VendorName" in result

    def test_what_is_x(self):
        result = _find_field_info("what is VendorName?")
        assert result is not None
        assert "VendorName" in result

    def test_columns_in_view(self):
        # _find_field_info requires a field-name pattern match that also
        # triggers the view-columns sub-path.  "describe the columns in PO"
        # matches the describe regex AND the view_pattern.
        result = _find_field_info("describe the columns in PO")
        assert result is not None
        # Should route to _build_view_columns_response
        assert "**" in result

    def test_no_field_match_returns_none(self):
        result = _find_field_info("random unparseble thing")
        assert result is None
