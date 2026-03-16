"""Tests for the deterministic validation node."""
from __future__ import annotations

import pytest

from ai_export_builder.graph.nodes.validate_intent import node_validate_intent
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator


def _make_state(intent: ExportIntent | None = None, **overrides) -> ExportState:
    """Helper to build a minimal ExportState for testing."""
    state: ExportState = {
        "user_query": "test",
        "intent": intent,
        "validation_errors": [],
        "status": "parsing",
        "retry_count": 0,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestValidIntents:
    def test_valid_intent_passes(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName", "CalculateExtendedAmount"],
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []

    def test_valid_intent_with_filters(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.eq, value="Medline"),
                FilterItem(
                    column="POReleaseDate",
                    operator=FilterOperator.between,
                    value=["2026-01-01", "2026-03-15"],
                ),
            ],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []


class TestInvalidIntents:
    def test_none_intent(self):
        result = node_validate_intent(_make_state(None))
        assert len(result["validation_errors"]) > 0
        assert "No intent" in result["validation_errors"][0]

    def test_invalid_view(self):
        intent = ExportIntent(
            selected_view="fake_view_xyz",
            columns=["VendorName"],
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert any("fake_view_xyz" in e for e in result["validation_errors"])

    def test_invalid_column(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName", "NonexistentColumn"],
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert any("NonexistentColumn" in e for e in result["validation_errors"])

    def test_invalid_filter_column(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="FakeColumn", operator=FilterOperator.eq, value="x"),
            ],
        )
        result = node_validate_intent(_make_state(intent))
        assert any("FakeColumn" in e for e in result["validation_errors"])

    def test_between_requires_two_values(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["POReleaseDate"],
            filters=[
                FilterItem(
                    column="POReleaseDate",
                    operator=FilterOperator.between,
                    value=["2026-01-01"],  # only 1 value
                ),
            ],
        )
        result = node_validate_intent(_make_state(intent))
        assert any("between" in e.lower() for e in result["validation_errors"])

    def test_valid_intent_clears_errors(self):
        """After a retry, a valid intent should produce an empty errors list."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
        )
        state = _make_state(intent, validation_errors=["old error from previous run"])
        result = node_validate_intent(state)
        assert result["validation_errors"] == []
