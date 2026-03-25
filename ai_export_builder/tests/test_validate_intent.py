"""Tests for the deterministic validation node."""
from __future__ import annotations

import pytest

from ai_export_builder.graph.nodes.validate_intent import node_validate_intent
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator, SortItem


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


class TestColumnResolution:
    """Test that _resolve_columns correctly expands columns after validation."""

    def test_basic_columns_always_included(self):
        """Even if the user only asks for VendorName, basic group columns should appear."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=[],  # LLM returns empty (basic auto-included)
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        # After resolution, intent.columns should include basic group columns
        assert "PO" in intent.columns
        assert "VendorName" in intent.columns
        assert "CalculateExtendedAmount" in intent.columns

    def test_enrichment_columns_added_when_requested(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["ContractCategory_Premier"],
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        assert "ContractCategory_Premier" in intent.columns
        # Basic columns should also be present
        assert "PO" in intent.columns

    def test_filter_column_added_to_output(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=[],
            filters=[
                FilterItem(column="CostCenterText", operator=FilterOperator.like, value="%ICU%"),
            ],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        assert "CostCenterText" in intent.columns

    def test_companion_column_auto_included(self):
        """If VendorName is in output, Vendor (its companion) should be too."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["CostCenterText"],  # has companion CostCenter
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        assert "CostCenterText" in intent.columns
        assert "CostCenter" in intent.columns

    def test_no_duplicate_columns(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName", "PO"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.like, value="%med%"),
            ],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        # VendorName should appear only once
        assert intent.columns.count("VendorName") == 1
        assert intent.columns.count("PO") == 1

    def test_concept_group_expansion(self):
        """Selecting a column with concept_id should pull in all siblings."""
        # VendorName and Vendor share the same concept_id (vendor_entity)
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []
        # Both display and id columns should be present
        assert "VendorName" in intent.columns
        assert "Vendor" in intent.columns


class TestSortByValidation:
    """Validate that sort_by columns are checked against the view."""

    def test_valid_sort_column_passes(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
            sort_by=[SortItem(column="VendorName", direction="ASC")],
        )
        result = node_validate_intent(_make_state(intent))
        assert result["validation_errors"] == []

    def test_invalid_sort_column_fails(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
            sort_by=[SortItem(column="FakeColumn", direction="ASC")],
        )
        result = node_validate_intent(_make_state(intent))
        assert any("Sort column" in e and "FakeColumn" in e for e in result["validation_errors"])
