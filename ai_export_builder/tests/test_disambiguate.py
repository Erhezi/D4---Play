"""Tests for the disambiguation node — mock DB to verify logic."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_export_builder.graph.nodes.disambiguate import node_disambiguate
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator


def _make_state(intent: ExportIntent | None = None, **overrides) -> ExportState:
    state: ExportState = {
        "user_query": "test",
        "intent": intent,
        "validation_errors": [],
        "status": "parsing",
        "retry_count": 0,
        "disambiguation_needed": False,
        "disambiguation_results": [],
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class TestDisambiguateNode:
    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    def test_like_on_vendor_triggers_disambiguation(self, mock_exec):
        """LIKE filter on VendorName (which has companion Vendor) should trigger."""
        mock_exec.return_value = pd.DataFrame({
            "VendorName": ["Medline Industries, LP", "Medline ReNewal"],
            "Vendor": ["100123", "100456"],
        })
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.like, value="%medline%"),
            ],
        )
        result = node_disambiguate(_make_state(intent))
        assert result["disambiguation_needed"] is True
        assert len(result["disambiguation_results"]) == 1
        matches = result["disambiguation_results"][0]["matches"]
        assert len(matches) == 2
        assert matches[0]["text"] == "Medline Industries, LP"
        assert matches[0]["id"] == "100123"

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    def test_eq_on_vendor_triggers_disambiguation(self, mock_exec):
        """eq filter on VendorName should also trigger disambiguation (with wildcard wrapping)."""
        mock_exec.return_value = pd.DataFrame({
            "VendorName": ["Medline Industries, LP"],
            "Vendor": ["100123"],
        })
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.eq, value="Medline"),
            ],
        )
        result = node_disambiguate(_make_state(intent))
        assert result["disambiguation_needed"] is True
        # Verify the LIKE value got wildcards
        call_args = mock_exec.call_args
        assert "%Medline%" in call_args[1]["params"] or "%Medline%" in call_args.kwargs.get("params", [])

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    def test_like_on_non_companion_skips(self, mock_exec):
        """LIKE filter on ItemDescription (no companion) should not trigger."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["ItemDescription"],
            filters=[
                FilterItem(column="ItemDescription", operator=FilterOperator.like, value="%glove%"),
            ],
        )
        result = node_disambiguate(_make_state(intent))
        assert result["disambiguation_needed"] is False
        assert result["disambiguation_results"] == []
        mock_exec.assert_not_called()

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    def test_zero_matches_sets_not_needed(self, mock_exec):
        """If disambiguation query returns no rows, disambiguation_needed should be False."""
        mock_exec.return_value = pd.DataFrame({"VendorName": [], "Vendor": []})
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.like, value="%zzzzz%"),
            ],
        )
        result = node_disambiguate(_make_state(intent))
        assert result["disambiguation_needed"] is False
        assert len(result["disambiguation_results"]) == 1
        assert result["disambiguation_results"][0]["matches"] == []

    def test_none_intent_returns_not_needed(self):
        result = node_disambiguate(_make_state(None))
        assert result["disambiguation_needed"] is False

    @patch("ai_export_builder.graph.nodes.disambiguate.execute_query_for_view")
    def test_between_operator_skips(self, mock_exec):
        """Non-LIKE/eq operators should not trigger disambiguation."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["POReleaseDate"],
            filters=[
                FilterItem(
                    column="VendorName",
                    operator=FilterOperator.in_,
                    value=["Medline", "Stryker"],
                ),
            ],
        )
        result = node_disambiguate(_make_state(intent))
        assert result["disambiguation_needed"] is False
        mock_exec.assert_not_called()
