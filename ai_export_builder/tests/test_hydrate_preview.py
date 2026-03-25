"""Tests for the hydrate_preview node — mock DB to verify preview + aggregation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ai_export_builder.graph.nodes.hydrate_preview import node_hydrate_preview
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator


def _make_state(intent: ExportIntent | None = None, **overrides) -> ExportState:
    base: dict = {
        "user_query": "test",
        "intent": intent,
        "validation_errors": [],
        "status": "previewing",
        "retry_count": 0,
        "refinement_count": 1,
        "guardrail_passed": True,
        "previous_intent": None,
        "preview_data": None,
        "aggregation_summary": None,
        "temporal_context": {"current_date": "2026-03-15", "fiscal_year_start_month": 1},
        "user_profile": {"user_id": "test", "user_name": "Test", "facilities": ["ALL"]},
        "result_df": None,
        "result_row_count": 0,
        "error_message": "",
    }
    base.update(overrides)
    return base


def _make_intent(**overrides) -> ExportIntent:
    defaults = {
        "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        "columns": ["VendorName", "CalculateExtendedAmount"],
        "filters": [],
    }
    defaults.update(overrides)
    return ExportIntent(**defaults)


class TestPreviewData:
    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_returns_preview_rows(self, mock_exec):
        preview_df = pd.DataFrame({
            "VendorName": ["Medline", "Stryker"],
            "CalculateExtendedAmount": [1000.0, 2000.0],
        })
        agg_df = pd.DataFrame({"row_count": [500], "total_CalculateExtendedAmount": [150000.0]})
        mock_exec.side_effect = [preview_df, agg_df]

        result = node_hydrate_preview(_make_state(_make_intent()))

        assert result["preview_data"] is not None
        assert len(result["preview_data"]) == 2
        assert result["preview_data"][0]["VendorName"] == "Medline"
        assert result["status"] == "previewing"

    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_returns_aggregation_summary(self, mock_exec):
        preview_df = pd.DataFrame({"VendorName": ["A"], "CalculateExtendedAmount": [100.0]})
        agg_df = pd.DataFrame({"row_count": [1000], "total_CalculateExtendedAmount": [500000.0]})
        mock_exec.side_effect = [preview_df, agg_df]

        result = node_hydrate_preview(_make_state(_make_intent()))

        assert result["aggregation_summary"] is not None
        assert result["aggregation_summary"]["row_count"] == 1000
        assert "CalculateExtendedAmount" in result["aggregation_summary"]["sums"]
        assert result["aggregation_summary"]["sums"]["CalculateExtendedAmount"] == 500000.0

    def test_none_intent_returns_failed(self):
        result = node_hydrate_preview(_make_state(None))
        assert result["preview_data"] is None
        assert result["aggregation_summary"] is None
        assert result["status"] == "failed"

    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_preview_db_failure_non_fatal(self, mock_exec):
        """If preview query fails, aggregation should still run."""
        agg_df = pd.DataFrame({"row_count": [100], "total_CalculateExtendedAmount": [5000.0]})
        mock_exec.side_effect = [RuntimeError("DB down"), agg_df]

        result = node_hydrate_preview(_make_state(_make_intent()))

        assert result["preview_data"] is None
        assert result["aggregation_summary"] is not None
        assert result["aggregation_summary"]["row_count"] == 100

    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_aggregation_db_failure_non_fatal(self, mock_exec):
        """If aggregation query fails, preview data should still be returned."""
        preview_df = pd.DataFrame({"VendorName": ["A"], "CalculateExtendedAmount": [10.0]})
        mock_exec.side_effect = [preview_df, RuntimeError("DB down")]

        result = node_hydrate_preview(_make_state(_make_intent()))

        assert result["preview_data"] is not None
        assert result["aggregation_summary"] is None

    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_both_fail_returns_previewing_status(self, mock_exec):
        mock_exec.side_effect = [RuntimeError("fail1"), RuntimeError("fail2")]

        result = node_hydrate_preview(_make_state(_make_intent()))

        assert result["preview_data"] is None
        assert result["aggregation_summary"] is None
        assert result["status"] == "previewing"

    @patch("ai_export_builder.graph.nodes.hydrate_preview.execute_query_for_view")
    def test_preview_uses_20_row_limit(self, mock_exec):
        """Verify the preview query is built with max_rows=20."""
        mock_exec.return_value = pd.DataFrame({"VendorName": [], "CalculateExtendedAmount": []})

        node_hydrate_preview(_make_state(_make_intent()))

        # First call is the preview; check that the SQL includes FETCH NEXT ? ROWS
        preview_call = mock_exec.call_args_list[0]
        params = preview_call[1].get("params", preview_call.kwargs.get("params", []))
        # The last param in build_query is the row limit
        assert params[-1] == 20
