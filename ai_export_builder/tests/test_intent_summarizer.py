"""Tests for the intent_summarizer service."""
from __future__ import annotations

import pytest

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.intent_summarizer import summarize_intent
from ai_export_builder.services.registry_loader import load_registry

_registry = load_registry()


def _make_intent(**overrides) -> ExportIntent:
    defaults = {
        "selected_view": "vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        "columns": [],
        "filters": [],
        "sort_by": [],
        "warnings": [],
    }
    defaults.update(overrides)
    return ExportIntent(**defaults)


class TestSummarizeIntent:
    def test_basic_intent_has_view_name(self):
        intent = _make_intent()
        result = summarize_intent(intent, _registry)
        # Should contain the display name of the view in bold
        assert "**" in result
        assert len(result) > 5

    def test_eq_filter_summary(self):
        intent = _make_intent(filters=[
            FilterItem(column="VendorName", operator=FilterOperator.eq, value="Stryker"),
        ])
        result = summarize_intent(intent, _registry)
        assert "Stryker" in result
        assert "filtered by" in result.lower()

    def test_like_filter_summary(self):
        intent = _make_intent(filters=[
            FilterItem(column="VendorName", operator=FilterOperator.like, value="Medline"),
        ])
        result = summarize_intent(intent, _registry)
        assert "Medline" in result
        assert "matching" in result.lower()

    def test_between_filter_summary(self):
        intent = _make_intent(filters=[
            FilterItem(column="POReleaseDate", operator=FilterOperator.between,
                       value=["2026-01-01", "2026-03-31"]),
        ])
        result = summarize_intent(intent, _registry)
        assert "2026-01-01" in result
        assert "2026-03-31" in result
        assert "from" in result.lower()

    def test_in_filter_summary(self):
        intent = _make_intent(filters=[
            FilterItem(column="VendorName", operator=FilterOperator.in_,
                       value=["Medline", "Cardinal"]),
        ])
        result = summarize_intent(intent, _registry)
        assert "Medline" in result
        assert "Cardinal" in result

    def test_multiple_filters(self):
        intent = _make_intent(filters=[
            FilterItem(column="VendorName", operator=FilterOperator.eq, value="Stryker"),
            FilterItem(column="POReleaseDate", operator=FilterOperator.between,
                       value=["2026-01-01", "2026-03-31"]),
        ])
        result = summarize_intent(intent, _registry)
        assert "Stryker" in result
        assert "2026" in result

    def test_no_filters_no_filtered_by(self):
        intent = _make_intent()
        result = summarize_intent(intent, _registry)
        assert "filtered by" not in result.lower()
