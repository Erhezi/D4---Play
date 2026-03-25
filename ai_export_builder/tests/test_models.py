"""Tests for Pydantic models — SortItem, FilterOperator.label, multi-value FilterItem."""
from __future__ import annotations

import pytest

from ai_export_builder.models.intent import (
    ExportIntent,
    FilterItem,
    FilterOperator,
    SortItem,
)


# ------------------------------------------------------------------
# FilterOperator.label
# ------------------------------------------------------------------

class TestFilterOperatorLabel:
    def test_eq_label(self):
        assert FilterOperator.eq.label == "="

    def test_neq_label(self):
        assert FilterOperator.neq.label == "≠"

    def test_gt_label(self):
        assert FilterOperator.gt.label == ">"

    def test_gte_label(self):
        assert FilterOperator.gte.label == "≥"

    def test_lt_label(self):
        assert FilterOperator.lt.label == "<"

    def test_lte_label(self):
        assert FilterOperator.lte.label == "≤"

    def test_like_label(self):
        assert FilterOperator.like.label == "contains"

    def test_in_label(self):
        assert FilterOperator.in_.label == "is one of"

    def test_between_label(self):
        assert FilterOperator.between.label == "between"


# ------------------------------------------------------------------
# SortItem
# ------------------------------------------------------------------

class TestSortItem:
    def test_default_direction(self):
        s = SortItem(column="VendorName")
        assert s.direction == "ASC"

    def test_explicit_desc(self):
        s = SortItem(column="CalculateExtendedAmount", direction="DESC")
        assert s.direction == "DESC"

    def test_column_required(self):
        with pytest.raises(Exception):
            SortItem()  # type: ignore[call-arg]


# ------------------------------------------------------------------
# FilterItem with multi-value (list) for like operator
# ------------------------------------------------------------------

class TestFilterItemMultiValue:
    def test_single_string_value(self):
        f = FilterItem(column="VendorName", operator=FilterOperator.like, value="Medline")
        assert f.value == "Medline"

    def test_list_value_for_like(self):
        f = FilterItem(
            column="VendorName",
            operator=FilterOperator.like,
            value=["Medline", "Cardinal"],
        )
        assert isinstance(f.value, list)
        assert len(f.value) == 2

    def test_list_value_for_in(self):
        f = FilterItem(
            column="Company",
            operator=FilterOperator.in_,
            value=["3000", "2600", "2000"],
        )
        assert isinstance(f.value, list)
        assert len(f.value) == 3

    def test_list_value_for_between(self):
        f = FilterItem(
            column="PostingDate",
            operator=FilterOperator.between,
            value=["2026-01-01", "2026-03-31"],
        )
        assert len(f.value) == 2


# ------------------------------------------------------------------
# ExportIntent with sort_by
# ------------------------------------------------------------------

class TestExportIntentSortBy:
    def test_default_empty_sort(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
        )
        assert intent.sort_by == []

    def test_sort_by_populated(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            sort_by=[SortItem(column="VendorName", direction="ASC")],
        )
        assert len(intent.sort_by) == 1
        assert intent.sort_by[0].column == "VendorName"
