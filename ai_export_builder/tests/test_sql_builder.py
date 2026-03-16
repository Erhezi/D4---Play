"""Tests for the SQL builder — parameterization, RLS injection, template output."""
from __future__ import annotations

import pytest

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.sql_builder import build_disambiguation_query, build_query


class TestBasicQuery:
    def test_simple_select_all_rls(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName", "CalculateExtendedAmount"],
            filters=[],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        assert "SELECT" in sql
        assert "[VendorName]" in sql
        assert "[CalculateExtendedAmount]" in sql
        assert "[vw_PO_PURCHASEORDER_LINE_WITH_PCAT]" in sql
        # RLS ALL means no facility restriction
        assert "FacilityCode" not in sql
        # Only the row-limit param
        assert params == [100]

    def test_rls_facility_filter(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[],
        )
        sql, params = build_query(intent, user_facilities=["FAC_A", "FAC_B"], max_rows=50)
        assert "[FacilityCode] IN (?, ?)" in sql
        assert params == ["FAC_A", "FAC_B", 50]


class TestParameterizedFilters:
    def test_eq_filter(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.eq, value="Medline"),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        assert "[VendorName] = ?" in sql
        assert "Medline" in params
        # No raw string interpolation of the value
        assert "Medline" not in sql

    def test_between_filter(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["POReleaseDate"],
            filters=[
                FilterItem(
                    column="POReleaseDate",
                    operator=FilterOperator.between,
                    value=["2026-01-01", "2026-03-15"],
                ),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        assert "[POReleaseDate] BETWEEN ? AND ?" in sql
        assert "2026-01-01" in params
        assert "2026-03-15" in params
        # Values should NOT appear literally in SQL
        assert "2026-01-01" not in sql
        assert "2026-03-15" not in sql

    def test_in_filter(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(
                    column="VendorName",
                    operator=FilterOperator.in_,
                    value=["Medline", "Stryker"],
                ),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        assert "[VendorName] IN (?, ?)" in sql
        assert "Medline" in params
        assert "Stryker" in params

    def test_like_filter(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.like, value="%glove%"),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        assert "[VendorName] LIKE ?" in sql
        assert "%glove%" in params

    def test_no_string_interpolation(self):
        """Ensure filter values are NEVER embedded directly in SQL."""
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.eq, value="Robert'); DROP TABLE Students;--"),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
        # The malicious string must be in params, not in SQL
        assert "Robert'); DROP TABLE Students;--" not in sql
        assert "Robert'); DROP TABLE Students;--" in params


class TestRowLimit:
    def test_row_limit_always_last_param(self):
        intent = ExportIntent(
            selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            columns=["VendorName"],
            filters=[
                FilterItem(column="VendorName", operator=FilterOperator.eq, value="Test"),
            ],
        )
        sql, params = build_query(intent, user_facilities=["ALL"], max_rows=500)
        assert params[-1] == 500
        assert "FETCH NEXT ? ROWS ONLY" in sql


class TestDisambiguationQuery:
    def test_basic_disambiguation(self):
        sql, params = build_disambiguation_query(
            view_id="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            text_col="VendorName",
            id_col="Vendor",
            like_value="%medline%",
            user_facilities=["ALL"],
        )
        assert "SELECT DISTINCT TOP 50" in sql
        assert "[VendorName]" in sql
        assert "[Vendor]" in sql
        assert "LIKE ?" in sql
        assert params == ["%medline%"]
        # Value not in SQL
        assert "%medline%" not in sql

    def test_disambiguation_with_rls(self):
        sql, params = build_disambiguation_query(
            view_id="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            text_col="VendorName",
            id_col="Vendor",
            like_value="%stryker%",
            user_facilities=["FAC_A", "FAC_B"],
        )
        assert "[FacilityCode] IN (?, ?)" in sql
        assert params == ["%stryker%", "FAC_A", "FAC_B"]

    def test_disambiguation_custom_max_rows(self):
        sql, params = build_disambiguation_query(
            view_id="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            text_col="CompanyText",
            id_col="Company",
            like_value="%monte%",
            user_facilities=["ALL"],
            max_rows=10,
        )
        assert "TOP 10" in sql

    def test_disambiguation_no_injection(self):
        """Ensure the LIKE value is parameterized, not interpolated."""
        sql, params = build_disambiguation_query(
            view_id="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
            text_col="VendorName",
            id_col="Vendor",
            like_value="'; DROP TABLE Students;--",
            user_facilities=["ALL"],
        )
        assert "'; DROP TABLE Students;--" not in sql
        assert "'; DROP TABLE Students;--" in params
