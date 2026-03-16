"""Tests for SQL generation with LIKE operator.

This standalone test file ensures pytest finds and runs the checks without
relying on importing tests as a package. It adjusts `sys.path` so imports
can resolve when running from the repository root.
"""
import os
import sys

# Ensure repository root is on sys.path so `ai_export_builder` imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.sql_builder import build_query


def test_build_query_like_operator():
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName", "ItemDescription"],
        filters=[
            FilterItem(column="VendorName", operator=FilterOperator.like, value="%Medline%"),
            FilterItem(column="ItemDescription", operator=FilterOperator.like, value="%mask%"),
        ],
    )

    sql, params = build_query(intent, user_facilities=["ALL"])

    assert "FROM [vw_PO_PURCHASEORDER_LINE_WITH_PCAT]" in sql
    assert "[VendorName] LIKE ?" in sql
    assert "[ItemDescription] LIKE ?" in sql
    assert "%Medline%" in params
    assert "%mask%" in params
    assert len(params) == 3


def test_build_query_like_operator_with_rls():
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName"],
        filters=[
            FilterItem(column="VendorName", operator=FilterOperator.like, value="Stryker%"),
        ],
    )

    sql, params = build_query(intent, user_facilities=["FAC1", "FAC2"])

    assert "AND [FacilityCode] IN (?, ?)" in sql
    assert "Stryker%" in params
    assert "FAC1" in params
    assert "FAC2" in params
