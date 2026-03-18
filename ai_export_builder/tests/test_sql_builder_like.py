"""Tests for SQL generation with LIKE operator."""
from __future__ import annotations

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.sql_builder import build_query

def test_build_query_like_operator():
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName", "ItemDescription"],
        filters=[
            FilterItem(column="VendorName", operator=FilterOperator.like, value="Medline"),
            FilterItem(column="ItemDescription", operator=FilterOperator.like, value="mask")
        ]
    )
    
    # Test with ALL facilities (no RLS)
    sql, params = build_query(intent, user_facilities=["ALL"])
    
    assert "FROM [vw_PO_PURCHASEORDER_LINE_WITH_PCAT]" in sql
    assert "[VendorName] LIKE ?" in sql
    assert "[ItemDescription] LIKE ?" in sql
    
    # assert params contains auto-wrapped wildcards
    assert "%Medline%" in params
    assert "%mask%" in params
    
    # Check simple param count: 2 filters + 1 row limit
    assert len(params) == 3

def test_build_query_like_operator_with_rls():
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName"],
        filters=[
            FilterItem(column="VendorName", operator=FilterOperator.like, value="Stryker")
        ]
    )
    
    # Test with specific facilities
    sql, params = build_query(intent, user_facilities=["FAC1", "FAC2"])
    
    assert "AND [FacilityCode] IN (?, ?)" in sql
    assert "%Stryker%" in params
    assert "FAC1" in params
    assert "FAC2" in params
