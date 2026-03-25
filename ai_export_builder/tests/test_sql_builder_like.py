"""Tests for SQL generation with LIKE operator."""
from __future__ import annotations

from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.sql_builder import build_aggregation_query, build_query

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


# ── Multi-value LIKE (within-field OR) ──────────────────────────────────


def test_multi_value_like_generates_or_group():
    """A list-valued LIKE filter generates (col LIKE ? OR col LIKE ?)."""
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName"],
        filters=[
            FilterItem(
                column="VendorName",
                operator=FilterOperator.like,
                value=["Medline", "Cardinal"],
            ),
        ],
    )
    sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
    assert "([VendorName] LIKE ? OR [VendorName] LIKE ?)" in sql
    assert "%Medline%" in params
    assert "%Cardinal%" in params
    assert len(params) == 3  # 2 LIKE values + row limit


def test_multi_value_like_with_other_and_filter():
    """Multi-value LIKE remains AND'd with filters on different columns."""
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName", "POReleaseDate"],
        filters=[
            FilterItem(
                column="VendorName",
                operator=FilterOperator.like,
                value=["Medline", "Cardinal"],
            ),
            FilterItem(
                column="POReleaseDate",
                operator=FilterOperator.between,
                value=["2026-01-01", "2026-06-30"],
            ),
        ],
    )
    sql, params = build_query(intent, user_facilities=["ALL"], max_rows=50)
    # OR group for vendor
    assert "([VendorName] LIKE ? OR [VendorName] LIKE ?)" in sql
    # AND for date
    assert "[POReleaseDate] BETWEEN ? AND ?" in sql
    assert params == ["%Medline%", "%Cardinal%", "2026-01-01", "2026-06-30", 50]


def test_multi_value_like_three_values():
    """Three-value LIKE generates three OR'd conditions."""
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName"],
        filters=[
            FilterItem(
                column="VendorName",
                operator=FilterOperator.like,
                value=["Medline", "Cardinal", "Owens"],
            ),
        ],
    )
    sql, params = build_query(intent, user_facilities=["ALL"], max_rows=100)
    assert sql.count("[VendorName] LIKE ?") == 3
    assert "%Owens%" in params


def test_multi_value_like_aggregation_query():
    """build_aggregation_query also handles multi-value LIKE."""
    intent = ExportIntent(
        selected_view="vw_PO_PURCHASEORDER_LINE_WITH_PCAT",
        columns=["VendorName"],
        filters=[
            FilterItem(
                column="VendorName",
                operator=FilterOperator.like,
                value=["Medline", "Cardinal"],
            ),
        ],
    )
    sql, params = build_aggregation_query(intent, sum_check_columns=[], user_facilities=["ALL"])
    assert "([VendorName] LIKE ? OR [VendorName] LIKE ?)" in sql
    assert "%Medline%" in params
    assert "%Cardinal%" in params
