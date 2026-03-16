"""Tests for the registry loader — alias resolution, view lookups, connection routing."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_export_builder.services.registry_loader import load_registry


@pytest.fixture()
def registry():
    """Load the real registry YAML files shipped with the project."""
    return load_registry()


# ------------------------------------------------------------------
# Alias resolution
# ------------------------------------------------------------------

class TestResolveAlias:
    def test_known_alias_supplier(self, registry):
        result = registry.resolve_alias("supplier")
        assert result is not None
        view_id, col_name = result
        assert view_id == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert col_name == "VendorName"

    def test_alias_case_insensitive(self, registry):
        result = registry.resolve_alias("SUPPLIER")
        assert result is not None
        assert result == registry.resolve_alias("supplier")

    def test_unknown_alias_returns_none(self, registry):
        assert registry.resolve_alias("nonexistent_alias_xyz") is None

    def test_alias_with_whitespace(self, registry):
        result = registry.resolve_alias("  supplier  ")
        assert result is not None

    def test_spend_alias(self, registry):
        result = registry.resolve_alias("spend")
        assert result is not None
        assert result[1] == "CalculateExtendedAmount"


# ------------------------------------------------------------------
# View lookups
# ------------------------------------------------------------------

class TestViewLookups:
    def test_view_exists(self, registry):
        assert registry.view_exists("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")

    def test_view_not_exists(self, registry):
        assert not registry.view_exists("fake_view")

    def test_all_view_ids(self, registry):
        ids = registry.all_view_ids()
        assert "vw_PO_PURCHASEORDER_LINE_WITH_PCAT" in ids
        assert "vw_invoice_detail" in ids

    def test_get_all_columns(self, registry):
        cols = registry.get_all_columns("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert "VendorName" in cols
        assert "CalculateExtendedAmount" in cols
        assert "POReleaseDate" in cols

    def test_get_all_columns_unknown_view(self, registry):
        assert registry.get_all_columns("nonexistent") == []

    def test_get_view_candidates(self, registry):
        candidates = registry.get_view_candidates(["supplier"])
        assert "vw_PO_PURCHASEORDER_LINE_WITH_PCAT" in candidates

    def test_get_column_meta(self, registry):
        meta = registry.get_column_meta("vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "VendorName")
        assert meta is not None
        assert "vendor" in meta.get("concept", "").lower()

    def test_get_view_meta(self, registry):
        meta = registry.get_view_meta("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert meta is not None
        assert meta["database"] == "PRIME"

    def test_registry_schema_for_prompt(self, registry):
        schema = registry.get_registry_schema_for_prompt()
        assert "vw_PO_PURCHASEORDER_LINE_WITH_PCAT" in schema
        assert "VendorName" in schema


# ------------------------------------------------------------------
# Connection routing
# ------------------------------------------------------------------

class TestConnectionRouting:
    def test_get_database_key(self, registry):
        assert registry.get_database_key("vw_PO_PURCHASEORDER_LINE_WITH_PCAT") == "PRIME"
        assert registry.get_database_key("vw_invoice_detail") == "SCS"

    def test_get_connection_config(self, registry):
        config = registry.get_connection_config("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert config is not None
        assert config["connection_string_env"] == "PRIME_DB_URL"

    def test_get_connection_string_reads_env(self, registry):
        with patch.dict(os.environ, {"PRIME_DB_URL": "Driver=test;Server=localhost"}):
            conn_str = registry.get_connection_string("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
            assert conn_str == "Driver=test;Server=localhost"

    def test_get_connection_string_missing_env_raises(self, registry):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="PRIME_DB_URL"):
                registry.get_connection_string("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")

    def test_get_connection_string_unknown_view_raises(self, registry):
        with pytest.raises(KeyError, match="No connection config"):
            registry.get_connection_string("nonexistent_view")


# ------------------------------------------------------------------
# Field groups & companion columns
# ------------------------------------------------------------------

class TestFieldGroups:
    def test_get_basic_columns(self, registry):
        basic = registry.get_basic_columns("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert len(basic) > 0
        assert "PO" in basic
        assert "VendorName" in basic
        assert "CalculateExtendedAmount" in basic

    def test_get_enrichment_columns(self, registry):
        enrichment = registry.get_field_group_columns(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "enrichment"
        )
        assert "ManufacturerName_Premier" in enrichment
        assert "ContractCategory_Premier" in enrichment

    def test_get_field_group_columns_unknown_view(self, registry):
        assert registry.get_field_group_columns("nonexistent", "basic") == []

    def test_get_field_group_columns_unknown_type(self, registry):
        assert registry.get_field_group_columns(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "nonexistent_type"
        ) == []

    def test_basic_and_enrichment_no_overlap(self, registry):
        basic = set(registry.get_basic_columns("vw_PO_PURCHASEORDER_LINE_WITH_PCAT"))
        enrichment = set(registry.get_field_group_columns(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "enrichment"
        ))
        assert basic.isdisjoint(enrichment)


class TestCompanionColumns:
    def test_vendor_companion(self, registry):
        # VendorName (text) ↔ Vendor (ID)
        assert registry.get_companion_column(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "VendorName"
        ) == "Vendor"
        assert registry.get_companion_column(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "Vendor"
        ) == "VendorName"

    def test_company_companion(self, registry):
        assert registry.get_companion_column(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "CompanyText"
        ) == "Company"
        assert registry.get_companion_column(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "Company"
        ) == "CompanyText"

    def test_no_companion(self, registry):
        assert registry.get_companion_column(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "ItemDescription"
        ) is None

    def test_companion_unknown_view(self, registry):
        assert registry.get_companion_column("nonexistent", "VendorName") is None

    def test_get_disambiguable_columns(self, registry):
        disambiguable = registry.get_disambiguable_columns(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        )
        assert "VendorName" in disambiguable
        assert disambiguable["VendorName"] == "Vendor"
        assert "CompanyText" in disambiguable
        assert disambiguable["CompanyText"] == "Company"

    def test_get_disambiguable_columns_unknown_view(self, registry):
        assert registry.get_disambiguable_columns("nonexistent") == {}


class TestRegistrySchemaForPrompt:
    def test_schema_includes_field_groups(self, registry):
        schema = registry.get_registry_schema_for_prompt()
        assert "Field Groups:" in schema
        assert "PO Line Details" in schema
        assert "type: basic" in schema

    def test_schema_includes_companion_info(self, registry):
        schema = registry.get_registry_schema_for_prompt()
        assert "companion:" in schema
