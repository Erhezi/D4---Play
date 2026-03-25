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
    def test_known_alias_spend(self, registry):
        result = registry.resolve_alias("spend")
        assert result is not None
        view_id, col_name = result
        assert view_id == "vw_PO_PURCHASEORDER_LINE_WITH_PCAT"
        assert col_name == "CalculateExtendedAmount"

    def test_alias_case_insensitive(self, registry):
        result = registry.resolve_alias("SPEND")
        assert result is not None
        assert result == registry.resolve_alias("spend")

    def test_unknown_alias_returns_none(self, registry):
        assert registry.resolve_alias("nonexistent_alias_xyz") is None

    def test_alias_with_whitespace(self, registry):
        result = registry.resolve_alias("  spend  ")
        assert result is not None

    def test_po_number_alias(self, registry):
        result = registry.resolve_alias("po number")
        assert result is not None
        assert result[1] == "PO"


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
        assert "vw_MainSavings" in ids

    def test_get_all_columns(self, registry):
        cols = registry.get_all_columns("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert "VendorName" in cols
        assert "CalculateExtendedAmount" in cols
        assert "POReleaseDate" in cols

    def test_get_all_columns_unknown_view(self, registry):
        assert registry.get_all_columns("nonexistent") == []

    def test_get_view_candidates(self, registry):
        candidates = registry.get_view_candidates(["spend"])
        assert "vw_PO_PURCHASEORDER_LINE_WITH_PCAT" in candidates

    def test_get_column_meta(self, registry):
        meta = registry.get_column_meta("vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "VendorName")
        assert meta is not None
        assert meta.get("concept_id") == "vendor"

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
        assert registry.get_database_key("vw_MainSavings") == "SCS"

    def test_get_connection_config(self, registry):
        config = registry.get_connection_config("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert config is not None
        assert config["connection_string_env"] == "PRIME_DB_URL"

    def test_get_connection_string_reads_env(self, registry):
        with patch.dict(os.environ, {"PRIME_DB_URL": "Driver=test;Server=localhost"}):
            conn_str = registry.get_connection_string("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
            assert conn_str == "Driver=test;Server=localhost"

    def test_get_connection_string_reads_settings_fallback(self, registry, monkeypatch):
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setattr(
                "ai_export_builder.config.settings.prime_db_url",
                "Driver=test;Server=from-settings",
            )
            conn_str = registry.get_connection_string("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
            assert conn_str == "Driver=test;Server=from-settings"

    def test_get_connection_string_missing_env_raises(self, registry):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                __import__("ai_export_builder.config", fromlist=["settings"]).settings,
                "prime_db_url",
                "",
            ):
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
        assert "Core PO Line Item Fields" in schema
        assert "type: core" in schema

    def test_schema_includes_concept_info(self, registry):
        schema = registry.get_registry_schema_for_prompt()
        assert "concept_id:" in schema
        assert "companion:" in schema


# ------------------------------------------------------------------
# Concept groups & sum_check
# ------------------------------------------------------------------

class TestConceptGroups:
    def test_vendor_concept_group(self, registry):
        group = registry.get_concept_group(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "vendor"
        )
        assert "VendorName" in group
        assert "Vendor" in group

    def test_concept_group_unknown_view(self, registry):
        assert registry.get_concept_group("nonexistent", "vendor") == []

    def test_concept_group_unknown_concept(self, registry):
        assert registry.get_concept_group(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "nonexistent_concept"
        ) == []

    def test_get_column_concept_id(self, registry):
        assert registry.get_column_concept_id(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "VendorName"
        ) == "vendor"

    def test_get_column_concept_id_unknown(self, registry):
        assert registry.get_column_concept_id(
            "vw_PO_PURCHASEORDER_LINE_WITH_PCAT", "nonexistent"
        ) is None


class TestSumCheck:
    def test_po_sum_check(self, registry):
        cols = registry.get_sum_check_columns("vw_PO_PURCHASEORDER_LINE_WITH_PCAT")
        assert "CalculateExtendedAmount" in cols

    def test_ap_sum_check(self, registry):
        cols = registry.get_sum_check_columns("vw_AP_PAYABLESINVOICE_WITH_VENDOR_GL_INDEX")
        assert "InvoiceGLAmount" in cols

    def test_savings_sum_check(self, registry):
        cols = registry.get_sum_check_columns("vw_MainSavings")
        assert "Total Savings" in cols

    def test_unknown_view_sum_check(self, registry):
        assert registry.get_sum_check_columns("nonexistent") == []


# ------------------------------------------------------------------
# Guardrail & topic summary
# ------------------------------------------------------------------

class TestGuardrailAndTopics:
    def test_get_guardrail_examples_loaded(self, registry):
        examples = registry.get_guardrail_examples()
        assert "blocked_categories" in examples
        assert "dml_or_injection" in examples["blocked_categories"]

    def test_get_available_topics_summary(self, registry):
        summary = registry.get_available_topics_summary()
        assert "Purchase Order" in summary or "purchase order" in summary.lower()
        assert "Invoice" in summary or "invoice" in summary.lower()
