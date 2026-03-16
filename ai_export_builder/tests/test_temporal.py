"""Tests for the temporal expression resolver."""
from __future__ import annotations

from datetime import date

import pytest

from ai_export_builder.services.temporal import resolve


class TestYTD:
    def test_ytd_mid_year(self):
        result = resolve("ytd", reference_date=date(2026, 3, 12))
        assert result == (date(2026, 1, 1), date(2026, 3, 12))

    def test_ytd_jan_1(self):
        result = resolve("ytd", reference_date=date(2026, 1, 1))
        assert result == (date(2026, 1, 1), date(2026, 1, 1))

    def test_ytd_case_insensitive(self):
        result = resolve("YTD", reference_date=date(2026, 6, 15))
        assert result is not None
        assert result[0] == date(2026, 1, 1)


class TestFiscalYear:
    def test_fytd_jan_start(self):
        result = resolve("fytd", reference_date=date(2026, 3, 12), fy_start_month=1)
        assert result == (date(2026, 1, 1), date(2026, 3, 12))

    def test_fytd_oct_start_before_fy(self):
        # March 2026, FY starts Oct → FY = Oct 2025 - Sep 2026
        result = resolve("fytd", reference_date=date(2026, 3, 12), fy_start_month=10)
        assert result == (date(2025, 10, 1), date(2026, 3, 12))

    def test_fytd_oct_start_after_fy(self):
        # Nov 2025, FY starts Oct → FY = Oct 2025 - Sep 2026
        result = resolve("fytd", reference_date=date(2025, 11, 15), fy_start_month=10)
        assert result == (date(2025, 10, 1), date(2025, 11, 15))

    def test_this_fiscal_year_jan(self):
        result = resolve("this fiscal year", reference_date=date(2026, 3, 12), fy_start_month=1)
        assert result == (date(2026, 1, 1), date(2026, 12, 31))

    def test_this_fiscal_year_oct(self):
        result = resolve("this fiscal year", reference_date=date(2026, 3, 12), fy_start_month=10)
        assert result == (date(2025, 10, 1), date(2026, 9, 30))

    def test_last_fiscal_year_jan(self):
        result = resolve("last fiscal year", reference_date=date(2026, 3, 12), fy_start_month=1)
        assert result == (date(2025, 1, 1), date(2025, 12, 31))

    def test_last_fiscal_year_oct(self):
        result = resolve("last fiscal year", reference_date=date(2026, 3, 12), fy_start_month=10)
        assert result == (date(2024, 10, 1), date(2025, 9, 30))


class TestQuarter:
    def test_last_quarter_q1(self):
        # In Q1 2026, last quarter = Q4 2025
        result = resolve("last quarter", reference_date=date(2026, 3, 12))
        assert result == (date(2025, 10, 1), date(2025, 12, 31))

    def test_last_quarter_q2(self):
        result = resolve("last quarter", reference_date=date(2026, 5, 15))
        assert result == (date(2026, 1, 1), date(2026, 3, 31))

    def test_last_quarter_q3(self):
        result = resolve("last quarter", reference_date=date(2026, 8, 1))
        assert result == (date(2026, 4, 1), date(2026, 6, 30))

    def test_last_quarter_q4(self):
        result = resolve("last quarter", reference_date=date(2026, 11, 20))
        assert result == (date(2026, 7, 1), date(2026, 9, 30))

    def test_this_quarter_q1(self):
        result = resolve("this quarter", reference_date=date(2026, 2, 14))
        assert result == (date(2026, 1, 1), date(2026, 3, 31))


class TestRelativePeriods:
    def test_last_3_months(self):
        result = resolve("last 3 months", reference_date=date(2026, 3, 15))
        assert result == (date(2025, 12, 15), date(2026, 3, 15))

    def test_last_1_month(self):
        result = resolve("last 1 month", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 2, 15), date(2026, 3, 15))

    def test_last_30_days(self):
        result = resolve("last 30 days", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 2, 13), date(2026, 3, 15))

    def test_last_2_weeks(self):
        result = resolve("last 2 weeks", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 3, 1), date(2026, 3, 15))

    def test_this_month(self):
        result = resolve("this month", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 3, 1), date(2026, 3, 31))

    def test_last_month(self):
        result = resolve("last month", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 2, 1), date(2026, 2, 28))

    def test_this_year(self):
        result = resolve("this year", reference_date=date(2026, 3, 15))
        assert result == (date(2026, 1, 1), date(2026, 12, 31))

    def test_last_year(self):
        result = resolve("last year", reference_date=date(2026, 3, 15))
        assert result == (date(2025, 1, 1), date(2025, 12, 31))


class TestUnknownExpression:
    def test_unknown_returns_none(self):
        assert resolve("next epoch", reference_date=date(2026, 1, 1)) is None

    def test_empty_string_returns_none(self):
        assert resolve("", reference_date=date(2026, 1, 1)) is None
