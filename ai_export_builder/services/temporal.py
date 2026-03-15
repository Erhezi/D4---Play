"""Resolve natural-language temporal expressions to (start_date, end_date) tuples."""

from __future__ import annotations

import re
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from ai_export_builder.config import settings


def _fiscal_year_start(ref: date, fy_start_month: int) -> date:
    """Return the start of the fiscal year containing *ref*."""
    if ref.month >= fy_start_month:
        return date(ref.year, fy_start_month, 1)
    return date(ref.year - 1, fy_start_month, 1)


def _quarter_of(d: date) -> int:
    """Return calendar quarter (1-4) for a date."""
    return (d.month - 1) // 3 + 1


def resolve(
    expression: str,
    reference_date: date | None = None,
    fy_start_month: int | None = None,
) -> tuple[date, date] | None:
    """Resolve a temporal expression relative to *reference_date*.

    Returns (start_date, end_date) inclusive, or None if unrecognised.
    """
    ref = reference_date or date.today()
    fy_month = fy_start_month or settings.fiscal_year_start_month
    expr = expression.strip().lower()

    # --- YTD ---
    if expr == "ytd":
        return (date(ref.year, 1, 1), ref)

    # --- FYTD (fiscal year to date) ---
    if expr in ("fytd", "fiscal ytd", "fiscal year to date"):
        return (_fiscal_year_start(ref, fy_month), ref)

    # --- This fiscal year ---
    if expr in ("this fiscal year", "current fiscal year"):
        start = _fiscal_year_start(ref, fy_month)
        end = start + relativedelta(years=1) - timedelta(days=1)
        return (start, end)

    # --- Last fiscal year ---
    if expr in ("last fiscal year", "previous fiscal year"):
        this_start = _fiscal_year_start(ref, fy_month)
        start = this_start - relativedelta(years=1)
        end = this_start - timedelta(days=1)
        return (start, end)

    # --- This quarter ---
    if expr in ("this quarter", "current quarter"):
        q = _quarter_of(ref)
        start = date(ref.year, 3 * (q - 1) + 1, 1)
        end = start + relativedelta(months=3) - timedelta(days=1)
        return (start, end)

    # --- Last quarter ---
    if expr in ("last quarter", "previous quarter"):
        q = _quarter_of(ref)
        if q == 1:
            start = date(ref.year - 1, 10, 1)
        else:
            start = date(ref.year, 3 * (q - 2) + 1, 1)
        end = start + relativedelta(months=3) - timedelta(days=1)
        return (start, end)

    # --- Last N months ---
    m = re.match(r"last\s+(\d+)\s+months?", expr)
    if m:
        n = int(m.group(1))
        start = ref - relativedelta(months=n)
        return (start, ref)

    # --- Last N days ---
    m = re.match(r"last\s+(\d+)\s+days?", expr)
    if m:
        n = int(m.group(1))
        return (ref - timedelta(days=n), ref)

    # --- Last N weeks ---
    m = re.match(r"last\s+(\d+)\s+weeks?", expr)
    if m:
        n = int(m.group(1))
        return (ref - timedelta(weeks=n), ref)

    # --- This month ---
    if expr in ("this month", "current month"):
        start = date(ref.year, ref.month, 1)
        end = start + relativedelta(months=1) - timedelta(days=1)
        return (start, end)

    # --- Last month ---
    if expr in ("last month", "previous month"):
        end_of_prev = date(ref.year, ref.month, 1) - timedelta(days=1)
        start = date(end_of_prev.year, end_of_prev.month, 1)
        return (start, end_of_prev)

    # --- This year ---
    if expr in ("this year", "current year"):
        return (date(ref.year, 1, 1), date(ref.year, 12, 31))

    # --- Last year ---
    if expr in ("last year", "previous year"):
        return (date(ref.year - 1, 1, 1), date(ref.year - 1, 12, 31))

    return None
