"""LangGraph state schema for the Export Builder workflow."""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd

from ai_export_builder.models.intent import ExportIntent


class UserProfile(TypedDict):
    user_id: str
    user_name: str
    facilities: list[str]


class TemporalContext(TypedDict):
    current_date: str  # ISO format
    fiscal_year_start_month: int


class ExportState(TypedDict, total=False):
    """Full state tracked across the LangGraph workflow."""

    # Input
    user_query: str

    # Orchestration
    intent: ExportIntent | None
    validation_errors: list[str]
    status: str  # parsing | pending_approval | disambiguating | executing | completed | failed
    retry_count: int  # max 2 retries before surfacing errors

    # Disambiguation
    disambiguation_needed: bool
    disambiguation_results: list[dict[str, Any]]  # [{column, companion, matches: [{text, id}]}]

    # Context injected at the start of each run
    temporal_context: TemporalContext
    user_profile: UserProfile

    # Output
    result_df: Any  # pd.DataFrame (Any for serialisation compatibility)
    result_row_count: int
    error_message: str
