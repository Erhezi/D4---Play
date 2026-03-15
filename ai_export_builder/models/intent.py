"""Pydantic models for parsed export intent."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class FilterOperator(str, Enum):
    eq = "eq"
    neq = "neq"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    like = "like"
    in_ = "in"
    between = "between"


class FilterItem(BaseModel):
    column: str = Field(..., description="Column name from the registry")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: str | list[str] = Field(
        ...,
        description="Filter value(s). For 'between': [start, end]. For 'in': list of values.",
    )


class ExportIntent(BaseModel):
    selected_view: str = Field(..., description="Registry view identifier")
    columns: list[str] = Field(..., description="Columns to include in the export")
    filters: list[FilterItem] = Field(default_factory=list, description="WHERE clause filters")
    warnings: list[str] = Field(default_factory=list, description="LLM-generated warnings or caveats")
