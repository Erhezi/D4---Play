"""Pydantic models for parsed export intent."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_serializer


# Human-readable labels for filter operators (shown in UI)
OPERATOR_LABELS: dict[str, str] = {
    "eq": "=",
    "neq": "≠",
    "gt": ">",
    "gte": "≥",
    "lt": "<",
    "lte": "≤",
    "like": "contains",
    "in": "is one of",
    "between": "between",
}


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

    @property
    def label(self) -> str:
        """Return a user-friendly display label for this operator."""
        return OPERATOR_LABELS.get(self.value, self.value)


class SortItem(BaseModel):
    """A single ORDER BY clause entry."""
    column: str = Field(..., description="Column name to sort by")
    direction: Literal["ASC", "DESC"] = Field("ASC", description="Sort direction")


class FilterItem(BaseModel):
    column: str = Field(..., description="Column name from the registry")
    operator: FilterOperator = Field(..., description="Comparison operator")
    value: str | list[str] = Field(
        ...,
        description=(
            "Filter value(s). For 'between': [start, end]. For 'in': list of values. "
            "For 'like' with multiple values: list of search terms (within-field OR)."
        ),
    )

    @field_serializer("operator")
    @classmethod
    def _serialize_operator(cls, v: FilterOperator) -> str:
        return v.value if isinstance(v, FilterOperator) else str(v)


class ExportIntent(BaseModel):
    selected_view: str = Field(..., description="Registry view identifier")
    columns: list[str] = Field(..., description="Columns to include in the export")
    filters: list[FilterItem] = Field(default_factory=list, description="WHERE clause filters")
    sort_by: list[SortItem] = Field(default_factory=list, description="ORDER BY clauses")
    warnings: list[str] = Field(default_factory=list, description="LLM-generated warnings or caveats")
