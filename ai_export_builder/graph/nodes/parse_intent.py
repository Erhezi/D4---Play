"""LangGraph node: translate user natural language into an ExportIntent via OpenAI."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState
from ai_export_builder.models.intent import ExportIntent, FilterItem, FilterOperator
from ai_export_builder.services.openai_client import build_openai_http_client
from ai_export_builder.services.registry_loader import load_registry
from ai_export_builder.services.temporal import resolve as resolve_temporal

logger = logging.getLogger(__name__)

_registry = load_registry()

SYSTEM_PROMPT = """\
You are a precise data-export assistant for a hospital procurement system.
Your ONLY job is to translate the user's natural-language request into a
structured JSON object (an "Export Intent") that selects data from ONE
registered view.

## Rules
1. Pick exactly ONE view from the registry below.
2. Column selection:
   - Do NOT include columns from "basic" field groups in your `columns` list.
     Basic columns are ALWAYS auto-included in every export by the system.
   - Only list columns from "enrichment" field groups or other columns that the
     user explicitly asks for.
   - If the user mentions a group by name (e.g. "include Premier data"), list
     all columns from that enrichment group.
   - Default to an empty `columns` list if the user does not request anything
     beyond what the basic group already provides.
3. Build filters using only columns that exist in the view and operators:
   eq, neq, gt, gte, lt, lte, like, in, between.
   - For 'like' filters, provide ONLY the search term without '%' wildcards.
     The system will automatically wrap the value with '%' for matching.
     Example: use value "mask" (NOT "%mask%").
   - For text columns that have a companion ID field (see "companion:" in the
     schema), prefer using 'like' so the system can show the
     user a preview of matching entities for confirmation.
   - When the user mentions multiple values for the same text column
     (e.g. "Medline or Cardinal", "vendors: Medline, Cardinal, Owens"),
     create a SINGLE FilterItem with operator "like" and value as a JSON
     list: ["Medline", "Cardinal"]. Do NOT create separate FilterItem
     entries for the same column — the system generates within-field OR
     logic automatically.
   - Different fields must be separate FilterItem entries (AND logic).
   - Any column used in a filter is automatically added to the output.
4. For date filters, resolve relative expressions (e.g. "last quarter",
   "YTD") using the temporal context provided.
5. NEVER produce JOINs, GROUP BY, or aggregations.
6. If the user's request is ambiguous or you cannot map something, add a
   note to `warnings` — do NOT invent columns.
7. If the user mentions sorting or ordering (e.g. "sort by vendor name",
   "order by amount descending", "highest spend first"), populate `sort_by`
   with one or more entries. Each entry has a `column` and optional
   `direction` ("ASC" or "DESC", default "ASC"). Only use columns that
   exist in the view.
8. Respond with ONLY the JSON object, no surrounding text.

## Temporal Context
- Current date: {current_date}
- Fiscal year start month: {fy_start_month}

## Registry Schema
{registry_schema}

## Required JSON Schema
{{
  "selected_view": "<view_id>",
  "columns": ["<col1>", "<col2>", ...],
  "filters": [
    {{"column": "<col>", "operator": "<op>", "value": "<val or [val1,val2]>"}}
  ],
  "sort_by": [
    {{"column": "<col>", "direction": "ASC|DESC"}}
  ],
  "warnings": ["<optional note>"]
}}
"""


def _build_system_prompt(state: ExportState) -> str:
    """Construct the system prompt with temporal context and registry schema."""
    tc = state.get("temporal_context", {})
    current_date = tc.get("current_date", "")
    fy_start_month = tc.get("fiscal_year_start_month", settings.fiscal_year_start_month)

    return SYSTEM_PROMPT.format(
        current_date=current_date,
        fy_start_month=fy_start_month,
        registry_schema=_registry.get_registry_schema_for_prompt(),
    )


def _undisambiguate_for_prompt(intent: ExportIntent, state: ExportState) -> ExportIntent:
    """Reverse disambiguation so the LLM sees text-column LIKE filters.

    After disambiguation, filters like ``VendorName LIKE 'TIDI'`` become
    ``VendorID eq '1001815'``.  When building a refinement prompt we need
    the LLM to see the original text-column form so it can properly merge
    new search terms (e.g. "add Stryker") and let disambiguation run again.
    """
    disambiguation_results = state.get("disambiguation_results", [])
    if not disambiguation_results:
        return intent

    # id_col → original text filter info
    id_to_text: dict[str, dict] = {}
    for r in disambiguation_results:
        id_to_text[r["companion"]] = {
            "column": r["column"],
            "operator": r.get("original_operator", "like"),
            "value": r["original_value"],
        }

    new_filters: list[FilterItem] = []
    for f in intent.filters:
        if f.column in id_to_text:
            orig = id_to_text[f.column]
            new_filters.append(FilterItem(
                column=orig["column"],
                operator=FilterOperator(orig["operator"]),
                value=orig["value"],
            ))
        else:
            new_filters.append(f)

    return intent.model_copy(update={"filters": new_filters})


def _build_user_message(state: ExportState) -> str:
    """Build the user message, including validation feedback and prior intent context."""
    msg = state["user_query"]

    # Delta-parsing: if a previous intent exists, include it as context
    previous_intent = state.get("previous_intent")
    if previous_intent is not None:
        # Reverse disambiguation so the LLM sees text-column LIKE filters
        prompt_intent = _undisambiguate_for_prompt(previous_intent, state)
        prev_json = prompt_intent.model_dump_json(indent=2)
        original_query = state.get("original_user_query", "")
        msg = (
            f"[ORIGINAL USER REQUEST]\n{original_query}\n\n"
            "[CURRENT EXPORT STATE — merge the user's new request with this existing intent. "
            "Add new filters, remove dropped ones, modify changed values, keep unchanged parts. "
            "If the user asks to change the view or start over, produce a fresh intent.]\n"
            f"{prev_json}\n\n"
            "[DISAMBIGUATION NOTE: Filters on text columns with companion ID fields "
            "(e.g. VendorName) are shown in their original text form. When the user "
            "asks to add more values for the same entity, ADD them to the existing "
            "text-column filter value list using LIKE. Do NOT use the companion ID "
            "column — the system handles ID resolution automatically.]\n\n"
            f"[USER'S REFINEMENT REQUEST]\n{msg}"
        )

    errors = state.get("validation_errors", [])
    if errors:
        feedback = "\n".join(f"- {e}" for e in errors)
        msg += (
            "\n\n[SYSTEM FEEDBACK — your previous attempt had validation errors. "
            "Fix them and try again.]\n" + feedback
        )
    return msg


def _resolve_temporal_filters(intent: ExportIntent, state: ExportState) -> ExportIntent:
    """Post-process: resolve any temporal string values in date filters."""
    from datetime import date as Date

    tc = state.get("temporal_context", {})
    current_date_str = tc.get("current_date", "")
    fy_start = tc.get("fiscal_year_start_month", settings.fiscal_year_start_month)
    ref_date = Date.fromisoformat(current_date_str) if current_date_str else None

    for f in intent.filters:
        # Only try resolution on string values that look like temporal expressions
        if isinstance(f.value, str) and not _looks_like_date(f.value):
            result = resolve_temporal(f.value, reference_date=ref_date, fy_start_month=fy_start)
            if result:
                start_d, end_d = result
                f.operator = FilterOperator.between
                f.value = [start_d.isoformat(), end_d.isoformat()]
    return intent


def _looks_like_date(val: str) -> bool:
    """Quick check: does the string look like an ISO date (YYYY-MM-DD)?"""
    import re
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", val.strip()))


def _format_llm_error(exc: Exception) -> str:
    """Return a user-facing error string with SSL guidance when applicable."""
    message = str(exc)
    lower_message = message.lower()

    if "certificate verify failed" in lower_message or "certificate_verify_failed" in lower_message:
        return (
            "LLM parsing error: TLS certificate validation failed while connecting to OpenAI. "
            "This environment may require the Windows/system certificate store or a corporate CA bundle. "
            "If the error persists, set OPENAI_CA_BUNDLE to your organization's PEM certificate file."
        )

    if isinstance(exc, FileNotFoundError) and "OPENAI_CA_BUNDLE" in message:
        return f"LLM configuration error: {message}"

    return f"LLM parsing error ({type(exc).__name__}): {exc}"


def node_parse_intent(state: ExportState) -> dict[str, Any]:
    """LangGraph node: call OpenAI to parse user query into ExportIntent."""
    logger.info("node_parse_intent: parsing user query (retry %d)", state.get("retry_count", 0))

    system_prompt = _build_system_prompt(state)
    user_message = _build_user_message(state)

    if not settings.openai_api_key:
        logger.error("node_parse_intent: OPENAI_API_KEY is not set")
        return {
            "intent": None,
            "validation_errors": ["OPENAI_API_KEY is not configured. "
                                  "Set it as an environment variable or in .env."],
            "status": "failed",
        }

    try:
        with build_openai_http_client() as http_client:
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                http_client=http_client,
            )
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                text={"format": {"type": "json_object"}},
            )

        raw = response.output_text
        print(f"\n{'='*60}\nLLM returned JSON:\n{raw}\n{'='*60}\n")
        logger.info("LLM returned JSON: %s", raw)
        data = json.loads(raw)
        intent = ExportIntent(**data)

        # Post-process temporal expressions in filters
        intent = _resolve_temporal_filters(intent, state)

        logger.info("node_parse_intent: parsed view=%s, %d cols, %d filters",
                     intent.selected_view, len(intent.columns), len(intent.filters))
        return {
            "intent": intent,
            "previous_intent": intent,
            "validation_errors": [],
            "status": "parsing",
        }
    except Exception as exc:
        import traceback
        logger.error(
            "node_parse_intent: LLM call failed — %s: %s\n%s",
            type(exc).__name__, exc, traceback.format_exc(),
        )
        return {
            "intent": None,
            "validation_errors": [_format_llm_error(exc)],
            "status": "failed",
        }
