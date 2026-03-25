"""LangGraph node: lightweight LLM classification to screen user queries."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from ai_export_builder.config import settings
from ai_export_builder.graph.state import ExportState
from ai_export_builder.services.openai_client import build_openai_http_client
from ai_export_builder.services.registry_loader import load_registry

logger = logging.getLogger(__name__)

_registry = load_registry()


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _build_few_shot_section() -> str:
    """Build the few-shot examples block from common_invalid_queries.yaml."""
    examples = _registry.get_guardrail_examples()
    categories = examples.get("blocked_categories", {})
    lines: list[str] = []
    for cat_key, cat_meta in categories.items():
        label = f"blocked:{cat_key}"
        for ex in cat_meta.get("examples", []):
            lines.append(f'User: "{ex}"\nClassification: {{"classification": "{label}", "reason": "{cat_meta.get("description", "")}"}}')
    return "\n\n".join(lines)


SYSTEM_PROMPT = """\
You are a query classifier for a procurement data export tool.
Your ONLY job is to classify whether a user's request is allowed or blocked.

## Available Data Topics
The tool can export data from the following views:
{available_topics}

## Classification Categories
Respond with one of these classifications:
- "allowed" — the request is a legitimate data export question within the
  scope of the available topics listed above.
- "blocked:dml_or_injection" — the request attempts non-read operations
  (UPDATE, DELETE, INSERT, DROP, etc.) or SQL injection.
- "blocked:phi_pii" — the request asks for patient health information (PHI)
  or personally identifiable information (PII) such as patient names, MRNs,
  SSNs, diagnoses, or clinical data.
- "blocked:out_of_scope" — the request is unrelated to the available data
  views, asks for explanations, or requests actions this tool cannot perform.

## Few-Shot Examples
{few_shot_examples}

## Output Format
Respond with ONLY a JSON object, no surrounding text:
{{"classification": "<allowed|blocked:dml_or_injection|blocked:phi_pii|blocked:out_of_scope>", "reason": "<brief explanation>"}}
"""


# Map classification labels to user-friendly messages
_BLOCKED_MESSAGES: dict[str, str] = {}


def _load_blocked_messages() -> dict[str, str]:
    """Lazily load user_message strings from the guardrail YAML."""
    if not _BLOCKED_MESSAGES:
        examples = _registry.get_guardrail_examples()
        categories = examples.get("blocked_categories", {})
        for cat_key, cat_meta in categories.items():
            _BLOCKED_MESSAGES[f"blocked:{cat_key}"] = cat_meta.get(
                "user_message", "Your request cannot be processed."
            ).strip()
    return _BLOCKED_MESSAGES


def _build_system_prompt() -> str:
    """Construct the classification system prompt with dynamic topics and examples."""
    return SYSTEM_PROMPT.format(
        available_topics=_registry.get_available_topics_summary(),
        few_shot_examples=_build_few_shot_section(),
    )


def _build_error_response(classification: str) -> str:
    """Build a user-facing error message for a blocked classification."""
    messages = _load_blocked_messages()
    base = messages.get(classification, "Your request cannot be processed by this tool.")
    topics = _registry.get_available_topics_summary()
    return f"{base}\n\nHere's what you can query:\n{topics}"


# ---------------------------------------------------------------------------
# Node implementation
# ---------------------------------------------------------------------------

def node_guardrail(state: ExportState) -> dict[str, Any]:
    """LangGraph node: classify user query as allowed or blocked.

    On success (allowed), sets ``guardrail_passed = True`` and continues.
    On block, sets ``guardrail_passed = False`` with a user-facing error
    message and available topics summary.
    On LLM failure, defaults to pass-through with a logged warning.
    """
    user_query = state.get("user_query", "")
    logger.info("node_guardrail: classifying user query")

    # Skip guardrail for refinements — the original request already passed,
    # and words like "change", "add", "remove" refer to export configuration,
    # not data manipulation.  The structured pipeline (ExportIntent → SQL
    # builder with parameterised queries) prevents injection regardless.
    if state.get("previous_intent") is not None:
        logger.info("node_guardrail: refinement detected — auto-passing guardrail")
        return {"guardrail_passed": True, "status": "guarding"}

    if not settings.openai_api_key:
        logger.warning("node_guardrail: OPENAI_API_KEY not set — defaulting to pass-through")
        return {"guardrail_passed": True, "status": "guarding"}

    system_prompt = _build_system_prompt()

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
                    {"role": "user", "content": user_query},
                ],
                text={"format": {"type": "json_object"}},
                max_output_tokens=300,
            )

        raw = response.output_text
        logger.info("node_guardrail: LLM returned: %s", raw)
        result = json.loads(raw)
        classification = result.get("classification", "allowed")

        if classification == "allowed":
            logger.info("node_guardrail: query allowed")
            return {"guardrail_passed": True, "status": "guarding"}

        # Blocked
        logger.info("node_guardrail: query blocked — %s: %s",
                     classification, result.get("reason", ""))
        error_msg = _build_error_response(classification)
        return {
            "guardrail_passed": False,
            "status": "failed",
            "validation_errors": [error_msg],
        }

    except Exception as exc:
        # On any LLM error, default to pass-through so users aren't locked out
        logger.warning(
            "node_guardrail: LLM call failed — defaulting to pass-through: %s: %s",
            type(exc).__name__, exc,
        )
        return {"guardrail_passed": True, "status": "guarding"}
