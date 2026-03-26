"""LangGraph node: answer meta-queries about tool capabilities and field definitions."""

from __future__ import annotations

import logging
import re
from typing import Any

from ai_export_builder.graph.state import ExportState
from ai_export_builder.services.registry_loader import load_registry

logger = logging.getLogger(__name__)

_registry = load_registry()


def _build_capabilities_response() -> str:
    """Build a response listing available views and sample questions."""
    lines: list[str] = [
        "Here's what I can help you with:\n",
    ]
    for view_id in _registry.all_view_ids():
        meta = _registry.get_view_meta(view_id)
        if not meta:
            continue
        display = meta.get("display_name", view_id)
        topic = meta.get("primary_topic", meta.get("description", ""))
        lines.append(f"**{display}**")
        lines.append(f"  {topic}")
        samples = meta.get("samples_of_valid_queries", [])
        if samples:
            for s in samples[:3]:
                lines.append(f'  - *"{s}"*')
        lines.append("")

    lines.append("You can also ask:")
    lines.append('- *"What does [field name] mean?"* — to learn about a specific column')
    lines.append('- *"What columns are in the PO view?"* — to see available fields')
    lines.append("\nJust describe the data you need in plain English and I'll build the export for you!")
    return "\n".join(lines)


def _find_field_info(query: str) -> str | None:
    """Try to find field/column info from the user's query."""
    # Try patterns like "what does X mean", "what is X", "explain X"
    patterns = [
        r"what\s+does\s+(.+?)\s+mean",
        r"what\s+is\s+(.+?)[\?\.]?\s*$",
        r"explain\s+(?:the\s+)?(.+?)[\?\.]?\s*$",
        r"describe\s+(?:the\s+)?(.+?)[\?\.]?\s*$",
        r"tell\s+me\s+about\s+(.+?)[\?\.]?\s*$",
    ]
    field_name = None
    for pattern in patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            field_name = m.group(1).strip().strip('"\'`')
            break

    if not field_name:
        return None

    # Check if the user is asking about columns in a specific view
    view_pattern = r"columns?\s+(?:in|of|for)\s+(?:the\s+)?(.+?)[\?\.]?\s*$"
    view_match = re.search(view_pattern, query, re.IGNORECASE)
    if view_match:
        return _build_view_columns_response(view_match.group(1).strip())

    # Search for the field across all views
    return _search_field(field_name)


def _build_view_columns_response(view_hint: str) -> str:
    """List columns for a view matching the hint."""
    hint_lower = view_hint.lower()
    matched_view = None
    for vid in _registry.all_view_ids():
        meta = _registry.get_view_meta(vid)
        display = (meta or {}).get("display_name", vid).lower()
        if hint_lower in vid.lower() or hint_lower in display:
            matched_view = vid
            break
        # Also match partial keywords like "po", "invoice", "savings"
        if hint_lower in vid.lower().replace("vw_", "").replace("_", " "):
            matched_view = vid
            break

    if not matched_view:
        return f"I couldn't find a view matching \"{view_hint}\". Available views: {', '.join(_registry.all_view_ids())}"

    meta = _registry.get_view_meta(matched_view)
    display = (meta or {}).get("display_name", matched_view)
    columns = _registry.get_all_columns(matched_view)
    lines = [f"**{display}** (`{matched_view}`) has the following columns:\n"]
    for col in columns:
        col_meta = _registry.get_column_meta(matched_view, col) or {}
        label = col_meta.get("label", col)
        desc = col_meta.get("description", "")
        if desc:
            lines.append(f"- **{label}** (`{col}`): {desc}")
        else:
            lines.append(f"- **{label}** (`{col}`)")
    return "\n".join(lines)


def _search_field(field_name: str) -> str | None:
    """Search for a field/column across all views and return its description."""
    field_lower = field_name.lower()
    matches: list[tuple[str, str, dict]] = []  # (view_id, col_name, col_meta)

    for vid in _registry.all_view_ids():
        # Direct column name match
        col_meta = _registry.get_column_meta(vid, field_name)
        if col_meta:
            matches.append((vid, field_name, col_meta))
            continue
        # Case-insensitive search
        for col in _registry.get_all_columns(vid):
            if col.lower() == field_lower:
                matches.append((vid, col, _registry.get_column_meta(vid, col) or {}))
                break
        # Alias search
        result = _registry.resolve_alias(field_name)
        if result:
            vid_r, col_r = result
            meta_r = _registry.get_column_meta(vid_r, col_r)
            if meta_r and (vid_r, col_r, meta_r) not in matches:
                matches.append((vid_r, col_r, meta_r))

    if not matches:
        return (
            f"I couldn't find a field called \"{field_name}\" in any of the available views. "
            f"Try asking *\"what columns are in the PO view?\"* to see available fields, "
            f"or *\"what data is available?\"* for an overview."
        )

    lines: list[str] = []
    for vid, col, meta in matches:
        view_meta = _registry.get_view_meta(vid)
        view_display = (view_meta or {}).get("display_name", vid)
        label = meta.get("label", col)
        desc = meta.get("description", "No description available.")
        col_type = meta.get("type", "string")
        aliases = meta.get("aliases", [])
        companion = _registry.get_companion_column(vid, col)

        lines.append(f"**{label}** (`{col}`) in *{view_display}*")
        lines.append(f"- Type: {col_type}")
        if desc:
            lines.append(f"- Description: {desc}")
        if aliases:
            lines.append(f"- Also known as: {', '.join(aliases)}")
        if companion:
            comp_meta = _registry.get_column_meta(vid, companion)
            comp_label = (comp_meta or {}).get("label", companion)
            lines.append(f"- Related field: {comp_label} (`{companion}`)")
        lines.append("")

    return "\n".join(lines)


def node_meta_responder(state: ExportState) -> dict[str, Any]:
    """Answer meta-queries about capabilities or field definitions."""
    classification = state.get("guardrail_classification", "")
    user_query = state.get("user_query", "")
    logger.info("node_meta_responder: handling %s", classification)

    if classification == "meta:capabilities":
        response = _build_capabilities_response()
    elif classification == "meta:field_info":
        response = _find_field_info(user_query)
        if not response:
            # Fallback to capabilities if we can't parse the field name
            response = (
                "I'm not sure which field you're asking about. "
                "Try asking something like *\"what does VendorName mean?\"* "
                "or *\"what columns are in the PO view?\"*\n\n"
                + _build_capabilities_response()
            )
    else:
        response = _build_capabilities_response()

    return {
        "meta_response": response,
        "status": "meta_response",
    }
