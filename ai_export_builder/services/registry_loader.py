"""Load registry.yaml and provide alias resolution and view/column lookups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "registry" / "registry.yaml"


class Registry:
    """In-memory representation of the metadata registry."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._views: dict[str, Any] = data.get("views", {})
        # Build lowercased alias → (view_id, column_name) index
        self._alias_index: dict[str, tuple[str, str]] = {}
        for view_id, view_meta in self._views.items():
            for col_name, col_meta in view_meta.get("columns", {}).items():
                for alias in col_meta.get("aliases", []):
                    self._alias_index[alias.lower()] = (view_id, col_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_alias(self, alias: str) -> tuple[str, str] | None:
        """Return (view_id, column_name) for a natural-language alias, or None."""
        return self._alias_index.get(alias.strip().lower())

    def get_view_candidates(self, terms: list[str]) -> list[str]:
        """Return view IDs that contain columns matching *any* of the given terms."""
        hits: set[str] = set()
        for term in terms:
            result = self.resolve_alias(term)
            if result:
                hits.add(result[0])
        return sorted(hits)

    def get_all_columns(self, view_id: str) -> list[str]:
        """Return all column names for a given view, or empty list if unknown."""
        view = self._views.get(view_id)
        if not view:
            return []
        return list(view.get("columns", {}).keys())

    def get_column_meta(self, view_id: str, column: str) -> dict[str, Any] | None:
        """Return metadata dict for a specific column, or None."""
        view = self._views.get(view_id)
        if not view:
            return None
        return view.get("columns", {}).get(column)

    def get_view_meta(self, view_id: str) -> dict[str, Any] | None:
        """Return full view metadata dict, or None."""
        return self._views.get(view_id)

    def view_exists(self, view_id: str) -> bool:
        return view_id in self._views

    def all_view_ids(self) -> list[str]:
        return list(self._views.keys())

    def get_registry_schema_for_prompt(self) -> str:
        """Return a compact text representation suitable for LLM system prompts."""
        lines: list[str] = []
        for view_id, view_meta in self._views.items():
            lines.append(f"View: {view_id}")
            lines.append(f"  Description: {view_meta.get('description', '')}")
            lines.append("  Columns:")
            for col_name, col_meta in view_meta.get("columns", {}).items():
                aliases = ", ".join(col_meta.get("aliases", []))
                lines.append(
                    f"    - {col_name} ({col_meta.get('type', 'string')}): "
                    f"{col_meta.get('description', '')}  aliases: [{aliases}]"
                )
            lines.append("")
        return "\n".join(lines)


def load_registry(path: Path | None = None) -> Registry:
    """Load the YAML registry from disk and return a Registry instance."""
    p = path or _REGISTRY_PATH
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Registry(data)
