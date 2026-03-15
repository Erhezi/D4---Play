"""Load registry files and provide alias resolution, view/column lookups, and per-view connection routing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
_VIEWS_PATH = _REGISTRY_DIR / "registry_views.yaml"
_CONNECTIONS_PATH = _REGISTRY_DIR / "connection.yaml"


def _normalize_views(raw_views: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert the views list (keyed by view_id) into a dict and normalize
    each view's columns from a list into a dict keyed by column name."""
    views: dict[str, Any] = {}
    for view in raw_views:
        view_id = view.get("view_id")
        if not view_id:
            continue
        # Normalize columns list → dict keyed by column name
        raw_cols = view.get("columns") or []
        cols: dict[str, Any] = {}
        for col in raw_cols:
            col_name = col.get("name")
            if col_name:
                cols[col_name] = {k: v for k, v in col.items() if k != "name"}
        view = {**view, "columns": cols}
        views[view_id] = view
    return views


class Registry:
    """In-memory representation of the metadata registry."""

    def __init__(
        self,
        views: dict[str, Any],
        connections: dict[str, Any],
    ) -> None:
        self._views = views
        # connections keyed by database key, e.g. "PRIME", "SCS"
        self._connections: dict[str, Any] = connections.get("databases", {})
        # Build lowercased alias → (view_id, column_name) index
        self._alias_index: dict[str, tuple[str, str]] = {}
        for view_id, view_meta in self._views.items():
            for col_name, col_meta in view_meta.get("columns", {}).items():
                for alias in col_meta.get("aliases", []):
                    self._alias_index[alias.lower()] = (view_id, col_name)

    # ------------------------------------------------------------------
    # Connection routing
    # ------------------------------------------------------------------

    def get_database_key(self, view_id: str) -> str | None:
        """Return the database key (e.g. 'PRIME', 'SCS') for a given view."""
        view = self._views.get(view_id)
        return view.get("database") if view else None

    def get_connection_config(self, view_id: str) -> dict[str, Any] | None:
        """Return the connection config dict from connection.yaml for a given view."""
        db_key = self.get_database_key(view_id)
        if not db_key:
            return None
        return self._connections.get(db_key)

    def get_connection_string(self, view_id: str) -> str:
        """Resolve the connection string for a view from its env variable.

        Raises KeyError if the view or its connection is not found.
        Raises EnvironmentError if the env variable is not set.
        """
        config = self.get_connection_config(view_id)
        if not config:
            raise KeyError(f"No connection config found for view '{view_id}'")
        env_var = config["connection_string_env"]
        conn_str = os.environ.get(env_var)
        if not conn_str:
            raise EnvironmentError(
                f"Environment variable '{env_var}' is not set "
                f"(required for view '{view_id}', database '{self.get_database_key(view_id)}')"
            )
        return conn_str

    # ------------------------------------------------------------------
    # View / column lookups
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
            db_key = view_meta.get("database", "")
            lines.append(f"View: {view_id}  [database: {db_key}]")
            lines.append(f"  Display Name: {view_meta.get('display_name', '')}")
            lines.append(f"  Description: {view_meta.get('description', '')}")
            # Sample questions help the LLM understand typical user requests
            samples = view_meta.get("sample_questions", [])
            if samples:
                lines.append("  Sample Questions:")
                for sq in samples:
                    lines.append(f"    - {sq}")
            lines.append("  Columns:")
            for col_name, col_meta in view_meta.get("columns", {}).items():
                parts: list[str] = [f"{col_name} ({col_meta.get('type', 'string')})"]
                label = col_meta.get("label")
                if label:
                    parts.append(f'label: "{label}"')
                concept = col_meta.get("concept")
                if concept:
                    parts.append(f"concept: {concept}")
                aliases = col_meta.get("aliases", [])
                if aliases:
                    parts.append(f"aliases: [{', '.join(aliases)}]")
                lines.append(f"    - {' | '.join(parts)}")
            lines.append("")
        return "\n".join(lines)


def load_registry(
    views_path: Path | None = None,
    connections_path: Path | None = None,
) -> Registry:
    """Load registry_views.yaml and connection.yaml, return a Registry instance."""
    vp = views_path or _VIEWS_PATH
    cp = connections_path or _CONNECTIONS_PATH

    with open(vp, encoding="utf-8") as f:
        raw_views: list[dict[str, Any]] = yaml.safe_load(f) or []

    with open(cp, encoding="utf-8") as f:
        connections: dict[str, Any] = yaml.safe_load(f) or {}

    return Registry(_normalize_views(raw_views), connections)
