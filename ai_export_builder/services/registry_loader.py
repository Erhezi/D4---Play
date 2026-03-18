"""Load registry files and provide alias resolution, view/column lookups, and per-view connection routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_export_builder.config import settings


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
        # Build bidirectional companion index: (view_id, col) → companion_col
        # e.g. ("vw_PO_...", "VendorName") → "Vendor" and vice-versa
        self._companion_index: dict[tuple[str, str], str] = {}
        for view_id, view_meta in self._views.items():
            for col_name, col_meta in view_meta.get("columns", {}).items():
                for alias in col_meta.get("aliases", []):
                    self._alias_index[alias.lower()] = (view_id, col_name)
                # Build companion pairs from required_for_field_mapping
                mapping_target = col_meta.get("required_for_field_mapping")
                if mapping_target:
                    # col_name (ID) is required when mapping_target (text) is selected
                    self._companion_index[(view_id, col_name)] = mapping_target
                    self._companion_index[(view_id, mapping_target)] = col_name

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
        Raises EnvironmentError if the env variable is not set in either the
        process environment or the loaded .env-backed application settings.
        """
        config = self.get_connection_config(view_id)
        if not config:
            raise KeyError(f"No connection config found for view '{view_id}'")
        env_var = config["connection_string_env"]
        conn_str = settings.get_named_connection_string(env_var)
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

    # ------------------------------------------------------------------
    # Field groups & companion columns
    # ------------------------------------------------------------------

    def get_field_group_columns(self, view_id: str, group_type: str) -> list[str]:
        """Return column names belonging to field groups of the given type."""
        view = self._views.get(view_id)
        if not view:
            return []
        result: list[str] = []
        for group in view.get("field_groups", []):
            if group.get("group_type") == group_type:
                result.extend(group.get("columns_included", []))
        return result

    def get_basic_columns(self, view_id: str) -> list[str]:
        """Return columns from all 'basic' field groups (always included in output)."""
        return self.get_field_group_columns(view_id, "basic")

    def get_companion_column(self, view_id: str, column: str) -> str | None:
        """Return the companion column for a text↔ID pair, or None."""
        return self._companion_index.get((view_id, column))

    def get_disambiguable_columns(self, view_id: str) -> dict[str, str]:
        """Return {text_column: id_column} for columns that have a companion pair.

        Only returns entries where the column has ``required_for_field_mapping``
        (i.e. the ID column pointing to the text column).
        """
        view = self._views.get(view_id)
        if not view:
            return {}
        result: dict[str, str] = {}
        for col_name, col_meta in view.get("columns", {}).items():
            mapping_target = col_meta.get("required_for_field_mapping")
            if mapping_target:
                # col_name is the ID, mapping_target is the text column
                result[mapping_target] = col_name
        return result

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
            # Field groups
            field_groups = view_meta.get("field_groups", [])
            if field_groups:
                lines.append("  Field Groups:")
                for fg in field_groups:
                    gname = fg.get("group_name", "")
                    gtype = fg.get("group_type", "")
                    gdesc = fg.get("description", "")
                    gcols = fg.get("columns_included", [])
                    lines.append(f"    - {gname} (type: {gtype}): {gdesc}")
                    lines.append(f"      Columns: [{', '.join(gcols)}]")
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
                companion = self.get_companion_column(view_id, col_name)
                if companion:
                    parts.append(f"companion: {companion}")
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
