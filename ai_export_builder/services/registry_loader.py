"""Load registry files and provide alias resolution, view/column lookups, and per-view connection routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai_export_builder.config import settings


_REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
_VIEWS_INDEX_PATH = _REGISTRY_DIR / "registry_views.yaml"
_VIEWS_DIR = _REGISTRY_DIR / "views"
_CONNECTIONS_PATH = _REGISTRY_DIR / "connection.yaml"
_GUARDRAIL_PATH = _REGISTRY_DIR / "common_invalid_queries.yaml"


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
        views_index: dict[str, Any] | None = None,
        guardrail_examples: dict[str, Any] | None = None,
    ) -> None:
        self._views = views
        self._views_index = views_index or {}
        self._guardrail_examples = guardrail_examples or {}
        # connections keyed by database key, e.g. "PRIME", "SCS"
        self._connections: dict[str, Any] = connections.get("databases", {})
        # Build lowercased alias → (view_id, column_name) index
        self._alias_index: dict[str, tuple[str, str]] = {}
        # Build bidirectional companion index: (view_id, col) → companion_col
        # e.g. ("vw_PO_...", "VendorName") → "Vendor" and vice-versa
        self._companion_index: dict[tuple[str, str], str] = {}
        # Build concept index: (view_id, concept_id) → [col_names]
        self._concept_index: dict[tuple[str, str], list[str]] = {}
        # Build sum_check index: view_id → [col_names]
        self._sum_check_index: dict[str, list[str]] = {}

        for view_id, view_meta in self._views.items():
            sum_check_cols: list[str] = []
            # First pass: collect concept roles and build alias/sum_check indices
            concept_roles: dict[str, dict[str, list[str]]] = {}  # concept_id → {role → [col_names]}
            for col_name, col_meta in view_meta.get("columns", {}).items():
                for alias in col_meta.get("aliases", []):
                    self._alias_index[alias.lower()] = (view_id, col_name)
                # Build concept index
                concept_id = col_meta.get("concept_id")
                if concept_id:
                    key = (view_id, concept_id)
                    if key not in self._concept_index:
                        self._concept_index[key] = []
                    self._concept_index[key].append(col_name)
                    # Track roles for companion pair building
                    role = col_meta.get("concept_role", "")
                    if role in ("id", "display"):
                        if concept_id not in concept_roles:
                            concept_roles[concept_id] = {}
                        if role not in concept_roles[concept_id]:
                            concept_roles[concept_id][role] = []
                        concept_roles[concept_id][role].append(col_name)
                # Legacy companion pairs from required_for_field_mapping
                mapping_target = col_meta.get("required_for_field_mapping")
                if mapping_target:
                    self._companion_index[(view_id, col_name)] = mapping_target
                    self._companion_index[(view_id, mapping_target)] = col_name
                # Build sum_check index
                if col_meta.get("sum_check"):
                    sum_check_cols.append(col_name)
            # Second pass: build companion pairs from concept_id roles (id ↔ display)
            for _concept_id, roles in concept_roles.items():
                id_cols = roles.get("id", [])
                display_cols = roles.get("display", [])
                if len(id_cols) == 1 and len(display_cols) == 1:
                    self._companion_index[(view_id, id_cols[0])] = display_cols[0]
                    self._companion_index[(view_id, display_cols[0])] = id_cols[0]
            if sum_check_cols:
                self._sum_check_index[view_id] = sum_check_cols

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
        """Return columns from 'core' (or legacy 'basic') field groups — always included in output."""
        result = self.get_field_group_columns(view_id, "core")
        if not result:
            result = self.get_field_group_columns(view_id, "basic")
        return result

    def get_companion_column(self, view_id: str, column: str) -> str | None:
        """Return the companion column for a text↔ID pair, or None."""
        return self._companion_index.get((view_id, column))

    def get_disambiguable_columns(self, view_id: str) -> dict[str, str]:
        """Return {text_column: id_column} for columns that have a companion pair.

        A column is disambiguable if it has a companion (id ↔ display) via
        concept_role grouping or legacy required_for_field_mapping.
        Returns the display (text) column as key, id column as value.
        """
        view = self._views.get(view_id)
        if not view:
            return {}
        result: dict[str, str] = {}
        for col_name, col_meta in view.get("columns", {}).items():
            # Legacy path: required_for_field_mapping
            mapping_target = col_meta.get("required_for_field_mapping")
            if mapping_target:
                result[mapping_target] = col_name
            # Concept-role path: display columns with an id companion
            concept_role = col_meta.get("concept_role")
            if concept_role == "display":
                companion = self._companion_index.get((view_id, col_name))
                if companion:
                    # Verify companion is the 'id' role
                    comp_meta = self.get_column_meta(view_id, companion)
                    if comp_meta and comp_meta.get("concept_role") == "id":
                        result[col_name] = companion
        return result

    # ------------------------------------------------------------------
    # Concept groups & sum_check
    # ------------------------------------------------------------------

    def get_concept_group(self, view_id: str, concept_id: str) -> list[str]:
        """Return all column names sharing a concept_id within a view."""
        return list(self._concept_index.get((view_id, concept_id), []))

    def get_column_concept_id(self, view_id: str, column: str) -> str | None:
        """Return the concept_id for a specific column, or None."""
        meta = self.get_column_meta(view_id, column)
        return meta.get("concept_id") if meta else None

    def get_sum_check_columns(self, view_id: str) -> list[str]:
        """Return column names marked sum_check: true for a view."""
        return list(self._sum_check_index.get(view_id, []))

    # ------------------------------------------------------------------
    # Guardrail & topic summary
    # ------------------------------------------------------------------

    def get_guardrail_examples(self) -> dict[str, Any]:
        """Return the guardrail examples loaded from common_invalid_queries.yaml."""
        return self._guardrail_examples

    def get_available_topics_summary(self) -> str:
        """Build a user-friendly summary of what topics/views can be queried.

        Uses the views_index (registry_views.yaml) for topic + description.
        """
        lines: list[str] = []
        for view_id, meta in self._views_index.items():
            display = meta.get("display_name", view_id)
            topic = meta.get("primary_topic", "")
            lines.append(f"- {display}: {topic}")
        return "\n".join(lines) if lines else "No views available."

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
            samples = view_meta.get("sample_questions") or view_meta.get("samples_of_valid_queries", [])
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
                concept_id = col_meta.get("concept_id")
                if concept_id:
                    parts.append(f"concept_id: {concept_id}")
                concept_role = col_meta.get("concept_role")
                if concept_role:
                    parts.append(f"role: {concept_role}")
                aliases = col_meta.get("aliases", [])
                if aliases:
                    parts.append(f"aliases: [{', '.join(aliases)}]")
                companion = self.get_companion_column(view_id, col_name)
                if companion:
                    parts.append(f"companion: {companion}")
                if col_meta.get("sum_check"):
                    parts.append("sum_check: true")
                lines.append(f"    - {' | '.join(parts)}")
            lines.append("")
        return "\n".join(lines)


def load_registry(
    views_index_path: Path | None = None,
    views_dir: Path | None = None,
    connections_path: Path | None = None,
    guardrail_path: Path | None = None,
) -> Registry:
    """Load registry YAML files and return a Registry instance.

    View metadata is assembled by merging:
    1. ``registry_views.yaml`` — header info (display_name, primary_topic, description, samples)
    2. Per-view YAML files in ``views/`` — column-level details, field_groups, keys, enums

    The per-view files are the canonical source for column definitions. Header
    metadata from registry_views.yaml is overlaid onto each view dict for
    convenience (display_name, description, etc. accessible on the same object).
    """
    vip = views_index_path or _VIEWS_INDEX_PATH
    vd = views_dir or _VIEWS_DIR
    cp = connections_path or _CONNECTIONS_PATH
    gp = guardrail_path or _GUARDRAIL_PATH

    # --- Load views index (header metadata) ---
    views_index: dict[str, Any] = {}
    if vip.exists():
        with open(vip, encoding="utf-8") as f:
            raw_index = yaml.safe_load(f) or {}
        # Handle both flat list and {views: [...]} formats
        raw_list = raw_index.get("views", raw_index) if isinstance(raw_index, dict) else raw_index
        if isinstance(raw_list, list):
            for entry in raw_list:
                vid = entry.get("view_id")
                if vid:
                    views_index[vid] = entry

    # --- Load per-view YAML files (column-level detail) ---
    raw_views: list[dict[str, Any]] = []
    if vd.is_dir():
        for yaml_file in sorted(vd.glob("*.yaml")):
            with open(yaml_file, encoding="utf-8") as f:
                view_data = yaml.safe_load(f)
            if isinstance(view_data, dict) and view_data.get("view_id"):
                # Overlay header metadata from views_index onto the view dict
                vid = view_data["view_id"]
                if vid in views_index:
                    header = views_index[vid]
                    for key in ("display_name", "primary_topic", "description",
                                "granularity", "time_coverage",
                                "samples_of_valid_queries", "samples_of_invalid_queries"):
                        if key in header and key not in view_data:
                            view_data[key] = header[key]
                raw_views.append(view_data)

    # --- Load connections ---
    connections: dict[str, Any] = {}
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            connections = yaml.safe_load(f) or {}

    # --- Load guardrail examples ---
    guardrail_examples: dict[str, Any] = {}
    if gp.exists():
        with open(gp, encoding="utf-8") as f:
            guardrail_examples = yaml.safe_load(f) or {}

    return Registry(
        _normalize_views(raw_views),
        connections,
        views_index=views_index,
        guardrail_examples=guardrail_examples,
    )
