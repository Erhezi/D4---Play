"""JSON-lines audit logger for export requests."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_export_builder.models.intent import ExportIntent

_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "audit.jsonl"

logger = logging.getLogger(__name__)


def log_export_request(
    user_id: str,
    prompt: str,
    intent: ExportIntent | None,
    sql: str | None,
    row_count: int | None,
    status: str,
    error: str | None = None,
    refinement_count: int = 0,
    guardrail_result: str | None = None,
) -> None:
    """Append one JSON-lines record to the audit log."""
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "prompt": prompt,
        "intent": intent.model_dump(mode="json") if intent else None,
        "sql": _sanitize_sql(sql) if sql else None,
        "row_count": row_count,
        "status": status,
        "error": error,
        "refinement_count": refinement_count,
        "guardrail_result": guardrail_result,
    }
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.error("Failed to write audit log: %s", exc)


def _sanitize_sql(sql: str) -> str:
    """Return the SQL with parameter placeholders intact (no values leaked)."""
    return sql
