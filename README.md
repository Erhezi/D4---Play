# AI-Assisted Export Builder

A **Streamlit + LangGraph** semantic export tool that translates natural language into constrained single-table SQL queries against SQL Server views. Uses OpenAI for intent parsing and safety screening, a YAML metadata registry, Jinja2 SQL templating, and parameterized queries throughout.

---

## What's New in v2.0

- **Guardrail node** — LLM screens every query for SQL injection, PHI exposure, and out-of-scope requests before any parsing begins. Blocked queries receive a clear explanation.
- **Stateful refinement loop** — after reviewing your export, click **Refine** to go back to chat and adjust filters, dates, or columns. The system carries forward your existing intent as context (delta-parsing). Up to 3 rounds before a graceful reset.
- **20-row data preview** — the verification card shows a live sample of matching rows and a row-count + spend summary before you commit to the full export.
- **Multi-value LIKE filtering** — say "Medline or Cardinal" and the system generates within-field OR logic automatically: `([VendorName] LIKE ? OR [VendorName] LIKE ?)`.
- **Sorting** — ask to "sort by vendor name descending" and it appears in the ORDER BY clause.
- **Concept-group column expansion** — selecting a display column (e.g. `VendorName`) automatically pulls in its ID sibling (`Vendor`) so lookups always have both.
- **Human-readable operator labels** — filters show `contains`, `between`, `is one of`, `≥` instead of raw enum codes throughout all UI surfaces.

---

## Architecture

```
User prompt
    │
    ▼
┌───────────┐
│ guardrail │  ← LLM safety screen (injection / PHI / out-of-scope)
└─────┬─────┘
      │ passed          blocked → END (with explanation)
      ▼
┌──────────────┐
│ orchestrator │  ← track refinement_count; route or reset
└──────┬───────┘
       │ round ≤ 3             round > 3 → reset_signal → END
       ▼
┌──────────────┐     retry (up to 2x)    ┌───────────────────┐
│ parse_intent │◀────────────────────────│ validate_intent   │
│  (OpenAI)    │────────────────────────▶│ (deterministic +  │
└──────────────┘                          │  concept-group    │
                                          │  column resolve)  │
                                          └────────┬──────────┘
                                                   │ valid
                                                   ▼
                                          ┌─────────────────┐      ┌──────────────────────┐
                                          │  disambiguate   │─────▶│ disambiguation_review │
                                          │ (SELECT DISTINCT│      │ (HITL — pick entities)│
                                          │  per LIKE value)│      └──────────┬────────────┘
                                          └───────┬─────────┘                │ confirmed
                                                  │ (no ambiguity)           │
                                                  ▼                          ▼
                                          ┌─────────────────┐
                                          │ hydrate_preview │  ← 20-row preview + row count + spend
                                          └───────┬─────────┘
                                                  ▼
                                          ┌─────────────────┐
                                          │  human_review   │  ← HITL: Confirm / Edit / Refine
                                          └───────┬─────────┘
                                 Confirm          │           Refine → back to orchestrator
                                                  ▼
                                          ┌─────────────────┐
                                          │ execute_export  │  ← SQL → DB → DataFrame
                                          └───────┬─────────┘
                                                  ▼
                                             CSV download
```

**Core constraints:**
- Single-table queries only — no JOINs, GROUP BY, or aggregations
- All SQL values passed as `?` parameters to pyodbc — never string-interpolated
- RLS (row-level security) facility filter auto-appended to every query
- Parse retry loop capped at 2 attempts before surfacing errors

**Column resolution (automatic):**
- **Core columns** (from `group_type: "core"` field groups) are always included in every export
- **Enrichment columns** are only included when explicitly requested
- Any column used in a filter is automatically added to the output
- All columns sharing a `concept_id` (e.g. `VendorName` + `Vendor`) are expanded together

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- ODBC Driver 17 for SQL Server
- Access to at least one SQL Server with the registered views

### 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3. Configure

```bash
cp ai_export_builder/.env.example ai_export_builder/.env
```

Edit `ai_export_builder/.env` with your actual values:
- `OPENAI_API_KEY` — your OpenAI API key
- `OPENAI_CA_BUNDLE` — optional PEM file for a corporate proxy/private CA if TLS interception is in place
- `OPENAI_USE_SYSTEM_CERT_STORE` — defaults to `true`; uses the Windows/OS trust store for OpenAI calls
- `PRIME_DB_URL` / `SCS_DB_URL` — pyodbc connection strings for each database
- Adjust `FISCAL_YEAR_START_MONTH`, `MAX_EXPORT_ROWS`, `MAX_REFINEMENT_ROUNDS`, etc. as needed

### 4. Run

```bash
streamlit run ai_export_builder/app.py
```

Open http://localhost:8501 in your browser.

---

## User Guide

### Asking for an export

Type a plain-English request in the chat box. Be as specific or as vague as you like — the system will ask for clarification if something is ambiguous.

**Example prompts:**
```
Export all Medline glove spend for last quarter
Show me purchase orders over $10,000 from Q1 2026, sorted by amount descending
AP invoices for vendors Medline or Cardinal, YTD, include GL index
```

**Tips for better results:**
| Want to… | Say… |
|---|---|
| Filter by a date range | "last quarter", "YTD", "between Jan 1 and Mar 15" |
| Filter by multiple vendors | "Medline or Cardinal" (generates OR logic for you) |
| Sort results | "sort by vendor name", "highest spend first" |
| Include extra data | "include Premier data", "with GL index" |
| See only basics | Just describe the filter — core columns are always included |

### Entity disambiguation

When you filter by vendor name or another text field that has a paired ID column, the system shows a **Confirm Matching Entities** card before the export runs. This lets you pick the exact records you want when multiple entities match your search term.

- Check or uncheck each matching entity
- Click **Confirm Selection** to proceed with those specific IDs
- Click **Keep Partial Match** to use the original LIKE filter as-is

### Reviewing your export (Verification Card)

After parsing (and disambiguation if needed), the **Verification Card** appears with:

1. **View & column list** — toggle any enrichment columns on/off
2. **Filters** — each filter is editable: change the column, operator, or value inline
3. **Data Preview** — a sample of the first 20 matching rows
4. **Summary metrics** — total row count and aggregate spend (where available)

You have three options:

| Button | What it does |
|---|---|
| ✅ Confirm Export | Runs the full query and makes the CSV available for download |
| ✏️ Edit & Resubmit | Saves your inline edits and re-runs validation |
| 🔄 Refine | Returns to chat so you can describe further changes in natural language |

### Refining your export

Click **🔄 Refine** on the verification card, then type what you want to change:
```
Actually filter by posting date instead, use Q4 2025
Add the ItemDescription column
Remove the company filter
```

The system carries your existing intent forward — you only need to describe what's different. The sidebar shows your current refinement round (e.g. "Round: 2/3"). After 3 rounds the session resets automatically.

### Downloading results

After a successful export, a **Download CSV** button appears in the left sidebar. The row count is shown in the chat. A preview of the first 100 rows is also displayed inline.

### SQL transparency

Click **🔍 Show SQL** in the sidebar at any time to inspect the exact parameterized SQL that will be (or was) sent to the database. Parameters are listed separately — no values are substituted into the SQL string.

### Safety & limits

- Queries that look like SQL injection, request PHI/PII, or ask for data outside the registered views are blocked automatically. You'll receive a clear explanation.
- Each user is limited to **10 requests per day** by default (configurable).
- Export row count is capped at **1,000,000 rows** by default (configurable).

---

## Project Structure

```
ai_export_builder/
├── app.py                      # Streamlit entry point
├── config.py                   # pydantic-settings — reads .env
├── .env.example                # Template for local secrets
├── registry/
│   ├── connection.yaml         # Named DB connections → env-var references
│   ├── registry_views.yaml     # Header index: view IDs, metadata summary
│   ├── registry_common.yaml    # Guardrail few-shot examples
│   └── views/
│       └── <view_id>.yaml      # Per-view: columns, aliases, concept_id, field groups
├── graph/
│   ├── state.py                # ExportState TypedDict for LangGraph
│   ├── workflow.py             # Graph definition + HITL interrupt points
│   └── nodes/
│       ├── guardrail.py        # LLM safety screen (injection / PHI / scope)
│       ├── orchestrator.py     # Refinement loop counter + routing
│       ├── reset_signal.py     # Friendly reset message node
│       ├── parse_intent.py     # LLM node: NL → ExportIntent (with delta-parsing)
│       ├── validate_intent.py  # Deterministic validation + concept-group resolution
│       ├── disambiguate.py     # SELECT DISTINCT preview (multi-value LIKE aware)
│       ├── hydrate_preview.py  # 20-row preview + aggregation summary
│       └── execute_export.py   # SQL gen → DB query → DataFrame
├── models/
│   └── intent.py               # ExportIntent, FilterItem, SortItem, FilterOperator + labels
├── services/
│   ├── registry_loader.py      # YAML registry: concept index, sum-check, alias, companion
│   ├── sql_builder.py          # Jinja2 SQL rendering, RLS, sort, multi-value LIKE, aggregation
│   ├── db.py                   # pyodbc connections with per-view routing
│   ├── temporal.py             # Resolve "YTD", "last quarter" → ISO date ranges
│   ├── rate_limiter.py         # In-memory daily request counter
│   └── audit.py                # JSON-lines audit logger (w/ refinement_count, guardrail_result)
├── templates/
│   └── select_query.sql.j2     # Parameterized SQL template (WHERE, ORDER BY, FETCH NEXT)
├── ui/
│   ├── chat.py                 # Streamlit chat components
│   ├── verification_card.py    # Preview + editable filters + Confirm/Edit/Refine buttons
│   └── disambiguation_card.py  # HITL entity selection card (multi-value aware, operator labels)
├── services/logs/
│   └── audit.jsonl             # Append-only audit log (gitignored)
└── tests/
    ├── test_registry.py        # Registry loading, alias resolution, concept groups, routing
    ├── test_temporal.py        # Temporal expression resolution
    ├── test_sql_builder.py     # Parameterized SQL, RLS, ORDER BY, aggregation, injection safety
    ├── test_sql_builder_like.py# LIKE operator + multi-value LIKE OR groups
    ├── test_validate_intent.py # Validation + concept-group expansion + sort validation
    ├── test_disambiguate.py    # Disambiguation node: single + multi-value, dedup
    ├── test_parse_intent.py    # Mock OpenAI parse + delta-parsing
    ├── test_guardrail.py       # Guardrail classification cases
    ├── test_orchestrator.py    # Refinement loop routing
    ├── test_reset_signal.py    # Reset node
    ├── test_hydrate_preview.py # Preview + aggregation node
    ├── test_models.py          # Pydantic model validation
    └── test_integration.py     # End-to-end workflow (mock LLM + DB)
```

---

## Running Tests

```bash
# All unit tests (skips live-DB connectivity tests)
pytest ai_export_builder/tests/ -v --ignore=ai_export_builder/tests/test_connectivity.py

# Full suite including connectivity (requires live DB)
pytest ai_export_builder/tests/ -v
```

196 tests, all passing (excluding live-DB connectivity tests).

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for intent parsing and guardrail classification |
| `OPENAI_CA_BUNDLE` | blank | Optional PEM bundle for corporate TLS inspection / private CAs |
| `OPENAI_USE_SYSTEM_CERT_STORE` | `true` | Use the OS certificate store for OpenAI HTTPS requests |
| `PRIME_DB_URL` | (required) | pyodbc connection string for PRIME database |
| `SCS_DB_URL` | (required) | pyodbc connection string for SCS/PBI database |
| `DAILY_REQUEST_LIMIT` | `10` | Max requests per user per day |
| `MAX_EXPORT_ROWS` | `1000000` | Row limit per export query |
| `MAX_REFINEMENT_ROUNDS` | `3` | Max refinement rounds before session reset |
| `FISCAL_YEAR_START_MONTH` | `1` | January = 1, October = 10 |
| `TEST_USER_FACILITIES` | `["ALL"]` | JSON list of facility codes for RLS |

---

## TLS / Corporate Proxy Notes

If OpenAI requests fail with `CERTIFICATE_VERIFY_FAILED`, the app uses the OS trust store by default for HTTPS validation. On locked-down corporate networks, export your organization's root/intermediate certificate as a PEM file and point `OPENAI_CA_BUNDLE` at it.
