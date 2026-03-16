# AI-Assisted Export Builder

A **Streamlit + LangGraph** semantic export tool that translates natural language into constrained single-table SQL queries against SQL Server views. Uses OpenAI for intent parsing, a YAML metadata registry, Jinja2 SQL templating, and parameterized queries throughout.

## Architecture

```
User prompt
    │
    ▼
┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
│ parse_intent │────▶│ validate_intent    │────▶│ human_review │
│  (OpenAI)    │◀────│  (deterministic)   │     │  (HITL pause)│
│              │retry│                    │     │              │
└──────────────┘     └───────────────────┘     └──────┬───────┘
                                                      │ confirm
                                                      ▼
                                               ┌──────────────┐
                                               │execute_export │
                                               │ (SQL → DB)   │
                                               └──────────────┘
                                                      │
                                                      ▼
                                                  CSV download
```

**Key constraints:**
- Single-table queries only — no JOINs, GROUP BY, or aggregation
- All SQL values passed as `?` parameters to pyodbc — never string-interpolated
- RLS (row-level security) facility filter auto-appended to every query
- Retry loop capped at 2 attempts before surfacing errors to the user

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
- `PRIME_DB_URL` / `SCS_DB_URL` — pyodbc connection strings for each database
- Adjust `FISCAL_YEAR_START_MONTH`, `MAX_EXPORT_ROWS`, etc. as needed

### 4. Run

```bash
streamlit run ai_export_builder/app.py
```

Open http://localhost:8501 in your browser.

### 5. Usage

1. Type a natural-language query in the chat input (e.g., *"Export all Medline glove spend for last month"*)
2. Review the verification card — check columns, edit filters, read any warnings
3. Click **Confirm Export** to execute the query
4. Download the CSV from the sidebar

## Project Structure

```
ai_export_builder/
├── app.py                    # Streamlit entry point
├── config.py                 # pydantic-settings — reads .env
├── .env.example              # Template for local secrets
├── registry/
│   ├── connection.yaml       # Named DB connections → env-var references
│   ├── registry_views.yaml   # View definitions: columns, aliases, database
│   └── registry_common_dimensions.yaml
├── graph/
│   ├── state.py              # ExportState TypedDict for LangGraph
│   ├── workflow.py           # Graph definition & compilation
│   └── nodes/
│       ├── parse_intent.py   # LLM node: NL → ExportIntent
│       ├── validate_intent.py# Deterministic validation
│       └── execute_export.py # SQL gen → DB query → DataFrame
├── models/
│   └── intent.py             # ExportIntent + FilterItem Pydantic models
├── services/
│   ├── registry_loader.py    # Load YAML registry, alias index, routing
│   ├── sql_builder.py        # Jinja2 SQL rendering + RLS injection
│   ├── db.py                 # pyodbc connections (default + per-view)
│   ├── temporal.py           # Resolve "YTD", "last quarter" → dates
│   ├── rate_limiter.py       # In-memory daily request counter
│   └── audit.py              # JSON-lines audit logger
├── templates/
│   └── select_query.sql.j2   # Parameterized SQL template
├── ui/
│   ├── chat.py               # Streamlit chat components
│   └── verification_card.py  # Editable verification card
├── logs/                     # JSON-lines audit logs (gitignored)
└── tests/
    ├── test_registry.py      # Alias resolution, view lookups, routing
    ├── test_temporal.py       # Temporal expression tests
    ├── test_sql_builder.py    # Parameterized SQL, RLS, injection safety
    ├── test_validate_intent.py# Validation node logic
    ├── test_parse_intent.py   # Mock OpenAI parse tests
    └── test_integration.py    # Full workflow with mock LLM + DB
```

## Running Tests

```bash
pytest ai_export_builder/tests/ -v
```

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for intent parsing |
| `PRIME_DB_URL` | (required) | pyodbc connection string for PRIME database |
| `SCS_DB_URL` | (required) | pyodbc connection string for SCS/PBI database |
| `DAILY_REQUEST_LIMIT` | `10` | Max requests per user per day |
| `MAX_EXPORT_ROWS` | `100000` | Row limit per export query |
| `FISCAL_YEAR_START_MONTH` | `1` | January = 1, October = 10 |
| `TEST_USER_FACILITIES` | `["ALL"]` | JSON list of facility codes for RLS |
