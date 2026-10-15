# EY Project — AI Assistant Portal

A Flask web application where authenticated users **chat with an AI assistant about their own data**. Upload a spreadsheet, PDF, Word or text file, ask questions in plain English, and get answers with **server-computed numbers** and **auto-generated charts** — including comparisons, rankings, relationships and trends, without having to name a column or a chart type.

> **Design principle:** the language model never does the maths. All totals, shares, correlations and chart data are computed in Python (pandas). The AI only chooses what to show and explains the result in words.

---

## Table of contents

1. [Features](#features)
2. [Tech stack](#tech-stack)
3. [Project structure](#project-structure)
4. [Getting started](#getting-started)
5. [Configuration](#configuration)
6. [How it works](#how-it-works)
7. [Chart questions you can ask](#chart-questions-you-can-ask)
8. [Routes](#routes)
9. [Data model](#data-model)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Troubleshooting](#troubleshooting)
13. [Known notes](#known-notes)

---

## Features

- **Authentication** — register, login, logout, profile, settings, password change (Flask-Login, CSRF-protected forms).
- **AI chat** — multi-conversation chat with history, rename/clear/delete, and export to **TXT, DOCX or PDF**.
- **File analysis** — upload `PDF`, `DOCX`, `TXT`, `CSV`, `XLSX`, `XLS` (16 MB default limit). The file is parsed, summarised, stored, and can be questioned later.
- **Deterministic analytics** — totals, averages, top-N, distinct values, counts, fiscal-year/quarter figures, margins and growth are calculated with pandas, not guessed by the model.
- **Automatic charts (PNG, server-side)** — bar, line, pie, scatter, heatmap, histogram, dashboards. Charts are returned inline in the chat.
- **Dynamic chart engine** — handles open-ended requests such as *"Create a graph comparing the different sales channels"* or *"Find the strongest relationships in the data"* on any dataset, and works even when the AI service is offline (see [How it works](#how-it-works)).
- **Natural-language chart customisation** — chart type, colours, titles, axis fields, background, date ranges.
- **Analytics dashboard** — per-user activity tracking (messages, uploads, response times).
- **Indian fiscal year support** — Q1 = Apr–Jun by default (configurable).
- **Messy-data tolerant** — schema detection works from values, not just English column names; numbers stored as text (e.g. `"1,234"`) are cleaned.

## Tech stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.10+ (tested on 3.13), Flask 3, Flask-SQLAlchemy, Flask-Login, Flask-WTF |
| Database | SQLite (`database/app.db`), auto-created and auto-migrated |
| AI | [Ollama Cloud](https://ollama.com) via the official `ollama` SDK, with a configurable model fallback chain |
| Data | pandas 2/3, NumPy, openpyxl, xlrd |
| Charts | Matplotlib (headless `Agg` backend), Seaborn |
| Documents | PyPDF2, python-docx, ReportLab |
| Frontend | Jinja2 templates, vanilla JS/CSS |
| Production | gunicorn |

## Project structure

```text
EY Project/
├── app.py                      # Application factory, blueprints, DB bootstrap + safe column migration
├── config.py                   # Dev / Production / Testing configuration (reads .env)
├── extensions.py               # db, login_manager, csrf
├── start.bat                   # Windows one-click launcher
├── requirements.txt
├── readme.md                   # This file
├── DOCUMENTATION.md            # Older notes (superseded by readme.md - safe to delete)
│
├── models/                     # SQLAlchemy models
│   ├── user.py                 #   User
│   ├── chat.py                 #   Conversation, ChatHistory (stores chart image)
│   ├── uploads.py              #   UploadedFile
│   └── analytics.py            #   AnalyticsLog
│
├── routes/                     # Flask blueprints
│   ├── auth.py                 #   register / login / profile / settings
│   ├── chatbot.py              #   /chat/* — the main question-answering pipeline
│   ├── uploads.py              #   /uploads/* — file upload and Q&A on a file
│   ├── analytics.py            #   /dashboard
│   └── charts.py               #   /api/charts/generate (see Known notes)
│
├── services/                   # Business logic (no Flask dependencies where possible)
│   ├── ai_service.py           #   Ollama client, model fallback chain, error handling
│   ├── file_service.py         #   Text/table extraction for all supported formats
│   ├── schema_service.py       #   Column type detection (number, currency, date, categorical, ...)
│   ├── data_profile_service.py #   Structured data profile fed to the planner
│   ├── insight_service.py      #   Deterministic computations (sums, top-N, distinct, FY/quarter, ...)
│   ├── chart_service.py        #   Matplotlib/Seaborn rendering, EY brand palette
│   ├── chart_intelligence.py   #   AI JSON chart decision -> strict validation -> pandas -> render
│   ├── chart_recommender.py    #   Alternative chart suggestions
│   ├── chart_customization.py  #   Parses "make it a pie chart in green titled ..." requests
│   ├── generic_chart_planner.py#   Business-language intents (profitability, growth, anomalies, ...)
│   ├── dynamic_chart_engine.py #   Comparison / ranking / share / relationship / overview engine
│   ├── prompt_guidance.py      #   System-prompt guidance for generic prompts and edge cases
│   ├── export_service.py       #   TXT / DOCX / PDF conversation export
│   └── synthetic_data.py       #   Realistic test-data generator
│
├── templates/                  # Jinja2 pages (landing, home, chat, uploads, dashboard, auth, errors)
├── static/                     # CSS / JS
├── tests/                      # unittest-style regression tests
├── tools/
│   └── generate_realistic_financials.py   # Builds a sample multi-year financial workbook
├── uploads/                    # Stored uploads (git-ignored)
└── database/                   # SQLite database (git-ignored)
```

## Getting started

### Prerequisites

- Python **3.10 or newer**
- An **Ollama Cloud API key** (free to create at <https://ollama.com/settings/keys>) — optional for charts and numeric answers, required for natural-language explanations.

### Install and run

```bash
# 1. Clone / unzip the project, then:
python -m venv .venv

# 2. Activate the environment
#    Windows (PowerShell):  .venv\Scripts\Activate.ps1
#    macOS / Linux:         source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your environment file
cp .env.example .env        # or create .env manually — see Configuration

# 5. Start the app
python app.py
```

Open <http://127.0.0.1:5000>, register an account, and start chatting.

**Windows shortcut:** double-click `start.bat`. It checks Python, installs dependencies, creates `uploads/` and `database/`, and starts the server.

On first launch the app creates the SQLite database and adds any missing columns automatically — no manual migrations.

## Configuration

Settings are read from a `.env` file in the project root.

```ini
# --- Flask ---
FLASK_SECRET_KEY=change-me-to-a-long-random-string
FLASK_ENV=development            # development | production | testing
FLASK_DEBUG=true

# --- AI (Ollama Cloud) ---
OLLAMA_API_KEY=your_key_here
OLLAMA_MODEL=gpt-oss:120b-cloud                  # cloud models end in -cloud
OLLAMA_FALLBACK_MODELS=qwen3:32b-cloud,llama3.1  # tried in order if the primary fails

# --- Uploads ---
MAX_CONTENT_LENGTH=16777216      # bytes (16 MB)

# --- Finance / charts ---
FISCAL_YEAR_START_MONTH=4        # 4 = Indian FY (Q1 = Apr-Jun), 1 = calendar year
CHART_PALETTE=ey                 # ey | blue | red | green | any seaborn/matplotlib palette
CHART_STYLE=darkgrid             # darkgrid | whitegrid | dark | white | ticks
CHART_CONTEXT=notebook           # paper | notebook | talk | poster
CHART_DPI=150
CHART_WIDTH_IN=10
CHART_HEIGHT_IN=6
CHART_MAX_POINTS=60
CHART_ALTERNATIVES=2             # number of alternative chart suggestions
CHART_AI_ENABLED=true            # let the AI choose columns/aggregations
CHART_TRIGGER_KEYWORDS=graph,chart,plot,visualize,trend,compare,comparison,versus,vs
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | dev placeholder | Session signing — **must be changed in production** |
| `FLASK_ENV` | `development` | Selects the config class |
| `OLLAMA_API_KEY` | _(empty)_ | Enables AI explanations; without it the app returns a helpful fallback message |
| `OLLAMA_MODEL` | `llama3.1` (config) | Primary model |
| `OLLAMA_FALLBACK_MODELS` | `qwen3:32b-cloud,llama3.1` | Backup models |
| `MAX_CONTENT_LENGTH` | 16 MB | Upload size cap |
| `FISCAL_YEAR_START_MONTH` | `4` | Fiscal quarter boundaries |
| `CHART_*` | see above | Chart appearance and behaviour |

> **Security:** `.env` is git-ignored. Never commit your API key or secret key.

## How it works

### Chat pipeline (`routes/chatbot.py` → `/chat/send`)

```text
User message (+ optional files)
   │
   ├─ parse file → DataFrame + schema/profile
   │
   ├─ deterministic shortcuts ........ "what are the distinct regions?", "how many orders?"
   │      (skipped when the user is actually asking for a chart)
   │
   ├─ chart request?
   │     ├─ customisation (type / colour / axes / title) ........ legacy renderer
   │     ├─ multi-metric comparison ("sales and profit by region") ... legacy renderer
   │     ├─ dynamic chart engine ........ comparison, ranking, share, trend-by-group,
   │     │                                 relationship, biggest changes, unusual, overview
   │     ├─ generic chart planner ....... profitability, growth, efficiency, anomalies, FY views
   │     └─ AI chart intelligence ....... AI picks columns -> validated -> pandas -> render
   │
   ├─ computed facts (authoritative numbers) + chart PNG
   │
   └─ AI writes the explanation using ONLY those facts
         └─ if the AI is unreachable, a deterministic markdown summary is shown instead
```

### Dynamic chart engine (`services/dynamic_chart_engine.py`)

Resolves your question against the **real columns** of the uploaded data:

1. **Intent** — compare, rank, share, trend-by-group, relationship, biggest changes, unusual values, or overview.
2. **Columns** — dimension (e.g. "channels" → `Sales Channel`, "categories" → `Category`), measure(s) (Sales by default; Profit alongside for comparisons) and date column, using plural/synonym matching. Words such as *chart type* are not mistaken for a column.
3. **Aggregation** — pandas sums for flows; ratios such as margin are recomputed from totals, not averaged.
4. **Rendering** — a multi-panel PNG in the EY palette, plus facts and a markdown summary.
5. **Transparency** — when it had to choose something for you, the assumption is stated in the answer (for example *"No grouping was named, so 'Sales Channel' was chosen automatically"*).

Example output for *"Create a graph comparing the different sales channels"*: grouped bars (Sales vs Profit per channel), a share-of-sales donut, and profit margin per channel against the overall average.

## Chart questions you can ask

You do not need to name a chart type or a column.

| Goal | Example prompt |
| --- | --- |
| Compare groups | *Create a graph comparing the different sales channels.* |
| Compare on a measure | *Compare profit across categories.* |
| Head-to-head | *Compare Online vs Retail Store sales.* |
| Ranking | *Show the top 5 products by sales.* |
| Share / contribution | *Show sales share by payment method.* · *Show the most important contributors to overall performance.* |
| Trend by group | *Show sales trend by region over time.* |
| Relationships | *Find the variables that are most closely related and show their relationship.* |
| Biggest changes | *Show me where the biggest changes are occurring.* |
| Unusual values | *Identify unusual patterns and visualize them.* |
| Open-ended | *Explore the dataset and create the most relevant visualization.* · *Build a visual overview without me specifying the chart type or columns.* |
| Customised | *Show sales by region as a green pie chart titled "Regional Mix".* |

Non-chart questions (*"What are the different sales channels?"*, *"How many orders are there?"*) still return a plain text answer.

## Routes

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Landing page (redirects to `/home` when logged in) |
| GET | `/home` | Authenticated home |
| GET/POST | `/register`, `/login` | Account creation and sign-in |
| GET | `/logout`, `/profile`, `/settings` | Account pages |
| POST | `/settings/edit`, `/settings/password` | Update profile / change password |
| GET | `/chat` | Chat UI |
| GET | `/chat/conversations` | List conversations |
| POST | `/chat/new` | New conversation |
| POST | `/chat/send` | Send a message (with optional files) |
| POST | `/chat/chart-choice` | Re-render a chart using a suggested alternative |
| GET | `/chat/history/<id>` | Load a conversation |
| DELETE | `/chat/delete/<id>` | Delete a conversation |
| POST | `/chat/clear/<id>` | Clear messages |
| GET | `/chat/export/<id>/<format>` | Export as `txt`, `docx` or `pdf` |
| GET | `/uploads` | Uploaded files page |
| POST | `/uploads/upload` | Upload and analyse a file |
| POST | `/uploads/ask/<id>` | Ask a question about a stored file |
| GET | `/uploads/details/<id>` | File details |
| DELETE | `/uploads/delete/<id>` | Delete a file |
| GET | `/dashboard`, `/dashboard/chart-data` | Activity analytics |

## Data model

| Table | Key columns |
| --- | --- |
| `users` | `id`, `username` (unique), `email` (unique), `password_hash`, `created_at` |
| `conversations` | `id`, `user_id`, `title`, `created_at`, `updated_at` |
| `chat_history` | `id`, `user_id`, `conversation_id`, `prompt`, `response`, `timestamp`, `response_time`, `chart_image` |
| `uploaded_files` | `id`, `user_id`, `filename`, `stored_filename`, `file_type`, `file_size`, `extracted_text`, `summary`, `upload_date` |
| `analytics_logs` | `id`, `user_id`, `activity_type`, `details`, `timestamp` |

Passwords are stored hashed. Missing columns are added automatically at startup by `_ensure_schema_up_to_date()`.

## Testing

```bash
python -m unittest discover -s tests -v
# or, if pytest is installed:
pytest tests
```

| Test file | Covers |
| --- | --- |
| `test_chatbot_file_upload.py` | Chat with attached files |
| `test_chart_intelligence.py` | AI chart decision validation and rendering |
| `test_generic_chart_planner.py` | Business-language chart intents |
| `test_dynamic_chart_engine.py` | Comparison / overview / relationship engine |

**Sample data:** generate a realistic multi-year workbook for manual testing:

```bash
python tools/generate_realistic_financials.py
```

## Deployment

```bash
pip install -r requirements.txt
export FLASK_ENV=production
export FLASK_SECRET_KEY="a-long-random-value"
export OLLAMA_API_KEY="..."
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

- Production mode sets `SESSION_COOKIE_SECURE=True`, so serve over **HTTPS**.
- Charts use the headless Matplotlib `Agg` backend, so no display server is needed (works on Render, Railway, etc.).
- SQLite is fine for a single instance. For several workers or instances, point `SQLALCHEMY_DATABASE_URI` at PostgreSQL or MySQL.
- Use persistent storage for `uploads/` and `database/`.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| *"The AI service is unavailable"* | `OLLAMA_API_KEY` missing/invalid, or the model name is wrong (cloud models end in `-cloud`). Charts and numeric answers still work; only the prose explanation is replaced by a built-in summary. |
| Asking for a graph returns only a list of values | Fixed in v1.1: the "distinct values" and "count" shortcuts are now skipped whenever the message is a chart request. If it still happens, confirm you are running the updated `routes/chatbot.py`. |
| Charts missing on pandas 3 | Fixed in v1.1: string columns use the `str` dtype in pandas 3, so `dtype == object` checks failed. The planner now checks for non-numeric columns instead. |
| Upload rejected | Only `pdf, docx, txt, csv, xlsx, xls` up to `MAX_CONTENT_LENGTH` are accepted. |
| Numbers look wrong in a "Gross Sales" vs "Sales" question | The engine uses `Sales` as the primary measure and states its assumption. Name the measure you want (e.g. "gross sales"). |
| `ModuleNotFoundError` | Activate the virtual environment and re-run `pip install -r requirements.txt`. |
| Port already in use | Set `PORT=5001` before `python app.py`. |

## Known notes

- `routes/charts.py` defines `/api/charts/generate` but its blueprint is **not registered** in `app.py`. Register `charts_bp` if you want that endpoint.
- `DOCUMENTATION.md` is an older document that still refers to `GEMINI_API_KEY`; the app now uses **Ollama Cloud** (`OLLAMA_API_KEY`). This `readme.md` supersedes it, and the old file can be deleted.
- Unusual-pattern questions on data with a date column are handled by the older time-series anomaly planner; without a date column the dynamic engine uses IQR outlier detection.
- The default `FLASK_SECRET_KEY` is a development placeholder — always override it outside local development.

## Changelog

### v1.1 — Dynamic chart engine
- **New:** `services/dynamic_chart_engine.py` — answers open-ended chart questions (compare, rank, share, trend by group, relationships, biggest changes, unusual values, overview) on any dataset, with comparison built in and stated assumptions.
- **Fixed:** "Create a graph comparing the different sales channels" returned only a list of distinct values, because the word *different* triggered the distinct-values shortcut. Shortcuts are now skipped for chart requests.
- **Fixed:** pandas 3 compatibility — `dtype == object` checks replaced with proper non-numeric checks in `generic_chart_planner.py`.
- **Fixed:** the words *chart type* are no longer mistaken for a `Customer Type` column, and the *sales* in "sales channels" is no longer treated as the only measure to compare.
- **Improved:** if the AI service is unreachable, a deterministic markdown summary is shown together with the chart.
- **Added:** `tests/test_dynamic_chart_engine.py`; this `readme.md`.

### v1.0 — Initial portal
- Authentication, AI chat with conversations and export (TXT/DOCX/PDF), file upload and analysis, analytics dashboard, deterministic financial insights, EY-themed charts, Indian fiscal-year support.