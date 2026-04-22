# MedAlertAI — Implementation Plan v1 (Plotly Dash Dashboard) — Flask Deployment
## 5-Contributor × 5-Phase Task Division

---

## Dashboard Tool: Plotly Dash (Multi-page)

**Framework**: Plotly Dash with `use_pages=True`  
**Visualization**: Plotly Express + Plotly Graph Objects  
**Geospatial**: GeoPandas + Plotly Mapbox choropleth  
**Deployment**: Flask built-in server serving Dash app (`python app.py` or `flask run`)

```
src/
  dashboard/
    app.py               ← Dash app entry point (use_pages=True, Flask server)
    pages/
      overview.py        ← /  (KPIs, donut, top-10 MPDS, stacked area)
      temporal.py        ← /temporal  (trends, anomaly markers, heatmap)
      geography.py       ← /geography  (choropleth, clusters, equity tab)
      forecast.py        ← /forecast  (4-quarter lookahead, model toggle)
      qa.py              ← /classification-qa  (agreement table, QA panel)
      assistant.py       ← /assistant  (RAG chat, example prompts)
    components/
      filters.py         ← Global filter bar (year, service type, MPDS group)
      map_utils.py       ← Shared choropleth + cluster helpers
      chat_ui.py         ← RAG chat component
    assets/
      custom.css
```

---

## Flask Deployment Details

### `app.py` entry point

```python
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

# Expose the underlying Flask server
server = app.server

app.layout = dbc.Container([
    # ... layout code ...
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    # Flask built-in dev server
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True,    # set False in production
        threaded=True   # handle concurrent requests
    )
```

### Run commands

```bash
# Development (recommended)
python src/dashboard/app.py
# → Runs Flask dev server on http://0.0.0.0:8050 with debug=True, threaded=True

# Alternative using Flask CLI
export FLASK_APP=src.dashboard.app:server
flask run --host=0.0.0.0 --port=8050 --debug
```

### `config/settings.py` Flask settings

```python
# Deployment settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8050
FLASK_DEBUG = True          # Set to False for production
FLASK_THREADED = True       # Enable threaded mode for concurrent requests
```

### `requirements.txt` changes

```diff
- gunicorn>=21.2
+ flask>=3.0
  dash>=2.17
  dash-bootstrap-components>=1.6
  plotly>=5.22
  pandas>=2.2
  geopandas>=0.14
```

> **Note**: Dash already depends on Flask internally, so adding `flask>=3.0` is mainly to pin a minimum version explicitly. Gunicorn is removed entirely.

### Production considerations

Flask's built-in development server is **not designed for production-grade traffic**. It is single-threaded by default and lacks process management. For a classroom/demo/internal deployment this is fine. If you anticipate real production traffic, consider placing Flask behind a reverse proxy (e.g., Nginx) with `--threaded` or use `waitress` as a lightweight WSGI alternative on macOS/Windows.

A major practical benefit of Flask over Gunicorn: **Flask works natively on Windows**, which is useful if any team member is developing on a Windows machine. Gunicorn is Linux/macOS only.

---

## Contributor Assignments

| Contributor | Name | Primary Domain | Secondary |
|---|---|---|---|
| **C1** | **Greeshma** | Data Engineering & Ingestion | Schema contracts |
| **C2** | **Suvarna** | ML — Classification & QA | Evaluation harness |
| **C3** | **Sanika** | ML — Forecasting & Clustering | Pipeline integration |
| **C4** | **Srileakhana** | RAG Pipeline & LLM | Chat UI backend |
| **C5** | **Deekshitha** | Dashboard (Plotly Dash) & Testing | DevOps/glue |

---

## Data Sources

### Primary Data (included in repo via Git LFS)
- `data/raw/EMS_Data.csv` — Pittsburgh EMS dispatch records (~2.3M rows, 398 MB)
- `data/raw/Fire_Data.csv` — Pittsburgh Fire dispatch records (~985K rows, 165 MB)

**Source**: [WPRDC — EMS & Fire Dispatch Data](https://data.wprdc.org/dataset/ems-fire-dispatch-data)

### Raw CSV Columns
| Column | Description |
|---|---|
| `_id` | Row ID |
| `call_id_hash` | Anonymized incident identifier |
| `service` | EMS or Fire |
| `priority` | Priority code (e.g., E4, F1) |
| `priority_desc` | Priority description |
| `call_quarter` | Quarter (Q1–Q4) |
| `call_year` | Year |
| `description_short` | Short call description (e.g., FALL, BACK PAIN, NATURAL GAS ISSUE) |
| `city_code` | City abbreviation |
| `city_name` | City full name |
| `geoid` | Census block group FIPS code |
| `census_block_group_center__x` | Longitude (block group centroid) |
| `census_block_group_center__y` | Latitude (block group centroid) |

### Reference Documents (RAG sources — fetched via URL at runtime)
- PA DOH EMS Protocols
- NFPA 1221 Standard
- WPRDC Data Dictionary
- NEMSIS v3 Data Dictionary
- MPDS Protocol References

---

## Phase 0 — Bootstrap (Day 1)

All 5 contributors set up their domain's scaffolding on Day 1.

| Contributor | Name | Task |
|---|---|---|
| **C1** | **Greeshma** | `src/data/` directory tree + `__init__.py` files; `scripts/download_data.py` for WPRDC CSVs; raw schema print |
| **C2** | **Suvarna** | `src/models/classifier/` scaffolding; `requirements.txt` ML section (LightGBM, Optuna, MLflow, XGBoost) |
| **C3** | **Sanika** | `src/models/forecasting/` + `src/models/clustering/` scaffolding; Prophet + DBSCAN + Isolation Forest deps |
| **C4** | **Srileakhana** | `src/rag/` scaffolding; `requirements.txt` RAG section (LangChain, ChromaDB, sentence-transformers, anthropic); `.env.example` |
| **C5** | **Deekshitha** | `src/dashboard/app.py` Dash app skeleton with `use_pages=True` and Flask `app.run()` block; blank page stubs; `config/settings.py` (with Flask host/port/debug); `.gitignore` |

**Shared deliverable by EOD Day 1**: `config/contracts.py` committed — locked NEMSIS v3 column names, Parquet schema, and Dash callback input/output contracts.

---

## Phase 1 — Data Ingestion & EDA (Days 1–3)

| Contributor | Name | Task | Deliverable |
|---|---|---|---|
| **C1** | **Greeshma** | NEMSIS-aligned preprocessing: normalize columns, coerce dtypes, rename to NEMSIS v3 equivalents; output `fact_dispatch_clean.parquet` | Clean Parquet |
| **C2** | **Suvarna** | `mpds_mapper.py`: map raw `description_short` → MPDS complaint codes (>80% coverage target); unit tests | `mpds_mapper.py` + tests |
| **C3** | **Sanika** | Feature engineering: temporal cyclical encoding (sin/cos hour/day/month), geographic target encoding, forecasting lag features | `feature_engineering.py` |
| **C4** | **Srileakhana** | Demographic join: census block-group data (poverty rate, population density, race/ethnicity); `dim_cbg_demographics.parquet` | Demographics Parquet |
| **C5** | **Deekshitha** | Pydantic NEMSIS-aligned schema validation; EDA notebooks `01_eda_overview.ipynb`, `02_temporal_eda.ipynb`, `03_geo_eda.ipynb`; `data_completeness_pct` scoring per row | 3 EDA notebooks + schemas |

> **Note**: All contributors can start Phase 1 immediately using the raw `EMS_Data.csv` and `Fire_Data.csv` files available in `data/raw/`. No prerequisite data processing needed — everyone reads the same CSVs.

**Milestone**: All clean Parquets written; >80% MPDS mapping confirmed.

---

## Phase 2 — ML Models (Days 3–7)

| Contributor | Name | Track | Task |
|---|---|---|---|
| **C1** | **Greeshma** | Support | Training splits for all 3 models; coordinate feature set contracts |
| **C2** | **Suvarna** | 2a — MPDS Classifier | LightGBM + sklearn Pipeline; Optuna HPO (50 trials); flag disagreement rows (confidence >0.7); target macro F1 >0.75 |
| **C3** | **Sanika** | 2b — Forecaster + 2c — Clustering | Prophet univariate + LightGBM ensemble (4-quarter horizon, walk-forward CV, MAPE <15%); DBSCAN hotspot (eps=0.3, min_samples=5); Isolation Forest anomaly (contamination=0.05) |
| **C4** | **Srileakhana** | 2b Ensemble + model serialization | LightGBM forecasting component + ensemble combiner; serialize all models to `models/artifacts/` for Dash callbacks |
| **C5** | **Deekshitha** | MLflow tracking + evaluation harness | Instrument all 3 models; `evaluate.py` with all metric targets; CI test stubs for each model |

**Milestone**: All 3 models hitting targets; serialized artifacts in `models/artifacts/`.

---

## Phase 3 — RAG Pipeline (Days 5–7, parallel with Phase 2)

| Contributor | Name | Task |
|---|---|---|
| **C1** | **Greeshma** | Source document collection and conversion: PA DOH EMS protocols, NFPA 1221, WPRDC glossary → text chunks via PyPDF2/python-docx |
| **C2** | **Suvarna** | MPDS protocol reference + NEMSIS v3 data dictionary ingestion; chunk quality validation (no garbled OCR text) |
| **C3** | **Sanika** | ChromaDB persistent vector store setup; batch embed all chunks with `all-MiniLM-L6-v2` (384-dim); verify store integrity |
| **C4** | **Srileakhana** | LangChain `RetrievalQA` chain with Claude `claude-haiku-4-5`; system prompt (protocol-only, citation format, fallback); `rag_chain.py` module |
| **C5** | **Deekshitha** | RAG evaluation: Precision@5 test suite; LLM-as-judge faithfulness scorer; latency benchmark (p50 <3s, p95 <8s target) |

**Milestone**: ChromaDB populated; `rag_chain.py` returning cited answers; latency targets met.

---

## Phase 4 — Dash Dashboard (Days 7–12)

**Overview Tab (`pages/overview.py`)**: Designed **separately & collaboratively** as the team's shared landing page containing the core KPIs, EMS vs Fire donut chart (`px.pie`), top-8 incident category bar chart, and the all-services historical line chart.
*Advanced Chart Additions:*
- **Sankey Chart** (`go.Sankey`): Visualizing the triage flow of incidents (`Service Type` → `Priority Level` → `Top Call Types`).
- **Funnel Chart** (`px.funnel`): Visualizing the dispatch lifecycle drop-off (`Total 911 Calls` → `Units Dispatched` → `Arrived on Scene` → `Transported`).

The remaining 5 specialized tabs are distributed 1:1 among the 5 contributors based on their domain strengths, matching the provided HTML mockup UI exactly:

| Contributor | Name | Tab | Implementation Details (Plotly Dash) |
|---|---|---|---|
| **C1** | **Greeshma** | `pages/temporal.py` (Temporal) | Builds the Quarter × Year Heatmap (`px.density_heatmap`) showing EMS call volume. Highlights specific cells corresponding to Isolation Forest anomalous spikes. **New:** Adds a **Slope Chart** to visualize rank/volume shifts in top incident categories from 2020 to 2023. |
| **C2** | **Suvarna** | `pages/qa.py` (QA / Classification) | Builds the Incident Classification QA `dash_table.DataTable` with conditional formatting (Match/Review/Mismatch badges). **New:** Uses a **Bullet Chart** (`go.Indicator`) to track 'Data Completeness' and 'Response Time Compliance' rigidly against the 90% NFPA performance targets. |
| **C3** | **Sanika** | `pages/geography.py` (Geography) | Builds the Plotly Mapbox choropleth overlaid with DBSCAN hotspot scatter markers (`go.Scattermapbox`), and the Response Equity scatter plot (`px.scatter` poverty vs compliance). |
| **C4** | **Srileakhana**| `pages/rag.py` (RAG Assistant) | Builds the chat UI with prompt chips, text input, message history pane, and dynamically formats the backend `rag_chain.py` answers with source citations. |
| **C5** | **Deekshitha**| `pages/forecast.py` (Forecast) | Builds the 4-quarter lookahead line chart with uncertainty bands (`go.Scatter` with `fill='tonexty'`), toggle buttons for Prophet/LightGBM/Ensemble, and quarterly stat tiles. |

**Deployment**: `python src/dashboard/app.py` → Flask dev server on `http://0.0.0.0:8050`

**Milestone**: All 6 pages rendering live data (Overview + 5 specialized tabs); global filters wired via `dcc.Store`.

---

## Phase 5 — Evaluation & Tests (Days 11–13)

| Contributor | Name | Responsibility | Target Metric |
|---|---|---|---|
| **C1** | **Greeshma** | Data pipeline tests: MPDS coverage, NEMSIS schema, completeness scoring | MPDS >80%, Pydantic pass |
| **C2** | **Suvarna** | Classifier: macro F1, confusion matrix, disagreement flagging recall | F1 >0.75 |
| **C3** | **Sanika** | Forecaster: MAPE, walk-forward CV; Hotspot: Silhouette, Recall@20 | MAPE <15%, Silhouette >0.4, Recall@20 >0.7 |
| **C4** | **Srileakhana** | RAG: Precision@5, LLM-as-judge faithfulness, latency p50/p95 | Precision@5 >0.6, Faithfulness avg >1.5, p50 <3s |
| **C5** | **Deekshitha** | Integration tests (Dash callbacks + data); `pytest` CI; `README.md` with Mermaid diagram | All CI green |

---

## Critical Dependencies & Handoff Points

```
Day 1 EOD  → config/contracts.py committed (all 5 align on schema)
Day 3 EOD  → fact_dispatch_clean.parquet ready (C2, C3, C4 unblock for Phase 2)
Day 5 EOD  → rag_chain.py importable (C4 dashboard page unblocks)
Day 7 EOD  → models/artifacts/ populated (C1–C5 dashboard callbacks unblock)
Day 10 EOD → All Dash pages rendering (C5 integration tests unblock)
Day 13 EOD → All tests green → Phase 5 complete
```

> **Note**: All contributors can work on Phase 1 independently using the raw CSVs.
> Phase 2+ has natural sequential dependencies (models → dashboard), but dashboard page
> layout/UI can be developed in parallel using placeholder data.

---

## Key Files

| Path | Owner | Phase |
|---|---|---|
| `config/contracts.py` | All | 0 |
| `config/settings.py` | Deekshitha (C5) | 0 |
| `src/data/preprocessing.py` | Greeshma (C1) | 1 |
| `src/data/mpds_mapper.py` | Suvarna (C2) | 1 |
| `src/data/feature_engineering.py` | Sanika (C3) | 1 |
| `src/data/demographic_join.py` | Srileakhana (C4) | 1 |
| `src/data/schemas.py` | Deekshitha (C5) | 1 |
| `src/models/classifier/train.py` | Suvarna (C2) | 2 |
| `src/models/forecasting/train.py` | Sanika (C3) | 2 |
| `src/models/clustering/train.py` | Sanika (C3) | 2 |
| `src/rag/ingest.py` | Greeshma (C1) + Suvarna (C2) | 3 |
| `src/rag/chain.py` | Srileakhana (C4) | 3 |
| `src/dashboard/app.py` | Deekshitha (C5) | 4 |
| `src/dashboard/pages/*.py` | All (see Phase 4 table) | 4 |
| `src/dashboard/components/*.py` | Deekshitha (C5) | 4 |
| `tests/` | All (see Phase 5 table) | 5 |
| `README.md` | Deekshitha (C5) | 5 |

---

## Verification

1. **Data**: `python scripts/download_data.py` → CSVs; `pytest tests/test_data.py`
2. **Models**: `python src/models/classifier/train.py` → check MLflow UI for F1 >0.75
3. **RAG**: `python src/rag/ingest.py` → ChromaDB populated; `python -c "from src.rag.chain import query; print(query('MPDS 17D1'))"`
4. **Dashboard**: `python src/dashboard/app.py` → `localhost:8050`; check all 6 pages + global filters
5. **All tests**: `pytest --cov=src tests/` → all green, coverage report

---

## Flask vs. Gunicorn — Summary of Changes

| Aspect | Original (Gunicorn) | Revised (Flask) |
|---|---|---|
| **Server** | Gunicorn WSGI server | Flask built-in dev server |
| **Run command** | `gunicorn src.dashboard.app:server -b 0.0.0.0:8050` | `python src/dashboard/app.py` |
| **Concurrency** | Pre-fork worker model (multi-process) | Single-process, `threaded=True` |
| **Dependencies** | `gunicorn>=21.2` in requirements | `flask>=3.0` (already a Dash dependency) |
| **OS compatibility** | Linux/macOS only | Cross-platform (Windows/macOS/Linux) |
| **Debug mode** | Requires `--reload` flag | `debug=True` in `app.run()` |
| **Production readiness** | Production-grade | Development/demo-grade (add Nginx for production) |
| **Config location** | CLI flags or `gunicorn.conf.py` | `config/settings.py` + `app.run()` kwargs |
