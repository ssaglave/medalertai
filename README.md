# 🚑 MedAlertAI

**AI-Powered Emergency Medical Dispatch Analytics Platform**

MedAlertAI transforms Pittsburgh EMS and Fire dispatch records into actionable intelligence through MPDS complaint classification, demand forecasting, geographic hotspot detection, and a RAG-powered protocol assistant — all accessible via an interactive Plotly Dash dashboard.

---

## Architecture

```mermaid
graph LR
    subgraph Data Layer
        A[("📁 WPRDC<br/>EMS_Data.csv<br/>Fire_Data.csv")] --> B["🔧 Preprocessing<br/>(Greeshma)"]
        B --> C[("📊 Clean Parquet<br/>fact_dispatch_clean")]
        D[("🌐 Census API")] --> E["🔗 Demographic Join<br/>(Srileakhana)"]
        E --> F[("👥 Demographics<br/>Parquet")]
    end

    subgraph ML Models
        C --> G["🏷️ MPDS Classifier<br/>(Suvarna)"]
        C --> H["📈 Forecaster<br/>(Sanika)"]
        C --> I["🗺️ Clustering<br/>(Sanika)"]
    end

    subgraph RAG Pipeline
        J[("📄 Protocol Docs<br/>(URLs)")] --> K["📥 Ingest & Embed<br/>(Greeshma + Suvarna)"]
        K --> L[("🔮 ChromaDB<br/>(Sanika)")]
        L --> M["🤖 LangChain QA<br/>(Srileakhana)"]
    end

    subgraph Dashboard
        G & H & I & F & M --> N["📊 Plotly Dash<br/>(Deekshitha)"]
    end
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/ssaglave/medalertai.git
cd medalertai
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in your API keys
```

### 2. Download Data

```bash
python scripts/download_data.py
# Downloads EMS_Data.csv (~398 MB) and Fire_Data.csv (~165 MB) from WPRDC
```

### 3. Run Dashboard

```bash
python src/dashboard/app.py
# → http://localhost:8050
```

---

## Project Structure

```
medalertai/
├── config/
│   ├── contracts.py             ← Shared schemas & column contracts (ALL)
│   └── settings.py              ← Flask, paths, model defaults (Deekshitha)
├── scripts/
│   └── download_data.py         ← WPRDC data downloader (Greeshma)
├── data/
│   ├── raw/                     ← EMS_Data.csv, Fire_Data.csv (downloaded via script)
│   ├── processed/               ← Clean Parquets (gitignored, regenerated)
│   └── external/                ← Census & demographic source files
├── models/
│   └── artifacts/               ← Serialized models (gitignored, regenerated)
├── notebooks/                   ← EDA Jupyter notebooks
├── src/
│   ├── data/
│   │   ├── preprocessing.py     ← NEMSIS normalization (Greeshma)
│   │   ├── mpds_mapper.py       ← Call type → MPDS mapping (Suvarna)
│   │   ├── feature_engineering.py ← Temporal/geo features (Sanika)
│   │   ├── demographic_join.py  ← Census data join (Srileakhana)
│   │   └── schemas.py           ← Pydantic validation (Deekshitha)
│   ├── models/
│   │   ├── classifier/train.py  ← LightGBM MPDS classifier (Suvarna)
│   │   ├── forecasting/train.py ← Prophet + LightGBM ensemble (Sanika)
│   │   ├── clustering/train.py  ← DBSCAN + Isolation Forest (Sanika)
│   │   └── evaluate.py          ← Unified eval harness (Deekshitha)
│   ├── rag/
│   │   ├── ingest.py            ← Protocol doc chunking (Greeshma + Suvarna)
│   │   └── chain.py             ← LangChain QA with Claude (Srileakhana)
│   └── dashboard/
│       ├── app.py               ← Dash multi-page entry point (Deekshitha)
│       ├── pages/
│       │   ├── overview.py      ← KPIs, donut, bar charts (Greeshma)
│       │   ├── temporal.py      ← Trend lines, heatmaps (Srileakhana)
│       │   ├── geography.py     ← Choropleth, clusters (Sanika)
│       │   ├── forecast.py      ← Forecast + model toggle (Deekshitha)
│       │   ├── qa.py            ← Classification QA (Suvarna)
│       │   └── assistant.py     ← RAG chat interface (Srileakhana)
│       ├── components/
│       │   ├── filters.py       ← Global filter bar (Deekshitha)
│       │   ├── map_utils.py     ← Choropleth helpers (Deekshitha)
│       │   └── chat_ui.py       ← Chat component (Deekshitha)
│       └── assets/
│           └── custom.css
└── tests/
    ├── test_data.py             ← Data pipeline tests (Greeshma)
    ├── test_models.py           ← ML model tests (Suvarna + Sanika)
    ├── test_rag.py              ← RAG pipeline tests (Srileakhana + Deekshitha)
    └── test_dashboard.py        ← Dashboard integration tests (Deekshitha)
```

---

## Data

### Source

Pittsburgh EMS and Fire dispatch records from the [Western Pennsylvania Regional Data Center (WPRDC)](https://data.wprdc.org/dataset/ems-fire-dispatch-data).

Data is **not stored in the repo** due to file size. Run `python scripts/download_data.py` to fetch directly from WPRDC.

| File | Rows | Size | Download URL |
|---|---|---|---|
| `EMS_Data.csv` | ~2.3M | 398 MB | [WPRDC EMS](https://tools.wprdc.org/downstream/ff33ca18-2e0c-4cb5-bdcd-60a5dc3c0418) |
| `Fire_Data.csv` | ~985K | 165 MB | [WPRDC Fire](https://tools.wprdc.org/downstream/b6340d98-69a0-4965-a9b4-3480cea1182b) |

### Columns

| Column | Description |
|---|---|
| `call_id_hash` | Anonymized incident ID |
| `service` | EMS or Fire |
| `priority` / `priority_desc` | Priority code and description |
| `call_quarter` / `call_year` | Time period |
| `description_short` | Call type (e.g., FALL, CHEST PAIN, NATURAL GAS ISSUE) |
| `city_code` / `city_name` | Municipality |
| `geoid` | Census block group FIPS code |
| `census_block_group_center__x/y` | Block group centroid (lon/lat) |

---

## Dashboard Pages

| Page | Route | Owner | Description |
|---|---|---|---|
| **Overview** | `/` | Greeshma | KPIs, EMS vs Fire donut, top-10 MPDS bar, stacked area |
| **Temporal** | `/temporal` | Srileakhana | Quarterly trends, anomaly markers, day-hour heatmap |
| **Geography** | `/geography` | Sanika | Choropleth map, DBSCAN clusters, response equity |
| **Forecast** | `/forecast` | Deekshitha | 4-quarter forecast, model comparison toggle |
| **Classification QA** | `/classification-qa` | Suvarna | Agreement table, data quality, compliance trends |
| **Assistant** | `/assistant` | Srileakhana | RAG-powered protocol Q&A with Claude |

---

## ML Model Targets

| Model | Metric | Target |
|---|---|---|
| MPDS Classifier | Macro F1 | > 0.75 |
| Demand Forecaster | MAPE | < 15% |
| Hotspot Detection | Silhouette Score | > 0.4 |
| Hotspot Detection | Recall@20 | > 0.7 |
| RAG Pipeline | Precision@5 | > 0.6 |
| RAG Pipeline | Latency p50 | < 3s |

---

## Team

All 5 contributors work across all 5 phases. Each person has specific deliverables in every phase:

| Name | Phase 1: Data | Phase 2: ML Models | Phase 3: RAG | Phase 4: Dashboard | Phase 5: Testing |
|---|---|---|---|---|---|
| **Greeshma** | Preprocessing, clean Parquet | Training splits, feature contracts | Doc collection & conversion | Overview page | Data pipeline tests |
| **Suvarna** | MPDS mapper | MPDS Classifier (LightGBM) | MPDS protocol ingestion | Classification QA page | Classifier metrics |
| **Sanika** | Feature engineering | Forecaster + Clustering | ChromaDB vector store | Geography page | Forecaster & clustering metrics |
| **Srileakhana** | Demographic join | Ensemble + model serialization | LangChain QA chain | Temporal + Assistant pages | RAG evaluation |
| **Deekshitha** | Pydantic schemas, EDA | MLflow + evaluation harness | RAG eval suite | Forecast page + components | Integration tests, CI |

---

## Development Workflow

### Branch Strategy

```
main                         ← protected, always working
├── dev                      ← integration branch, PRs merge here first
├── phase0/bootstrap         ← Phase 0: Project scaffold & contracts
├── phase1/data-ingestion    ← Phase 1: Preprocessing, MPDS mapping, features, demographics, EDA
├── phase2/ml-models         ← Phase 2: Classifier, forecaster, clustering, evaluation
├── phase3/rag-pipeline      ← Phase 3: Doc ingestion, ChromaDB, LangChain QA chain
├── phase4/dashboard         ← Phase 4: All 6 Dash pages + components
└── phase5/testing           ← Phase 5: Tests, CI, final integration
```

All 5 contributors work on the same phase branch simultaneously, then merge to `dev` at each phase milestone.

### Workflow

```bash
# Setup
git checkout -b dev origin/main

# Start work on current phase
git checkout -b phase1/data-ingestion dev

# Each contributor works on their files within the phase branch
# Push and create PRs to dev at milestone completion
```

### PR Flow

1. Work on the current phase branch
2. Open PR → `dev` at phase milestone
3. Get at least 1 review
4. Merge to `dev`
5. Periodically: `dev` → `main` after integration testing

> ⚠️ Changes to `config/contracts.py` require **all 5 contributors to approve**.

---

## Project Progress Tracker

> **Last updated**: 2026-04-12 — Phase 0 complete ✅

### Phase 0 — Bootstrap ✅

| Task | Owner | Status |
|---|---|---|
| `src/data/` directory tree + `__init__.py` files | Suvarna | ✅ Done |
| `scripts/download_data.py` — WPRDC CSV downloader with URLs | Suvarna | ✅ Done |
| `src/models/classifier/` scaffolding | Suvarna | ✅ Done |
| `requirements.txt` — ML section (LightGBM, Optuna, MLflow, XGBoost) | Suvarna | ✅ Done |
| `src/models/forecasting/` + `src/models/clustering/` scaffolding | Suvarna | ✅ Done |
| `requirements.txt` — Prophet, DBSCAN, Isolation Forest deps | Suvarna | ✅ Done |
| `src/rag/` scaffolding | Suvarna | ✅ Done |
| `requirements.txt` — RAG section (LangChain, ChromaDB, sentence-transformers) | Suvarna | ✅ Done |
| `.env.example` | Suvarna | ✅ Done |
| `src/dashboard/app.py` — Dash skeleton with `use_pages=True` + Flask | Suvarna | ✅ Done |
| 6 blank page stubs in `src/dashboard/pages/` | Suvarna | ✅ Done |
| `config/settings.py` — Flask host/port/debug | Suvarna | ✅ Done |
| `.gitignore` | Suvarna | ✅ Done |
| `config/contracts.py` — NEMSIS v3 columns, Parquet schema, Dash contracts | Suvarna | ✅ Done |
| `README.md` + `implementation_plan.md` | Suvarna | ✅ Done |

---

### Phase 1 — Data Ingestion & EDA (Days 1–3)

| Task | Owner | Status |
|---|---|---|
| NEMSIS-aligned preprocessing → `fact_dispatch_clean.parquet` | Greeshma | ☐ |
| `mpds_mapper.py` — map `description_short` → MPDS codes (>80% coverage) | Suvarna | ☐ |
| Unit tests for MPDS mapper | Suvarna | ☐ |
| `feature_engineering.py` — temporal cyclical encoding, geo encoding, lag features | Sanika | ☐ |
| `demographic_join.py` — census block-group data join → `dim_cbg_demographics.parquet` | Srileakhana | ☐ |
| `schemas.py` — Pydantic NEMSIS-aligned validation | Deekshitha | ☐ |
| `01_eda_overview.ipynb` | Deekshitha | ☐ |
| `02_temporal_eda.ipynb` | Deekshitha | ☐ |
| `03_geo_eda.ipynb` | Deekshitha | ☐ |
| `data_completeness_pct` scoring per row | Deekshitha | ☐ |

**Milestone**: All clean Parquets written; >80% MPDS mapping confirmed.

---

### Phase 2 — ML Models (Days 3–7)

| Task | Owner | Status |
|---|---|---|
| Training splits for all 3 models; coordinate feature set contracts | Greeshma | ☐ |
| LightGBM + sklearn Pipeline; Optuna HPO (50 trials); target macro F1 >0.75 | Suvarna | ☐ |
| Flag disagreement rows (confidence >0.7) | Suvarna | ☐ |
| Prophet univariate + LightGBM ensemble (4-quarter horizon, MAPE <15%) | Sanika | ☐ |
| DBSCAN hotspot detection (eps=0.3, min_samples=5) | Sanika | ☐ |
| Isolation Forest anomaly detection (contamination=0.05) | Sanika | ☐ |
| LightGBM forecasting component + ensemble combiner | Srileakhana | ☐ |
| Serialize all models to `models/artifacts/` | Srileakhana | ☐ |
| MLflow tracking instrumentation for all 3 models | Deekshitha | ☐ |
| `evaluate.py` with all metric targets | Deekshitha | ☐ |
| CI test stubs for each model | Deekshitha | ☐ |

**Milestone**: All 3 models hitting targets; serialized artifacts in `models/artifacts/`.

---

### Phase 3 — RAG Pipeline (Days 5–7)

| Task | Owner | Status |
|---|---|---|
| Source document collection: PA DOH EMS protocols, NFPA 1221, WPRDC glossary → text chunks | Greeshma | ☐ |
| MPDS protocol reference + NEMSIS v3 data dictionary ingestion | Suvarna | ☐ |
| Chunk quality validation (no garbled OCR text) | Suvarna | ☐ |
| ChromaDB persistent vector store setup; batch embed with `all-MiniLM-L6-v2` | Sanika | ☐ |
| Verify ChromaDB store integrity | Sanika | ☐ |
| LangChain `RetrievalQA` chain with Claude `claude-haiku-4-5` | Srileakhana | ☐ |
| System prompt (protocol-only, citation format, fallback) | Srileakhana | ☐ |
| RAG evaluation: Precision@5 test suite | Deekshitha | ☐ |
| LLM-as-judge faithfulness scorer | Deekshitha | ☐ |
| Latency benchmark (p50 <3s, p95 <8s) | Deekshitha | ☐ |

**Milestone**: ChromaDB populated; `rag_chain.py` returning cited answers; latency targets met.

---

### Phase 4 — Dash Dashboard (Days 7–12)

| Task | Owner | Status |
|---|---|---|
| `pages/overview.py` — KPIs, EMS vs Fire donut, top-10 MPDS bar, stacked area | Greeshma | ☐ |
| `pages/qa.py` — agreement DataTable, completeness chart, compliance trend | Suvarna | ☐ |
| `pages/geography.py` — Mapbox choropleth, DBSCAN clusters, Response Equity tab | Sanika | ☐ |
| `pages/temporal.py` — quarterly trend, anomaly markers, day-hour heatmap | Srileakhana | ☐ |
| `pages/assistant.py` — chat interface, example prompts, source accordion | Srileakhana | ☐ |
| `pages/forecast.py` — 4-quarter forecast, uncertainty bands, model toggle | Deekshitha | ☐ |
| `components/filters.py` — global filter bar (year, service type, MPDS group) | Deekshitha | ☐ |
| `components/map_utils.py` — shared choropleth + cluster helpers | Deekshitha | ☐ |
| `components/chat_ui.py` — RAG chat component | Deekshitha | ☐ |

**Milestone**: All 6 pages rendering live data; global filters wired via `dcc.Store`.

---

### Phase 5 — Evaluation & Tests (Days 11–13)

| Task | Owner | Status |
|---|---|---|
| Data pipeline tests: MPDS coverage >80%, NEMSIS schema, completeness | Greeshma | ☐ |
| Classifier: macro F1 >0.75, confusion matrix, disagreement recall | Suvarna | ☐ |
| Forecaster: MAPE <15%, walk-forward CV | Sanika | ☐ |
| Hotspot: Silhouette >0.4, Recall@20 >0.7 | Sanika | ☐ |
| RAG: Precision@5 >0.6, faithfulness avg >1.5, latency p50 <3s | Srileakhana | ☐ |
| Integration tests (Dash callbacks + data) | Deekshitha | ☐ |
| `pytest` CI pipeline | Deekshitha | ☐ |
| Final `README.md` with Mermaid architecture diagram | Deekshitha | ☐ |

**Milestone**: All tests green; project submission ready.

---

## Tech Stack

| Category | Tools |
|---|---|
| **Dashboard** | Plotly Dash, Dash Bootstrap Components |
| **Visualization** | Plotly Express, Plotly Graph Objects |
| **ML** | LightGBM, XGBoost, Prophet, scikit-learn, Optuna |
| **RAG** | LangChain, ChromaDB, sentence-transformers, Claude claude-haiku-4-5 |
| **Geospatial** | GeoPandas, Plotly Mapbox |
| **Data** | Pandas, PyArrow, Pydantic |
| **Experiment Tracking** | MLflow |
| **Testing** | pytest, pytest-cov |
| **Deployment** | Flask (built-in dev server) |

---

## License

MIT
