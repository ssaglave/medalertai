"""
MedAlertAI Dashboard — Plotly Dash entry point.

Owner: Deekshitha (C5)
Phase: 4

Run:
  python src/dashboard/app.py
  → http://localhost:8050
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so config/ imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from src.dashboard.components.filters import create_filter_bar

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

app.title = "MedAlertAI"
dict.__setitem__(app.config, "use_pages", True)

# Expose the underlying Flask server
server = app.server

# Load data for filter initialization. The filter bar only needs unique
# year / service / call_type values, so read the small overview aggregate
# (~100 KB) instead of the 134 MB raw fact table. Fall back to a
# column-subset read of the raw parquet, then a tiny synthetic frame.
_PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
_OVERVIEW_AGG = _PROCESSED_DIR / "overview_agg.parquet"
_FACT_PARQUET = _PROCESSED_DIR / "fact_dispatch_clean.parquet"

if _OVERVIEW_AGG.exists():
    _agg = pd.read_parquet(
        _OVERVIEW_AGG, columns=["year", "service", "call_type"]
    )
    data_df = _agg.rename(
        columns={"year": "CALL_YEAR", "service": "service_type"}
    )
elif _FACT_PARQUET.exists():
    data_df = pd.read_parquet(
        _FACT_PARQUET, columns=["CALL_YEAR", "service_type", "call_type"]
    )
else:
    data_df = pd.DataFrame({
        'CALL_YEAR': [2023, 2024],
        'service_type': ['EMS', 'Fire'],
        'call_type': ['Respiratory', 'Trauma'],
    })

# ── Layout ──
app.layout = dbc.Container([
    # Navigation
    dbc.NavbarSimple(
        brand="🚑 MedAlertAI",
        brand_href="/",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Overview", href="/")),
            dbc.NavItem(dbc.NavLink("Temporal", href="/temporal")),
            dbc.NavItem(dbc.NavLink("Geography", href="/geography")),
            dbc.NavItem(dbc.NavLink("Forecast", href="/forecast")),
            dbc.NavItem(dbc.NavLink("Classification QA", href="/classification-qa")),
            dbc.NavItem(dbc.NavLink("Assistant", href="/assistant")),
        ],
    ),

    # Global Filter Bar (Critical Blocker — C5)
    create_filter_bar(data_df),

    # Page content
    html.Div(dash.page_container, className="mt-3"),

    # Footer
    html.Footer(
        html.P("MedAlertAI © 2026 — AI-Powered Emergency Dispatch Analytics",
               className="text-center text-muted mt-4 mb-2"),
    ),
], fluid=True, className="px-0")


if __name__ == "__main__":
    from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, FLASK_THREADED

    print(f"MedAlertAI Dashboard starting on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=FLASK_THREADED,
    )
