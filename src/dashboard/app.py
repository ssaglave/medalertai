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

# Expose the underlying Flask server
server = app.server

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

    # Global filter store
    dcc.Store(id="global-filters", data={}),

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

    print(f"🚑 MedAlertAI Dashboard starting on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=FLASK_THREADED,
    )
