"""
pages/forecast.py — Demand Forecasting dashboard page.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - 4-quarter forecast with go.Scatter uncertainty bands (fill='tonexty')
  - Prophet/LightGBM/Ensemble dcc.RadioItems toggle
  - Loads model artifacts
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/forecast", name="Forecast", order=3)

layout = dbc.Container([
    html.H2("🔮 Demand Forecast", className="my-3"),
    html.P("4-quarter demand forecast with model comparison."),
    # TODO: Add forecast chart with uncertainty bands, model toggle
    html.Div(id="forecast-content", children=[
        dbc.Alert("Forecast page — implementation in progress.", color="info"),
    ]),
])
