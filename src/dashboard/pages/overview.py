"""
pages/overview.py — Overview dashboard page.

Owner: Greeshma (C1)
Phase: 4

Features:
  - KPI tile row (dbc.Card)
  - EMS vs Fire donut chart (px.pie)
  - Top-10 MPDS bar chart (px.bar)
  - Stacked area by year (px.area)
  - Reads from fact_dispatch_clean.parquet
"""
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", name="Overview", order=0)

layout = dbc.Container([
    html.H2("📊 Overview", className="my-3"),
    html.P("KPIs and high-level dispatch analytics."),
    # TODO: Add KPI cards, donut chart, bar chart, stacked area
    html.Div(id="overview-content", children=[
        dbc.Alert("Overview page — implementation in progress.", color="info"),
    ]),
])
