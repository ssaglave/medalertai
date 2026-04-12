"""
pages/geography.py — Geographic Analysis dashboard page.

Owner: Sanika (C3)
Phase: 4

Features:
  - Plotly Mapbox choropleth (block-group density)
  - DBSCAN cluster scatter overlay
  - Response Equity tab (call burden vs poverty scatter)
  - Loads cluster Parquet
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/geography", name="Geography", order=2)

layout = dbc.Container([
    html.H2("🗺️ Geographic Analysis", className="my-3"),
    html.P("Choropleth maps, hotspot clusters, and response equity analysis."),
    # TODO: Add choropleth, cluster overlay, equity tab
    html.Div(id="geography-content", children=[
        dbc.Alert("Geography page — implementation in progress.", color="info"),
    ]),
])
