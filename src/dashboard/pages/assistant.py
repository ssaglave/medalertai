"""
pages/assistant.py — RAG Chat Assistant dashboard page.

Owner: Srileakhana (C4)
Phase: 4

Features:
  - Chat interface with dcc.Textarea + history div
  - Disclaimer notice
  - Example prompt buttons
  - Source accordion (citations)
  - Calls src.rag.chain in callback
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/assistant", name="Assistant", order=5)

layout = dbc.Container([
    html.H2("🤖 Protocol Assistant", className="my-3"),
    html.P("Ask questions about EMS/Fire dispatch protocols."),
    # TODO: Add chat UI, example prompts, source accordion
    html.Div(id="assistant-content", children=[
        dbc.Alert("RAG Assistant page — implementation in progress.", color="info"),
    ]),
])
