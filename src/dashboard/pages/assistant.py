"""
pages/assistant.py - RAG Chat Assistant dashboard page.

Owner: Srileakhana (C4)
Phase: 4
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

from src.dashboard.components.chat_ui import create_chat_container


dash.register_page(__name__, path="/assistant", name="Assistant", order=5)


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Protocol Assistant", className="my-3"),
                        html.P(
                            "Ask protocol and dispatch-policy questions against the "
                            "MedAlertAI RAG knowledge base.",
                            className="text-muted",
                        ),
                    ],
                    lg=8,
                ),
            ]
        ),
        dbc.Alert(
            [
                html.Strong("Clinical safety note: "),
                "This assistant summarizes retrieved EMS, fire dispatch, MPDS, "
                "NEMSIS, NFPA, and WPRDC reference material. It is not medical "
                "direction and does not replace official protocols or supervisor review.",
            ],
            color="warning",
            className="mb-3",
        ),
        create_chat_container(),
    ],
    fluid=True,
)
