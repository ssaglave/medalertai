"""
components/chat_ui.py - RAG chat interface component.

Owner: Deekshitha (C5)
Phase: 4

Used by Srileakhana's Assistant page for Phase 4B.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import dash
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc

from src.rag.chain import FALLBACK_ANSWER, RagChainError, query


PROMPTS = [
    "What does the retrieved protocol say about MPDS 17D1?",
    "What NFPA guidance is available for dispatch response-time compliance?",
    "Which source explains NEMSIS fields used in this dashboard?",
    "Summarize any protocol caveats for high-priority EMS calls.",
]


def create_chat_container() -> dbc.Container:
    """Create the Assistant page chat UI."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                id="chat-messages",
                                children=[
                                    _create_message(
                                        "Hi, I am MedAlertAI. Ask a protocol, "
                                        "dispatch, MPDS, NEMSIS, NFPA, or WPRDC "
                                        "question and I will answer from retrieved sources.",
                                        sender="assistant",
                                    )
                                ],
                                className="assistant-chat-messages",
                            ),
                            dcc.Loading(
                                dbc.Alert(
                                    id="assistant-status",
                                    children="Ready",
                                    color="secondary",
                                    className="assistant-status",
                                ),
                                type="default",
                            ),
                            html.Div(
                                [
                                    html.P("Example prompts", className="fw-bold mb-2"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    prompt,
                                                    id=f"assistant-prompt-{index}",
                                                    color="secondary",
                                                    outline=True,
                                                    size="sm",
                                                    className="assistant-prompt-button w-100",
                                                ),
                                                md=6,
                                                className="mb-2",
                                            )
                                            for index, prompt in enumerate(PROMPTS)
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Label("Your question", html_for="chat-input", className="fw-bold"),
                            dcc.Textarea(
                                id="chat-input",
                                placeholder="Ask about a protocol, citation, response target, or dispatch data definition...",
                                rows=4,
                                className="assistant-input form-control",
                            ),
                            dbc.Button(
                                "Send",
                                id="send-btn",
                                color="primary",
                                className="mt-2",
                            ),
                            dcc.Store(id="chat-history-store", data=[]),
                        ],
                        lg=7,
                    ),
                    dbc.Col(
                        [
                            html.H5("Retrieved Sources", className="mb-3"),
                            html.Div(
                                id="source-accordion-container",
                                children=dbc.Alert(
                                    "Sources will appear here after the first answer.",
                                    color="info",
                                ),
                            ),
                        ],
                        lg=5,
                        className="mt-4 mt-lg-0",
                    ),
                ]
            )
        ],
        fluid=True,
        className="assistant-shell px-0",
    )


def _create_message(text: str, sender: str, timestamp: str | None = None) -> dbc.Row:
    """Create a single chat bubble."""
    timestamp = timestamp or datetime.now().strftime("%H:%M")
    is_user = sender == "user"
    align_class = "justify-content-end" if is_user else "justify-content-start"
    bubble_class = "assistant-message-user" if is_user else "assistant-message-bot"
    sender_label = "You" if is_user else "MedAlertAI"

    return dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(sender_label, className="assistant-message-sender"),
                        dcc.Markdown(text, className="assistant-message-text"),
                        html.Small(timestamp, className="text-muted"),
                    ]
                ),
                className=f"assistant-message {bubble_class}",
            ),
            md=9,
            lg=8,
        ),
        className=f"mb-3 {align_class}",
    )


def _create_sources_accordion(sources: list[dict[str, Any]]) -> dbc.Accordion | dbc.Alert:
    if not sources:
        return dbc.Alert(
            "No citations were returned. The assistant may not have found enough source context.",
            color="warning",
        )

    items = []
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or source.get("source_id") or f"Source {index}"
        citation = source.get("citation") or title
        snippet = source.get("snippet") or "No snippet available."
        url = source.get("url")
        file_name = source.get("file_name")

        body = [
            html.P(citation, className="fw-bold mb-2"),
            html.P(snippet, className="mb-2"),
        ]
        if file_name:
            body.append(html.Small(f"File: {file_name}", className="d-block text-muted"))
        if url:
            body.append(html.A("Open source", href=url, target="_blank", rel="noreferrer"))

        items.append(
            dbc.AccordionItem(
                body,
                title=f"{index}. {title}",
                item_id=f"source-{index}",
            )
        )

    return dbc.Accordion(items, start_collapsed=True, always_open=True)


def _resolve_prompt(triggered_id: Any, typed_question: str | None) -> str:
    if triggered_id == "send-btn":
        return (typed_question or "").strip()

    for index, prompt in enumerate(PROMPTS):
        if triggered_id == f"assistant-prompt-{index}":
            return prompt

    return ""


@callback(
    Output("chat-messages", "children"),
    Output("chat-input", "value"),
    Output("chat-history-store", "data"),
    Output("source-accordion-container", "children"),
    Output("assistant-status", "children"),
    Output("assistant-status", "color"),
    Input("send-btn", "n_clicks"),
    Input("assistant-prompt-0", "n_clicks"),
    Input("assistant-prompt-1", "n_clicks"),
    Input("assistant-prompt-2", "n_clicks"),
    Input("assistant-prompt-3", "n_clicks"),
    State("chat-input", "value"),
    State("chat-messages", "children"),
    State("chat-history-store", "data"),
    prevent_initial_call=True,
)
def update_chat(
    send_clicks,
    prompt_0_clicks,
    prompt_1_clicks,
    prompt_2_clicks,
    prompt_3_clicks,
    user_input,
    current_messages,
    chat_history,
):
    """Append a user question, call the RAG chain, and render citations."""
    del send_clicks, prompt_0_clicks, prompt_1_clicks, prompt_2_clicks, prompt_3_clicks

    current_messages = current_messages or []
    chat_history = chat_history or []
    question = _resolve_prompt(dash.ctx.triggered_id, user_input)

    if not question:
        return (
            current_messages,
            "",
            chat_history,
            dash.no_update,
            "Enter a question or choose an example prompt.",
            "warning",
        )

    new_messages = current_messages + [_create_message(question, sender="user")]

    try:
        result = query(question)
        answer = result.get("answer") or FALLBACK_ANSWER
        sources = result.get("sources", [])
        status = f"Answered with {len(sources)} retrieved source(s)."
        status_color = "success"
    except (RagChainError, ValueError, ImportError, FileNotFoundError) as exc:
        answer = (
            f"{FALLBACK_ANSWER}\n\n"
            f"Dashboard detail: `{exc}`"
        )
        sources = []
        status = "RAG assistant is not ready yet. Check that Ollama is running, the model is pulled, and the vector store/ingestion outputs exist."
        status_color = "warning"
    except Exception as exc:
        answer = (
            f"{FALLBACK_ANSWER}\n\n"
            f"Unexpected dashboard detail: `{exc}`"
        )
        sources = []
        status = "The RAG request failed. Check the dashboard logs for details."
        status_color = "danger"

    new_messages.append(_create_message(answer, sender="assistant"))
    chat_history.append(
        {
            "question": question,
            "answer": answer,
            "sources": sources,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )

    return (
        new_messages,
        "",
        chat_history,
        _create_sources_accordion(sources),
        status,
        status_color,
    )
