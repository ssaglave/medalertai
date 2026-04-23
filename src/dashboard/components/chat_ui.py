"""
components/chat_ui.py — RAG chat interface component.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - Chat message display
  - Input textarea
  - Example prompt buttons
"""

import dash
from dash import dcc, html, callback, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
from datetime import datetime


def create_chat_container() -> dbc.Container:
    """
    Create a chat UI component for the Assistant page.
    
    Returns:
        dbc.Container with chat messages, input, and prompt buttons
    """
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("🤖 MedAlertAI Assistant", className="mb-3"),
                html.P("Ask questions about emergency dispatch patterns, forecasts, and insights.",
                       className="text-muted")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                # Chat message history
                html.Div(
                    id='chat-messages',
                    children=[
                        _create_message(
                            "Hi! I'm MedAlertAI. Ask me anything about your dispatch data.",
                            sender='bot',
                            timestamp=datetime.now().strftime('%H:%M')
                        )
                    ],
                    className="border rounded p-3 mb-3",
                    style={
                        'height': '400px',
                        'overflowY': 'auto',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                
                # Example prompts
                html.Div([
                    html.P("📌 Try asking:", className="fw-bold mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button(
                            "What are peak hours?",
                            id={'type': 'prompt-btn', 'index': 0},
                            size='sm',
                            className="mb-2"
                        ),
                        dbc.Button(
                            "Forecast next quarter",
                            id={'type': 'prompt-btn', 'index': 1},
                            size='sm',
                            className="mb-2"
                        ),
                        dbc.Button(
                            "Hotspot analysis",
                            id={'type': 'prompt-btn', 'index': 2},
                            size='sm',
                            className="mb-2"
                        ),
                        dbc.Button(
                            "Service type trends",
                            id={'type': 'prompt-btn', 'index': 3},
                            size='sm',
                            className="mb-2"
                        ),
                    ], vertical=True, className="w-100")
                ], className="mb-3"),
                
                # Input area
                dbc.InputGroup([
                    dbc.Input(
                        id='chat-input',
                        type='textarea',
                        placeholder='Type your question here...',
                        rows=3,
                        className="form-control"
                    ),
                    dbc.Button(
                        "Send ✓",
                        id='send-btn',
                        color='primary',
                        className="btn-block"
                    )
                ], className="mb-3"),
                
                # Store for chat history
                dcc.Store(id='chat-history-store', data=[])
                
            ], width=12)
        ]),
        
    ], fluid=True)


def _create_message(text: str, sender: str = 'user', timestamp: str = None) -> dbc.Row:
    """
    Helper function to create a chat message component.
    
    Args:
        text: Message text
        sender: 'user' or 'bot'
        timestamp: Message timestamp
    
    Returns:
        dbc.Row with formatted message
    """
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%H:%M')
    
    is_user = sender == 'user'
    bg_color = '#e3f2fd' if is_user else '#f5f5f5'
    text_align = 'end' if is_user else 'start'
    col_width = 8
    
    return dbc.Row([
        dbc.Col(width=12) if not is_user else dbc.Col(width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P(text, className="mb-1"),
                    html.Small(timestamp, className="text-muted")
                ])
            ], className=f"bg-light")
        ], width=col_width),
        dbc.Col(width=4) if is_user else dbc.Col(width=12),
    ], className="mb-2")


@callback(
    Output('chat-messages', 'children'),
    Output('chat-input', 'value'),
    Output('chat-history-store', 'data'),
    Input('send-btn', 'n_clicks'),
    Input({'type': 'prompt-btn', 'index': MATCH}, 'n_clicks'),
    State('chat-input', 'value'),
    State('chat-messages', 'children'),
    State('chat-history-store', 'data'),
    prevent_initial_call=True
)
def update_chat(send_clicks, prompt_clicks, user_input, current_messages, chat_history):
    """
    Update chat messages when user sends a message or clicks a prompt button.
    """
    
    # Determine which input was triggered
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_messages, '', chat_history
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle prompt button clicks
    if 'prompt-btn' in triggered_id:
        prompts = [
            "What are peak hours?",
            "Forecast next quarter",
            "Hotspot analysis",
            "Service type trends"
        ]
        prompt_idx = eval(triggered_id.split('index')[1].strip('":'))
        user_input = prompts[prompt_idx]
    
    if not user_input or not user_input.strip():
        return current_messages, '', chat_history
    
    # Add user message
    new_messages = current_messages + [
        _create_message(user_input, sender='user')
    ]
    
    # Simulate bot response (replace with actual RAG chain call)
    bot_response = f"I received your question: '{user_input}'. This would connect to the RAG pipeline for context-aware responses."
    new_messages = new_messages + [
        _create_message(bot_response, sender='bot')
    ]
    
    # Update history
    chat_history.append({'user': user_input, 'bot': bot_response})
    
    return new_messages, '', chat_history
