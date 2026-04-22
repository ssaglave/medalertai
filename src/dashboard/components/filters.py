"""
components/filters.py — Global filter bar component.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - Year multi-select dropdown
  - Service type (EMS/Fire) toggle
  - MPDS group multi-select
  - Updates dcc.Store('global-filter-store')
"""

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd


def create_filter_bar(data_df: pd.DataFrame) -> dbc.Row:
    """
    Create a global filter bar component.

    Args:
        data_df: DataFrame with columns: CALL_YEAR, service_type, call_type

    Returns:
        dbc.Row containing the filter UI and dcc.Store for shared state
    """

    # Extract unique values for dropdowns (use actual parquet column names)
    years = sorted(data_df['CALL_YEAR'].unique().tolist())
    services = sorted(data_df['service_type'].unique().tolist())
    call_types = sorted(data_df['call_type'].dropna().unique().tolist())

    return dbc.Row([
        # Global Filter Store (all callbacks wire to this)
        dcc.Store(id='global-filter-store', data={
            'years': years,
            'services': services,
            'call_types': call_types
        }),

        dbc.Col([
            dbc.Label("Year(s)", className="fw-bold"),
            dcc.Dropdown(
                id='year-filter',
                options=[{'label': str(y), 'value': y} for y in years],
                value=years,  # Default: all years selected
                multi=True,
                placeholder="Select year(s)"
            )
        ], md=3),

        dbc.Col([
            dbc.Label("Service Type", className="fw-bold"),
            dcc.Dropdown(
                id='service-filter',
                options=[{'label': s, 'value': s} for s in services],
                value=services,  # Default: all services selected
                multi=True,
                placeholder="Select service type(s)"
            )
        ], md=3),

        dbc.Col([
            dbc.Label("Incident Category", className="fw-bold"),
            dcc.Dropdown(
                id='mpds-filter',
                options=[{'label': m, 'value': m} for m in call_types],
                value=[],  # Default: no filter (show all)
                multi=True,
                placeholder="Filter by category (optional)"
            )
        ], md=5),

        # Reset button
        dbc.Col([
            html.Br(),
            dbc.Button("Reset Filters", id='reset-filters-btn',
                       color="secondary", size="sm", className="mt-1")
        ], md=1),

    ], className="mb-4 p-3 border rounded bg-dark")


@callback(
    Output('global-filter-store', 'data'),
    Input('year-filter', 'value'),
    Input('service-filter', 'value'),
    Input('mpds-filter', 'value'),
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=False
)
def update_global_filters(years, services, call_types, reset_clicks):
    """
    Update the global filter store whenever any filter changes.
    Other pages' callbacks subscribe to 'global-filter-store' data.
    """
    return {
        'years': years if years else [],
        'services': services if services else [],
        'call_types': call_types if call_types else []
    }

