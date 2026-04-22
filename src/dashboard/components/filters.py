"""
components/filters.py — Global filter bar component.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - Year multi-select dropdown
  - Service type (EMS/Fire) toggle
  - MPDS group multi-select
  - Updates dcc.Store('global-filters')
"""

import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd


def create_filter_bar(data_df: pd.DataFrame) -> dbc.Row:
    """
    Create a global filter bar component.
    
    Args:
        data_df: DataFrame with columns: call_year, service, description_short (MPDS category)
    
    Returns:
        dbc.Row containing the filter UI and dcc.Store for shared state
    """
    
    # Extract unique values for dropdowns
    years = sorted(data_df['call_year'].unique())
    services = sorted(data_df['service'].unique())
    mpds_categories = sorted(data_df['description_short'].unique())
    
    return dbc.Row([
        # Global Filter Store (all callbacks wire to this)
        dcc.Store(id='global-filter-store', data={
            'years': years,
            'services': services,
            'mpds_categories': mpds_categories
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
            dbc.Label("MPDS Category", className="fw-bold"),
            dcc.Dropdown(
                id='mpds-filter',
                options=[{'label': m, 'value': m} for m in mpds_categories],
                value=mpds_categories,  # Default: all categories selected
                multi=True,
                placeholder="Select MPDS category(ies)"
            )
        ], md=6),
        
        # Clear button
        dbc.Col([
            html.Br(),
            dbc.Button("Reset Filters", id='reset-filters-btn', color="secondary", size="sm")
        ], md=12, className="mt-3")
        
    ], className="mb-4 p-3 border rounded bg-light")


@callback(
    Output('global-filter-store', 'data'),
    Input('year-filter', 'value'),
    Input('service-filter', 'value'),
    Input('mpds-filter', 'value'),
    Input('reset-filters-btn', 'n_clicks'),
    prevent_initial_call=False
)
def update_global_filters(years, services, mpds, reset_clicks):
    """
    Update the global filter store whenever any filter changes.
    Other pages' callbacks subscribe to 'global-filter-store' data.
    """
    return {
        'years': years if years else [],
        'services': services if services else [],
        'mpds_categories': mpds if mpds else []
    }
