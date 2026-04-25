"""
src/dashboard/pages/temporal.py
--------------------------------
Phase 4B - Temporal Analysis Page
Owner: Greeshma (C1)
Route: /temporal

Charts:
  1. Quarter x Year Heatmap - EMS call volume
     Highlights anomaly spikes
  2. Slope Chart - volume shifts in top incident categories
     from earliest year to latest year in the data

Data sources (pre-aggregated by scripts/build_temporal_aggregates.py):
  - data/processed/temporal_heatmap_agg.parquet
  - data/processed/temporal_slope_agg.parquet
Listens to: dcc.Store('global-filter-store') set by components/filters.py
"""

from __future__ import annotations

from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc


dash.register_page(
    __name__,
    path="/temporal",
    name="Temporal",
    title="MedAlertAI - Temporal Analysis",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
HEATMAP_AGG_PATH = _REPO_ROOT / "data" / "processed" / "temporal_heatmap_agg.parquet"
SLOPE_AGG_PATH = _REPO_ROOT / "data" / "processed" / "temporal_slope_agg.parquet"

ACCENT_RED = "#E84545"
GRID_COLOR = "rgba(255,255,255,0.08)"
BG_COLOR = "rgba(0,0,0,0)"
FONT_COLOR = "#DEE2E6"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=FONT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, showline=False),
    yaxis=dict(gridcolor=GRID_COLOR, showline=False),
)

_heatmap_cache: pd.DataFrame | None = None
_slope_cache: pd.DataFrame | None = None


def _load_heatmap_agg() -> pd.DataFrame:
    global _heatmap_cache
    if _heatmap_cache is not None:
        return _heatmap_cache
    if HEATMAP_AGG_PATH.exists():
        _heatmap_cache = pd.read_parquet(HEATMAP_AGG_PATH)
    else:
        _heatmap_cache = _synthetic_heatmap_agg()
    return _heatmap_cache


def _load_slope_agg() -> pd.DataFrame:
    global _slope_cache
    if _slope_cache is not None:
        return _slope_cache
    if SLOPE_AGG_PATH.exists():
        _slope_cache = pd.read_parquet(SLOPE_AGG_PATH)
    else:
        _slope_cache = _synthetic_slope_agg()
    return _slope_cache


def _synthetic_heatmap_agg() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for year in [2020, 2021, 2022, 2023]:
        for quarter in ["Q1", "Q2", "Q3", "Q4"]:
            for service in ["EMS", "Fire"]:
                rows.append(
                    {
                        "year": year,
                        "quarter": quarter,
                        "service": service,
                        "call_count": int(rng.integers(200, 800)),
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_slope_agg() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    call_types = [
        "FALL", "CHEST PAIN", "BREATHING PROBLEM", "UNCONSCIOUS", "BACK PAIN",
        "NATURAL GAS ISSUE", "VEHICLE ACCIDENT", "FIRE ALARM", "TRAUMA",
    ]
    rows = []
    for year in [2020, 2021, 2022, 2023]:
        for service in ["EMS", "Fire"]:
            for ct in call_types:
                rows.append(
                    {
                        "year": year,
                        "service": service,
                        "call_type": ct,
                        "call_count": int(rng.integers(20, 400)),
                    }
                )
    return pd.DataFrame(rows)


def _selected_years(df: pd.DataFrame, filters: dict) -> list:
    selected = filters.get("years") or filters.get("year")
    if selected:
        return selected
    return sorted(df["year"].dropna().unique().tolist())


def _selected_services(filters: dict) -> list[str]:
    selected = filters.get("services") or filters.get("service_type")
    if selected:
        return selected
    return ["EMS", "Fire"]


def _flag_anomalies(df_agg: pd.DataFrame, value_col: str = "call_count") -> pd.DataFrame:
    artifact_path = _REPO_ROOT / "models" / "artifacts" / "isolation_forest_flags.parquet"

    if artifact_path.exists():
        flags = pd.read_parquet(artifact_path)
        flags.columns = [c.lower() for c in flags.columns]
        if "quarter" in flags.columns:
            flags["quarter"] = (
                flags["quarter"]
                .astype("string")
                .str.upper()
                .str.replace("QUARTER ", "Q", regex=False)
                .str.replace(" ", "", regex=False)
                .replace({"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"})
            )
        if {"year", "quarter", "is_anomaly"} <= set(flags.columns):
            df_agg = df_agg.merge(
                flags[["year", "quarter", "is_anomaly"]], on=["year", "quarter"], how="left"
            )
            df_agg["is_anomaly"] = df_agg["is_anomaly"].fillna(False)
            return df_agg

    q1 = df_agg[value_col].quantile(0.25)
    q3 = df_agg[value_col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    df_agg["is_anomaly"] = df_agg[value_col] > upper
    return df_agg


def _build_heatmap(filters: dict) -> go.Figure:
    df = _load_heatmap_agg()

    selected_years = _selected_years(df, filters)
    selected_service = _selected_services(filters)

    services_upper = [s.upper() for s in selected_service]
    mask = df["year"].isin(selected_years) & df["service"].str.upper().isin(services_upper)

    df_ems = df[mask & (df["service"].str.upper() == "EMS")].copy()

    df_agg = (
        df_ems.groupby(["year", "quarter"], observed=True)["call_count"]
        .sum()
        .reset_index()
    )

    if df_agg.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No data for selected filters")
        return fig

    df_agg = _flag_anomalies(df_agg)

    fig = px.density_heatmap(
        df_agg,
        x="year",
        y="quarter",
        z="call_count",
        color_continuous_scale="Blues",
        labels={"year": "Year", "quarter": "Quarter", "call_count": "EMS Calls"},
        title="EMS Call Volume - Quarter x Year Heatmap",
    )

    fig.update_traces(
        hovertemplate="<b>Year:</b> %{x}<br><b>Quarter:</b> %{y}<br><b>Calls:</b> %{z:,}<extra></extra>"
    )

    anomalies = df_agg[df_agg["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["year"],
                y=anomalies["quarter"],
                mode="markers",
                marker=dict(
                    symbol="square-open",
                    size=28,
                    color=ACCENT_RED,
                    line=dict(width=3, color=ACCENT_RED),
                ),
                name="Anomaly spike",
                hovertemplate="<b>Anomaly detected</b><br>Year: %{x}<br>Quarter: %{y}<extra></extra>",
            )
        )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(tickmode="linear", gridcolor=GRID_COLOR)
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Q1", "Q2", "Q3", "Q4"],
        gridcolor=GRID_COLOR,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title="Calls", tickfont=dict(color=FONT_COLOR)),
        legend=dict(font=dict(color=FONT_COLOR)),
    )

    return fig


def _build_slope_chart(filters: dict) -> go.Figure:
    df = _load_slope_agg()

    selected_years = _selected_years(df, filters)
    selected_service = _selected_services(filters)

    services_upper = [s.upper() for s in selected_service]
    mask = df["year"].isin(selected_years) & df["service"].str.upper().isin(services_upper)
    df_f = df[mask].copy()

    if df_f.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No data for selected filters")
        return fig

    year_min = int(df_f["year"].min())
    year_max = int(df_f["year"].max())

    if year_min == year_max:
        fig = go.Figure()
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Only one year ({year_min}) selected - slope chart needs 2+ years",
        )
        return fig

    df_ends = df_f[df_f["year"].isin([year_min, year_max])]

    counts = (
        df_ends.groupby(["year", "call_type"], observed=True)["call_count"]
        .sum()
        .reset_index(name="count")
    )

    top_types = (
        counts.groupby("call_type")["count"]
        .sum()
        .nlargest(10)
        .index.tolist()
    )

    counts = counts[counts["call_type"].isin(top_types)]

    pivot = counts.pivot(index="call_type", columns="year", values="count").fillna(0)
    pivot = pivot.reset_index()

    fig = go.Figure()
    colours = px.colors.qualitative.Set2

    for i, row in pivot.iterrows():
        call_type = row["call_type"]
        val_a = row.get(year_min, 0)
        val_b = row.get(year_max, 0)
        colour = colours[i % len(colours)]

        fig.add_trace(
            go.Scatter(
                x=[year_min, year_max],
                y=[val_a, val_b],
                mode="lines+markers+text",
                line=dict(color=colour, width=2),
                marker=dict(size=9, color=colour),
                text=[f"{call_type}<br>{int(val_a):,}", f"{call_type}<br>{int(val_b):,}"],
                textposition=["middle left", "middle right"],
                textfont=dict(size=11, color=colour),
                name=call_type,
                hovertemplate=(
                    f"<b>{call_type}</b><br>"
                    "Year: %{x}<br>"
                    "Calls: %{y:,}<extra></extra>"
                ),
            )
        )

    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(
        tickvals=[year_min, year_max],
        ticktext=[str(year_min), str(year_max)],
        range=[year_min - 0.6, year_max + 0.6],
        gridcolor=GRID_COLOR,
    )
    fig.update_yaxes(
        title="Call Volume",
        gridcolor=GRID_COLOR,
    )
    fig.update_layout(
        title=f"Top Incident Categories - Volume Shift {year_min} to {year_max}",
        showlegend=False,
    )
    return fig


def layout() -> html.Div:
    return html.Div(
        [
            dbc.Row(
                dbc.Col(
                    html.H2(
                        "Temporal Analysis",
                        className="mb-0",
                        style={"color": FONT_COLOR, "fontWeight": 700},
                    ),
                    width=12,
                ),
                className="mb-1 mt-2",
            ),
            dbc.Row(
                dbc.Col(
                    html.P(
                        "EMS call volume patterns over time - heatmap view and incident category trend shifts.",
                        className="text-muted mb-3",
                    ),
                    width=12,
                ),
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span(
                                        [
                                            html.Strong("Quarter x Year Heatmap"),
                                            html.Span(
                                                " - red squares mark anomaly spikes",
                                                style={"fontSize": "0.82rem", "color": "#adb5bd"},
                                            ),
                                        ]
                                    ),
                                    style={"background": "rgba(255,255,255,0.04)"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="temporal-heatmap",
                                        config={"displayModeBar": False},
                                        style={"height": "380px"},
                                    )
                                ),
                            ],
                            className="shadow-sm border-0 mb-4",
                        ),
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.Span(
                                        [
                                            html.Strong("Incident Category Slope Chart"),
                                            html.Span(
                                                " - volume shifts from first to last year",
                                                style={"fontSize": "0.82rem", "color": "#adb5bd"},
                                            ),
                                        ]
                                    ),
                                    style={"background": "rgba(255,255,255,0.04)"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="temporal-slope",
                                        config={"displayModeBar": False},
                                        style={"height": "420px"},
                                    )
                                ),
                            ],
                            className="shadow-sm border-0 mb-4",
                        ),
                        width=12,
                    ),
                ]
            ),
            dcc.Store(id="temporal-init-trigger", data=True),
        ],
        style={"padding": "0 1.5rem"},
    )


@callback(
    Output("temporal-heatmap", "figure"),
    Input("global-filter-store", "data"),
    Input("temporal-init-trigger", "data"),
    prevent_initial_call=False,
)
def update_heatmap(filters_data, _trigger):
    return _build_heatmap(filters_data or {})


@callback(
    Output("temporal-slope", "figure"),
    Input("global-filter-store", "data"),
    Input("temporal-init-trigger", "data"),
    prevent_initial_call=False,
)
def update_slope(filters_data, _trigger):
    return _build_slope_chart(filters_data or {})
