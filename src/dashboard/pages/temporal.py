"""
src/dashboard/pages/temporal.py
--------------------------------
Phase 4B — Temporal Analysis Page
Owner: Greeshma (C1)
Route: /temporal

Charts:
  1. Quarter × Year Heatmap   — EMS call volume (px.density_heatmap)
                                 Highlights Isolation Forest anomaly spikes
  2. Slope Chart              — Rank/volume shifts in top incident categories
                                 from earliest year to latest year in the data

Data source: data/processed/fact_dispatch_clean.parquet
Listens to:  dcc.Store('global-filters') set by components/filters.py
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

# ---------------------------------------------------------------------------
# Register page
# ---------------------------------------------------------------------------
dash.register_page(
    __name__,
    path="/temporal",
    name="Temporal",
    title="MedAlertAI — Temporal Analysis",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
PARQUET_PATH = _REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"

# ---------------------------------------------------------------------------
# Colour palette (consistent with DARKLY theme)
# ---------------------------------------------------------------------------
ACCENT_BLUE   = "#4A90D9"
ACCENT_ORANGE = "#E87A2F"
ACCENT_RED    = "#E84545"
ACCENT_GREEN  = "#2ECC71"
GRID_COLOR    = "rgba(255,255,255,0.08)"
BG_COLOR      = "rgba(0,0,0,0)"        # transparent — let Bootstrap handle bg
FONT_COLOR    = "#DEE2E6"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=BG_COLOR,
    font=dict(color=FONT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, showline=False),
    yaxis=dict(gridcolor=GRID_COLOR, showline=False),
)

# ---------------------------------------------------------------------------
# Data loader  (cached so it only reads disk once per server session)
# ---------------------------------------------------------------------------
_df_cache: pd.DataFrame | None = None


def _load_data() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    if not PARQUET_PATH.exists():
        # Return a small synthetic sample so the page still renders
        # during development before the real parquet is available.
        _df_cache = _synthetic_sample()
        return _df_cache

    df = pd.read_parquet(PARQUET_PATH)

    # Normalise column names to lowercase for robustness
    df.columns = [c.lower() for c in df.columns]

    # Accept either naming convention from preprocessing
    col_map = {
        "call_year":    "year",
        "call_quarter": "quarter",
        "service":      "service",
        "description_short": "call_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Keep only EMS rows for the volume heatmap (per spec)
    _df_cache = df
    return _df_cache


def _synthetic_sample() -> pd.DataFrame:
    """Tiny placeholder so the page renders without real data."""
    rng = np.random.default_rng(42)
    years = [2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    services = ["EMS", "Fire"]
    call_types = ["FALL", "CHEST PAIN", "BREATHING PROBLEM",
                  "UNCONSCIOUS", "BACK PAIN", "NATURAL GAS ISSUE",
                  "VEHICLE ACCIDENT", "FIRE ALARM", "TRAUMA"]
    rows = []
    for y in years:
        for q in quarters:
            n = rng.integers(200, 800)
            for _ in range(int(n)):
                rows.append({
                    "year":     y,
                    "quarter":  q,
                    "service":  rng.choice(services),
                    "call_type": rng.choice(call_types),
                    "completeness_score": float(rng.uniform(0.6, 1.0)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly detection  (simple IQR-based flag when Isolation Forest
# artifacts are not yet available — swapped out once models land)
# ---------------------------------------------------------------------------

def _flag_anomalies(df_agg: pd.DataFrame, value_col: str = "call_count") -> pd.DataFrame:
    """
    Mark rows as anomalous using IQR rule.
    When models/artifacts/isolation_forest_flags.parquet exists,
    this function merges those labels instead.
    """
    artifact_path = _REPO_ROOT / "models" / "artifacts" / "isolation_forest_flags.parquet"
    if artifact_path.exists():
        flags = pd.read_parquet(artifact_path)
        # Expected columns: year, quarter, is_anomaly (bool)
        df_agg = df_agg.merge(flags, on=["year", "quarter"], how="left")
        df_agg["is_anomaly"] = df_agg["is_anomaly"].fillna(False)
        return df_agg

    # Fallback: IQR rule
    q1 = df_agg[value_col].quantile(0.25)
    q3 = df_agg[value_col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df_agg["is_anomaly"] = df_agg[value_col] > upper
    return df_agg


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _build_heatmap(filters: dict) -> go.Figure:
    """
    Quarter × Year density heatmap of EMS call volume.
    Anomalous cells get a bright red border overlay.
    """
    df = _load_data()

    # Apply global filters
    selected_years = filters.get("years") or sorted(df["year"].dropna().unique().tolist())
    selected_service = filters.get("service") or ["EMS", "Fire"]

    mask = df["year"].isin(selected_years)
    if "service" in df.columns:
        mask &= df["service"].str.upper().isin([s.upper() for s in selected_service])

    df_ems = df[mask & (df.get("service", pd.Series(["EMS"] * len(df))).str.upper() == "EMS")].copy() \
        if "service" in df.columns else df[mask].copy()

    # Aggregate
    df_agg = (
        df_ems.groupby(["year", "quarter"], observed=True)
        .size()
        .reset_index(name="call_count")
    )

    if df_agg.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No data for selected filters")
        return fig

    df_agg = _flag_anomalies(df_agg)

    # Pivot for heatmap
    pivot = df_agg.pivot(index="quarter", columns="year", values="call_count").fillna(0)

    fig = px.density_heatmap(
        df_agg,
        x="year",
        y="quarter",
        z="call_count",
        color_continuous_scale="Blues",
        labels={"year": "Year", "quarter": "Quarter", "call_count": "EMS Calls"},
        title="EMS Call Volume — Quarter × Year Heatmap",
    )
    fig.update_traces(
        hovertemplate="<b>Year:</b> %{x}<br><b>Q%{y}</b><br><b>Calls:</b> %{z:,}<extra></extra>"
    )

    # Overlay anomaly markers
    anomalies = df_agg[df_agg["is_anomaly"]]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["year"],
            y=anomalies["quarter"],
            mode="markers",
            marker=dict(
                symbol="square-open",
                size=28,
                color=ACCENT_RED,
                line=dict(width=3, color=ACCENT_RED),
            ),
            name="⚠ Anomaly spike",
            hovertemplate=(
                "<b>Anomaly detected</b><br>"
                "Year: %{x}<br>Quarter: Q%{y}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        yaxis=dict(
            tickvals=[1, 2, 3, 4],
            ticktext=["Q1", "Q2", "Q3", "Q4"],
            gridcolor=GRID_COLOR,
        ),
        xaxis=dict(tickmode="linear", gridcolor=GRID_COLOR),
        coloraxis_colorbar=dict(title="Calls", tickfont=dict(color=FONT_COLOR)),
        legend=dict(font=dict(color=FONT_COLOR)),
    )
    return fig


def _build_slope_chart(filters: dict) -> go.Figure:
    """
    Slope chart showing rank / volume shifts of top incident categories
    between the earliest and latest year in the filtered data.
    """
    df = _load_data()

    if "call_type" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="call_type column not available")
        return fig

    selected_years = filters.get("years") or sorted(df["year"].dropna().unique().tolist())
    selected_service = filters.get("service") or ["EMS", "Fire"]

    mask = df["year"].isin(selected_years)
    if "service" in df.columns:
        mask &= df["service"].str.upper().isin([s.upper() for s in selected_service])

    df_f = df[mask].copy()

    if df_f.empty or "year" not in df_f.columns:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, title="No data for selected filters")
        return fig

    year_min = int(df_f["year"].min())
    year_max = int(df_f["year"].max())

    if year_min == year_max:
        fig = go.Figure()
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Only one year ({year_min}) selected — slope chart needs 2+ years",
        )
        return fig

    # Count per (year, call_type) for earliest and latest year only
    df_ends = df_f[df_f["year"].isin([year_min, year_max])]
    counts = (
        df_ends.groupby(["year", "call_type"], observed=True)
        .size()
        .reset_index(name="count")
    )

    # Find top-10 call types by total volume across both years
    top_types = (
        counts.groupby("call_type")["count"]
        .sum()
        .nlargest(10)
        .index.tolist()
    )
    counts = counts[counts["call_type"].isin(top_types)]

    # Pivot
    pivot = counts.pivot(index="call_type", columns="year", values="count").fillna(0)
    pivot = pivot.reset_index()

    fig = go.Figure()

    colours = px.colors.qualitative.Set2
    for i, row in pivot.iterrows():
        cat   = row["call_type"]
        val_a = row.get(year_min, 0)
        val_b = row.get(year_max, 0)
        colour = colours[i % len(colours)]

        # Line connecting the two years
        fig.add_trace(go.Scatter(
            x=[year_min, year_max],
            y=[val_a, val_b],
            mode="lines+markers+text",
            line=dict(color=colour, width=2),
            marker=dict(size=9, color=colour),
            text=[f"{cat}<br>{int(val_a):,}", f"{int(val_b):,}"],
            textposition=["middle left", "middle right"],
            textfont=dict(size=11, color=colour),
            name=cat,
            hovertemplate=(
                f"<b>{cat}</b><br>"
                "Year: %{x}<br>"
                "Calls: %{y:,}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Top Incident Categories — Volume Shift {year_min} → {year_max}",
        xaxis=dict(
            tickvals=[year_min, year_max],
            ticktext=[str(year_min), str(year_max)],
            range=[year_min - 0.6, year_max + 0.6],
            gridcolor=GRID_COLOR,
        ),
        yaxis=dict(title="Call Volume", gridcolor=GRID_COLOR),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout() -> html.Div:  # noqa: D401
    return html.Div([

        # ── Page header ──────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(html.H2(
                "📅 Temporal Analysis",
                className="mb-0",
                style={"color": FONT_COLOR, "fontWeight": 700},
            ), width=12),
            className="mb-1 mt-2",
        ),
        dbc.Row(
            dbc.Col(html.P(
                "EMS call volume patterns over time — heatmap view and "
                "incident category trend shifts.",
                className="text-muted mb-3",
            ), width=12),
        ),

        # ── Chart 1 — Quarter × Year Heatmap ─────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Span([
                            "🌡 ",
                            html.Strong("Quarter × Year Heatmap"),
                            html.Span(
                                "  — red squares mark Isolation Forest anomaly spikes",
                                style={"fontSize": "0.82rem", "color": "#adb5bd"},
                            ),
                        ]),
                        style={"background": "rgba(255,255,255,0.04)"},
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            id="temporal-heatmap",
                            config={"displayModeBar": False},
                            style={"height": "380px"},
                        )
                    ),
                ], className="shadow-sm border-0 mb-4"),
                width=12,
            ),
        ]),

        # ── Chart 2 — Slope Chart ─────────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Span([
                            "📈 ",
                            html.Strong("Incident Category Slope Chart"),
                            html.Span(
                                "  — volume shifts from first to last year",
                                style={"fontSize": "0.82rem", "color": "#adb5bd"},
                            ),
                        ]),
                        style={"background": "rgba(255,255,255,0.04)"},
                    ),
                    dbc.CardBody(
                        dcc.Graph(
                            id="temporal-slope",
                            config={"displayModeBar": False},
                            style={"height": "420px"},
                        )
                    ),
                ], className="shadow-sm border-0 mb-4"),
                width=12,
            ),
        ]),

        # Hidden store to trigger initial load
        dcc.Store(id="temporal-init-trigger", data=True),

    ], style={"padding": "0 1.5rem"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("temporal-heatmap", "figure"),
    Output("temporal-slope",   "figure"),
    Input("global-filters",        "data"),
    Input("temporal-init-trigger", "data"),
    prevent_initial_call=False,
)
def update_charts(filters_data, _trigger):
    """Re-render both charts whenever global filters change."""
    filters = filters_data or {}
    heatmap_fig = _build_heatmap(filters)
    slope_fig   = _build_slope_chart(filters)
    return heatmap_fig, slope_fig