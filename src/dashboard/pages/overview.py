"""
pages/overview.py — Overview dashboard page.

Lead: Suvarna (C2) — Phase 4A collaborative page
Phase: 4

Charts:
  1. KPI tile row (dbc.Card)  — Total calls, EMS count, Fire count, avg/quarter
  2. EMS vs Fire donut        (px.pie)
  3. Top-8 incident bar       (px.bar)
  4. Stacked area by year     (px.area)
  5. Sankey chart              (go.Sankey) — Service → Priority → Call Type
  6. Priority Distribution Funnel (px.funnel)
  7. Service Pipeline Bar      (px.bar horizontal)

Data source: data/processed/fact_dispatch_clean.parquet
"""

import sys
from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Page Registration ──
dash.register_page(__name__, path="/", name="Overview", order=0)

# ── Data Loading ──
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PARQUET = _REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"

try:
    DF = pd.read_parquet(_PARQUET)
except FileNotFoundError:
    DF = pd.DataFrame()

# ── Color Palette ──
COLORS = {
    "ems":        "#00d4ff",   # cyan
    "fire":       "#ff6b35",   # orange
    "accent":     "#a855f7",   # purple
    "bg_card":    "#1a1a2e",   # dark card
    "bg_card2":   "#16213e",
    "text":       "#e0e0e0",
    "success":    "#10b981",
    "warning":    "#f59e0b",
}

PRIORITY_COLOR_MAP = {
    "Life Threatening":  "#ef4444",
    "ALS":               "#f97316",
    "BLS":               "#eab308",
    "Non-Emergency":     "#22c55e",
    "Other":             "#6b7280",
}


# ═══════════════════════════════════════════════════════════════
# Helper: apply global filters to DataFrame
# ═══════════════════════════════════════════════════════════════
def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter dataframe based on global-filter-store data."""
    if not filters or df.empty:
        return df

    years = filters.get("years", [])
    services = filters.get("services", [])
    call_types = filters.get("call_types", [])

    mask = pd.Series(True, index=df.index)

    if years:
        mask &= df["CALL_YEAR"].isin(years)
    if services:
        mask &= df["service_type"].isin(services)
    if call_types:
        mask &= df["call_type"].isin(call_types)

    return df[mask]


# ═══════════════════════════════════════════════════════════════
# Helper: map priority_description to broad triage level
# ═══════════════════════════════════════════════════════════════
def _map_priority_level(desc: str) -> str:
    """Map verbose priority descriptions to broad triage levels."""
    if pd.isna(desc):
        return "Other"
    desc_lower = desc.lower()
    if "life threatening" in desc_lower:
        return "Life Threatening"
    elif "advanced life support" in desc_lower or "als" in desc_lower:
        return "ALS"
    elif "basic life support" in desc_lower or "bls" in desc_lower:
        return "BLS"
    elif any(k in desc_lower for k in ["assist", "admin", "non emergency",
                                        "no immediate threat", "mark out"]):
        return "Non-Emergency"
    else:
        return "Other"


# ═══════════════════════════════════════════════════════════════
# KPI Card builder
# ═══════════════════════════════════════════════════════════════
def _kpi_card(title: str, value, icon: str, color: str) -> dbc.Col:
    """Create a single KPI card."""
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "2rem"}),
                    html.H3(f"{value:,.0f}" if isinstance(value, (int, float))
                             else str(value),
                             className="kpi-value mb-0 mt-2",
                             style={"color": color, "fontSize": "2.2rem",
                                    "fontWeight": "700"}),
                    html.P(title, className="kpi-label mt-1 mb-0",
                           style={"opacity": "0.7", "fontSize": "0.85rem"}),
                ], style={"textAlign": "center"})
            ])
        ], className="kpi-card h-100",
           style={"backgroundColor": COLORS["bg_card"],
                  "border": f"1px solid {color}33",
                  "borderRadius": "12px"}),
        xs=12, sm=6, md=3, className="mb-3"
    )


# ═══════════════════════════════════════════════════════════════
# Chart builders  (pure functions — no side effects)
# ═══════════════════════════════════════════════════════════════

def _build_kpi_row(df: pd.DataFrame) -> dbc.Row:
    """KPI tile row: Total, EMS, Fire, Avg per Quarter."""
    total = len(df)
    ems = (df["service_type"] == "EMS").sum()
    fire = (df["service_type"] == "Fire").sum()
    n_quarters = df.groupby(["CALL_YEAR", "CALL_QUARTER"]).ngroups
    avg_q = total / max(n_quarters, 1)

    return dbc.Row([
        _kpi_card("Total Calls", total, "📞", COLORS["accent"]),
        _kpi_card("EMS Calls", ems, "🚑", COLORS["ems"]),
        _kpi_card("Fire Calls", fire, "🔥", COLORS["fire"]),
        _kpi_card("Avg / Quarter", avg_q, "📊", COLORS["success"]),
    ])


def _build_donut(df: pd.DataFrame) -> go.Figure:
    """EMS vs Fire donut chart."""
    counts = df["service_type"].value_counts().reset_index()
    counts.columns = ["Service", "Count"]

    fig = px.pie(
        counts, names="Service", values="Count",
        hole=0.55,
        color="Service",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.05),
        title=dict(text="EMS vs Fire Distribution", x=0.5,
                   font=dict(size=16)),
    )
    return fig


def _build_top8_bar(df: pd.DataFrame) -> go.Figure:
    """Top-8 incident category horizontal bar chart."""
    top = (df["call_type"]
           .value_counts()
           .head(8)
           .sort_values(ascending=True)
           .reset_index())
    top.columns = ["Category", "Count"]

    fig = px.bar(
        top, x="Count", y="Category", orientation="h",
        color="Count",
        color_continuous_scale=["#1e3a5f", COLORS["ems"]],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=10, r=10),
        coloraxis_showscale=False,
        title=dict(text="Top 8 Incident Categories", x=0.5,
                   font=dict(size=16)),
        yaxis_title="",
        xaxis_title="Number of Calls",
    )
    return fig


def _build_stacked_area(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart by year and service type."""
    yearly = (df.groupby(["CALL_YEAR", "service_type"])
              .size()
              .reset_index(name="Count"))

    fig = px.area(
        yearly, x="CALL_YEAR", y="Count", color="service_type",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
        labels={"CALL_YEAR": "Year", "Count": "Calls",
                "service_type": "Service"},
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=30, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.15),
        title=dict(text="Call Volume Over Time", x=0.5,
                   font=dict(size=16)),
        xaxis=dict(dtick=1),
    )
    return fig


def _build_sankey(df: pd.DataFrame) -> go.Figure:
    """Sankey: Service Type → Priority Level → Top Call Types."""
    # Map priorities to broad levels
    df = df.copy()
    df["priority_level"] = df["priority_description"].apply(_map_priority_level)

    # Limit to top-8 call types for readability
    top_types = df["call_type"].value_counts().head(8).index.tolist()
    df_sankey = df[df["call_type"].isin(top_types)].copy()

    # --- Build node/link lists ---
    services = df_sankey["service_type"].unique().tolist()
    priorities = df_sankey["priority_level"].unique().tolist()
    call_types = top_types

    all_labels = services + priorities + call_types
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    # Service → Priority links
    sp = (df_sankey.groupby(["service_type", "priority_level"])
          .size().reset_index(name="count"))
    # Priority → Call Type links
    pc = (df_sankey.groupby(["priority_level", "call_type"])
          .size().reset_index(name="count"))

    sources, targets, values = [], [], []

    for _, row in sp.iterrows():
        sources.append(label_idx[row["service_type"]])
        targets.append(label_idx[row["priority_level"]])
        values.append(row["count"])

    for _, row in pc.iterrows():
        sources.append(label_idx[row["priority_level"]])
        targets.append(label_idx[row["call_type"]])
        values.append(row["count"])

    # Node colors
    node_colors = []
    for lbl in all_labels:
        if lbl in ("EMS",):
            node_colors.append(COLORS["ems"])
        elif lbl in ("Fire",):
            node_colors.append(COLORS["fire"])
        elif lbl in PRIORITY_COLOR_MAP:
            node_colors.append(PRIORITY_COLOR_MAP[lbl])
        else:
            node_colors.append(COLORS["accent"])

    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=20,
                  label=all_labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values,
                  color="rgba(168, 85, 247, 0.25)"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Triage Flow: Service → Priority → Call Type",
                   x=0.5, font=dict(size=16)),
    )
    return fig


def _build_priority_funnel(df: pd.DataFrame) -> go.Figure:
    """Priority Distribution Funnel chart."""
    df = df.copy()
    df["priority_level"] = df["priority_description"].apply(_map_priority_level)

    # Order levels from most to least severe
    level_order = ["Life Threatening", "ALS", "BLS", "Non-Emergency", "Other"]
    counts = df["priority_level"].value_counts()

    funnel_data = []
    for level in level_order:
        if level in counts.index:
            funnel_data.append({"Level": level, "Count": counts[level]})

    funnel_df = pd.DataFrame(funnel_data)

    fig = px.funnel(
        funnel_df, x="Count", y="Level",
        color="Level",
        color_discrete_map=PRIORITY_COLOR_MAP,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False,
        title=dict(text="Priority Distribution Funnel", x=0.5,
                   font=dict(size=16)),
    )
    return fig


def _build_pipeline_bar(df: pd.DataFrame) -> go.Figure:
    """Service Pipeline Bar: Total → With Coords → Complete Data."""
    total = len(df)
    with_coords = df["longitude"].notna().sum()
    high_completeness = (df["completeness_score"] >= 0.75).sum()

    stages = ["Total Calls", "With Coordinates", "High Completeness (≥75%)"]
    values = [total, with_coords, high_completeness]
    colors = [COLORS["ems"], COLORS["accent"], COLORS["success"]]

    fig = go.Figure(go.Bar(
        x=values,
        y=stages,
        orientation="h",
        marker_color=colors,
        text=[f"{v:,.0f}" for v in values],
        textposition="auto",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Data Quality Pipeline", x=0.5,
                   font=dict(size=16)),
        xaxis_title="Number of Records",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════

layout = dbc.Container([
    html.H2("📊 Overview", className="my-3",
            style={"fontWeight": "700"}),
    html.P("High-level dispatch analytics for Pittsburgh EMS & Fire services.",
           className="text-muted mb-4"),

    # KPI Row (rendered via callback)
    html.Div(id="overview-kpi-row"),

    html.Hr(style={"borderColor": "#333"}),

    # Row 1: Donut + Top-8 Bar
    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-donut"), md=5),
        dbc.Col(dcc.Graph(id="overview-top8-bar"), md=7),
    ], className="mb-4"),

    # Row 2: Stacked Area (full width)
    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-stacked-area"), md=12),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#333"}),

    # Row 3: Sankey (full width)
    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-sankey",
                          style={"height": "500px"}), md=12),
    ], className="mb-4"),

    # Row 4: Priority Funnel + Pipeline Bar
    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-priority-funnel"), md=6),
        dbc.Col(dcc.Graph(id="overview-pipeline-bar"), md=6),
    ], className="mb-4"),

])


# ═══════════════════════════════════════════════════════════════
# Callbacks — all charts react to global filter store
# ═══════════════════════════════════════════════════════════════

@callback(
    Output("overview-kpi-row", "children"),
    Output("overview-donut", "figure"),
    Output("overview-top8-bar", "figure"),
    Output("overview-stacked-area", "figure"),
    Output("overview-sankey", "figure"),
    Output("overview-priority-funnel", "figure"),
    Output("overview-pipeline-bar", "figure"),
    Input("global-filter-store", "data"),
)
def update_overview(filters):
    """Re-render all overview charts when global filters change."""
    df = _apply_filters(DF, filters)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No data matches filters",
                              showarrow=False, font=dict(size=18))]
        )
        return (
            dbc.Alert("No data matches the current filters.", color="warning"),
            empty_fig, empty_fig, empty_fig,
            empty_fig, empty_fig, empty_fig,
        )

    return (
        _build_kpi_row(df),
        _build_donut(df),
        _build_top8_bar(df),
        _build_stacked_area(df),
        _build_sankey(df),
        _build_priority_funnel(df),
        _build_pipeline_bar(df),
    )
