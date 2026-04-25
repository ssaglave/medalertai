"""
pages/overview.py — Overview dashboard page.

Lead: Suvarna (C2) — Phase 4A collaborative page
Phase: 4

Charts:
  1. KPI tile row              — Total calls, EMS count, Fire count, avg/quarter
  2. EMS vs Fire donut         (px.pie)
  3. Top-8 incident bar        (px.bar)
  4. Stacked area by year      (px.area)
  5. Sankey                    (go.Sankey) — Service → Priority → Call Type
  6. Priority Distribution Funnel (px.funnel)
  7. Service Pipeline Bar      (px.bar horizontal)

Data source (built by scripts/build_overview_aggregates.py):
    data/processed/overview_agg.parquet
        per (year, quarter, service, priority_level, call_type):
            call_count, with_coords_count, high_completeness_count
"""

from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, path="/", name="Overview", order=0)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_AGG_PATH = _REPO_ROOT / "data" / "processed" / "overview_agg.parquet"

_AGG_CACHE: pd.DataFrame | None = None


def _load_agg() -> pd.DataFrame:
    global _AGG_CACHE
    if _AGG_CACHE is None:
        if _AGG_PATH.exists():
            _AGG_CACHE = pd.read_parquet(_AGG_PATH)
        else:
            _AGG_CACHE = pd.DataFrame(
                columns=["year", "quarter", "service", "priority_level", "call_type",
                         "call_count", "with_coords_count", "high_completeness_count"]
            )
    return _AGG_CACHE


COLORS = {
    "ems":        "#00d4ff",
    "fire":       "#ff6b35",
    "accent":     "#a855f7",
    "bg_card":    "#1a1a2e",
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

PRIORITY_ORDER = ["Life Threatening", "ALS", "BLS", "Non-Emergency", "Other"]


def _filter(filters: dict) -> pd.DataFrame:
    df = _load_agg()
    if df.empty:
        return df
    filters = filters or {}
    years = filters.get("years") or []
    services = filters.get("services") or []
    call_types = filters.get("call_types") or []

    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["year"].isin(years)
    if services:
        mask &= df["service"].isin(services)
    if call_types:
        mask &= df["call_type"].isin(call_types)
    return df[mask]


def _kpi_card(title: str, value, icon: str, color: str) -> dbc.Col:
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


def _empty_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text="No data matches filters",
                          showarrow=False, font=dict(size=18))]
    )
    return fig


def _build_kpi_row(filters: dict):
    df = _filter(filters)
    if df.empty:
        return dbc.Alert("No data matches the current filters.", color="warning")

    total = int(df["call_count"].sum())
    ems = int(df.loc[df["service"] == "EMS", "call_count"].sum())
    fire = int(df.loc[df["service"] == "Fire", "call_count"].sum())
    n_quarters = df.groupby(["year", "quarter"]).ngroups
    avg_q = total / max(n_quarters, 1)

    return dbc.Row([
        _kpi_card("Total Calls", total, "📞", COLORS["accent"]),
        _kpi_card("EMS Calls", ems, "🚑", COLORS["ems"]),
        _kpi_card("Fire Calls", fire, "🔥", COLORS["fire"]),
        _kpi_card("Avg / Quarter", avg_q, "📊", COLORS["success"]),
    ])


def _build_donut(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()
    counts = (df.groupby("service", observed=True)["call_count"].sum()
              .reset_index())
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
        title=dict(text="EMS vs Fire Distribution", x=0.5, font=dict(size=16)),
    )
    return fig


def _build_top8_bar(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()

    top = (df.groupby("call_type", observed=True)["call_count"].sum()
             .nlargest(8).sort_values(ascending=True).reset_index())
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
        title=dict(text="Top 8 Incident Categories", x=0.5, font=dict(size=16)),
        yaxis_title="",
        xaxis_title="Number of Calls",
    )
    return fig


def _build_stacked_area(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()

    yearly = (df.groupby(["year", "service"], observed=True)["call_count"]
              .sum()
              .reset_index(name="Count"))
    yearly = yearly.rename(columns={"year": "CALL_YEAR", "service": "service_type"})

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
        title=dict(text="Call Volume Over Time", x=0.5, font=dict(size=16)),
        xaxis=dict(dtick=1),
    )
    return fig


def _build_sankey(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()

    top_types = (df.groupby("call_type", observed=True)["call_count"].sum()
                   .nlargest(8).index.tolist())
    df = df[df["call_type"].isin(top_types)]
    if df.empty:
        return _empty_fig()

    services = sorted(df["service"].unique().tolist())
    priorities = [p for p in PRIORITY_ORDER if p in df["priority_level"].unique()]
    call_types = top_types

    all_labels = services + priorities + call_types
    label_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    sp = (df.groupby(["service", "priority_level"], observed=True)["call_count"]
            .sum().reset_index(name="count"))
    pc = (df.groupby(["priority_level", "call_type"], observed=True)["call_count"]
            .sum().reset_index(name="count"))

    sources, targets, values = [], [], []
    for _, row in sp.iterrows():
        sources.append(label_idx[row["service"]])
        targets.append(label_idx[row["priority_level"]])
        values.append(int(row["count"]))
    for _, row in pc.iterrows():
        sources.append(label_idx[row["priority_level"]])
        targets.append(label_idx[row["call_type"]])
        values.append(int(row["count"]))

    node_colors = []
    for lbl in all_labels:
        if lbl == "EMS":
            node_colors.append(COLORS["ems"])
        elif lbl == "Fire":
            node_colors.append(COLORS["fire"])
        elif lbl in PRIORITY_COLOR_MAP:
            node_colors.append(PRIORITY_COLOR_MAP[lbl])
        else:
            node_colors.append(COLORS["accent"])

    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=20, label=all_labels, color=node_colors),
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


def _build_priority_funnel(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()

    counts = df.groupby("priority_level", observed=True)["call_count"].sum()
    funnel_df = pd.DataFrame(
        [{"Level": lvl, "Count": int(counts[lvl])}
         for lvl in PRIORITY_ORDER if lvl in counts.index]
    )
    if funnel_df.empty:
        return _empty_fig()

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
        title=dict(text="Priority Distribution Funnel", x=0.5, font=dict(size=16)),
    )
    return fig


def _build_pipeline_bar(filters: dict) -> go.Figure:
    df = _filter(filters)
    if df.empty:
        return _empty_fig()

    total = int(df["call_count"].sum())
    with_coords = int(df["with_coords_count"].sum())
    high_completeness = int(df["high_completeness_count"].sum())

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
        title=dict(text="Data Quality Pipeline", x=0.5, font=dict(size=16)),
        xaxis_title="Number of Records",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════

layout = dbc.Container([
    html.H2("📊 Overview", className="my-3", style={"fontWeight": "700"}),
    html.P("High-level dispatch analytics for Pittsburgh EMS & Fire services.",
           className="text-muted mb-4"),

    html.Div(id="overview-kpi-row"),

    html.Hr(style={"borderColor": "#333"}),

    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-donut"), md=5),
        dbc.Col(dcc.Graph(id="overview-top8-bar"), md=7),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-stacked-area"), md=12),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#333"}),

    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-sankey",
                          style={"height": "500px"}), md=12),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="overview-priority-funnel"), md=6),
        dbc.Col(dcc.Graph(id="overview-pipeline-bar"), md=6),
    ], className="mb-4"),
])


# ═══════════════════════════════════════════════════════════════
# Callbacks — one per output
# ═══════════════════════════════════════════════════════════════

@callback(Output("overview-kpi-row", "children"),
          Input("global-filter-store", "data"))
def _cb_kpi(filters):
    return _build_kpi_row(filters or {})


@callback(Output("overview-donut", "figure"),
          Input("global-filter-store", "data"))
def _cb_donut(filters):
    return _build_donut(filters or {})


@callback(Output("overview-top8-bar", "figure"),
          Input("global-filter-store", "data"))
def _cb_top8(filters):
    return _build_top8_bar(filters or {})


@callback(Output("overview-stacked-area", "figure"),
          Input("global-filter-store", "data"))
def _cb_area(filters):
    return _build_stacked_area(filters or {})


@callback(Output("overview-sankey", "figure"),
          Input("global-filter-store", "data"))
def _cb_sankey(filters):
    return _build_sankey(filters or {})


@callback(Output("overview-priority-funnel", "figure"),
          Input("global-filter-store", "data"))
def _cb_funnel(filters):
    return _build_priority_funnel(filters or {})


@callback(Output("overview-pipeline-bar", "figure"),
          Input("global-filter-store", "data"))
def _cb_pipeline(filters):
    return _build_pipeline_bar(filters or {})
