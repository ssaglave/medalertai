"""
pages/geography.py — Geographic Analysis dashboard page.

Owner: Sanika (C3)
Phase: 4B

Charts / Components:
  1. KPI row  — Unique CBGs covered, Total Calls w/ coords, Hotspot clusters, Equity score
  2. Map Tab  — Call-density bubble map (go.Scattermapbox, sized by call count)
               overlaid with DBSCAN hotspot cluster markers (distinct colors per cluster)
  3. Equity Tab — Response-equity scatter (city call burden vs avg completeness_score)
                  + City-level summary table

Data sources:
  - data/processed/fact_dispatch_clean.parquet   (main dispatch data)
  - data/processed/dim_cbg_demographics.parquet  (optional demographics)
  - models/artifacts/clustering/hotspots.parquet (optional pre-computed clusters)
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
dash.register_page(__name__, path="/geography", name="Geography", order=2)

# ── Paths ──
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PARQUET    = _REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"
_DEMO_PATH  = _REPO_ROOT / "data" / "processed" / "dim_cbg_demographics.parquet"
_HOTSPOT    = _REPO_ROOT / "models" / "artifacts" / "clustering" / "hotspots.parquet"

# ── Data Loading ──
try:
    DF = pd.read_parquet(_PARQUET)
    # Ensure expected columns exist
    for col in ("CALL_YEAR", "service_type", "call_type",
                "latitude", "longitude", "completeness_score"):
        if col not in DF.columns:
            DF[col] = np.nan
except FileNotFoundError:
    DF = pd.DataFrame()

try:
    DEMO_DF = pd.read_parquet(_DEMO_PATH)
except FileNotFoundError:
    DEMO_DF = pd.DataFrame()

try:
    HOTSPOT_DF = pd.read_parquet(_HOTSPOT)
except FileNotFoundError:
    HOTSPOT_DF = pd.DataFrame()

# Pittsburgh default centre
_PGH_LAT, _PGH_LON = 40.4406, -79.9959

# ── Colour Palette ──
COLORS = {
    "ems":       "#00d4ff",
    "fire":      "#ff6b35",
    "accent":    "#a855f7",
    "bg_card":   "#1a1a2e",
    "bg_card2":  "#16213e",
    "text":      "#e0e0e0",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "hotspot":   "#ff6b35",
}

CLUSTER_PALETTE = [
    "#00d4ff", "#a855f7", "#10b981", "#f59e0b",
    "#ff6b35", "#ec4899", "#3b82f6", "#84cc16",
]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply global filter store selections."""
    if not filters or df.empty:
        return df
    years      = filters.get("years", [])
    services   = filters.get("services", [])
    call_types = filters.get("call_types", [])

    mask = pd.Series(True, index=df.index)
    if years:
        col = "CALL_YEAR" if "CALL_YEAR" in df.columns else "year"
        if col in df.columns:
            mask &= df[col].isin(years)
    if services and "service_type" in df.columns:
        mask &= df["service_type"].isin(services)
    if call_types and "call_type" in df.columns:
        mask &= df["call_type"].isin(call_types)
    return df[mask]


def _geo_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that have valid lat/lon."""
    if df.empty:
        return df
    return df.dropna(subset=["latitude", "longitude"])


def _dbscan_clusters(geo: pd.DataFrame, eps: float = 0.03,
                     min_samples: int = 5) -> pd.Series:
    """Run DBSCAN on lat/lon and return cluster labels."""
    try:
        from sklearn.cluster import DBSCAN
        coords  = geo[["latitude", "longitude"]].values
        labels  = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
        return pd.Series(labels, index=geo.index)
    except Exception:
        return pd.Series(-1, index=geo.index)


def _map_centre(geo: pd.DataFrame):
    if geo.empty:
        return _PGH_LAT, _PGH_LON
    return geo["latitude"].mean(), geo["longitude"].mean()


# ── KPI Card ─────────────────────────────────────────────────────
def _kpi_card(title: str, value, icon: str, color: str,
               subtitle: str = "") -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.Span(icon, style={"fontSize": "1.8rem"}),
                    html.H3(value,
                             className="mb-0 mt-2",
                             style={"color": color, "fontSize": "2rem",
                                    "fontWeight": "700"}),
                    html.P(title, className="mt-1 mb-0",
                           style={"opacity": "0.7", "fontSize": "0.85rem",
                                  "textTransform": "uppercase",
                                  "letterSpacing": "0.8px"}),
                    html.Small(subtitle, className="text-muted") if subtitle else None,
                ], style={"textAlign": "center"})
            ])
        ], className="kpi-card h-100",
           style={"backgroundColor": COLORS["bg_card"],
                  "border": f"1px solid {color}33",
                  "borderRadius": "12px"}),
        xs=12, sm=6, md=3, className="mb-3"
    )


# ═══════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════

def _build_kpi_row(df: pd.DataFrame) -> dbc.Row:
    """4 KPI tiles: CBGs, calls w/ coords, hotspot clusters, equity score."""
    geo = _geo_df(df)

    n_cbg = df["census_block_group"].nunique() if "census_block_group" in df.columns else 0
    n_geo = len(geo)

    # Hotspot clusters (excluding noise label -1)
    if not geo.empty:
        labels = _dbscan_clusters(geo)
        n_clusters = int((labels >= 0).any() and labels[labels >= 0].nunique())
    else:
        n_clusters = 0

    # Equity score — coefficient of variation of per-city completeness (lower = more equitable)
    if not df.empty and "completeness_score" in df.columns and "city_name" in df.columns:
        city_scores = df.groupby("city_name")["completeness_score"].mean()
        equity = 1 - (city_scores.std() / max(city_scores.mean(), 1e-6))
        equity = max(0.0, min(1.0, equity))
        equity_str = f"{equity:.2%}"
    else:
        equity_str = "N/A"

    return dbc.Row([
        _kpi_card("Census Block Groups", f"{n_cbg:,}", "📍",
                  COLORS["accent"], "Unique CBGs w/ calls"),
        _kpi_card("Calls w/ Coordinates", f"{n_geo:,}", "🗺️",
                  COLORS["ems"], "Geo-located records"),
        _kpi_card("Hotspot Clusters", f"{n_clusters:,}", "🔥",
                  COLORS["hotspot"], "DBSCAN clusters"),
        _kpi_card("Geographic Equity", equity_str, "⚖️",
                  COLORS["success"], "Response consistency"),
    ])


# ── Map: density bubble + DBSCAN overlay ─────────────────────────
def _build_density_map(df: pd.DataFrame) -> go.Figure:
    """
    Call-density bubble map.
    Each point = census block group centroid; size = call count.
    DBSCAN hotspot clusters overlaid in bright colours.
    """
    geo = _geo_df(df)

    fig = go.Figure()

    if geo.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            mapbox=dict(style="open-street-map",
                        center=dict(lat=_PGH_LAT, lon=_PGH_LON), zoom=10),
            height=540,
            margin=dict(l=0, r=0, t=40, b=0),
            annotations=[dict(text="No geo-located data available",
                              showarrow=False, font=dict(size=18))],
        )
        return fig

    # ── Layer 1: call density per location ──
    grp_col = "census_block_group" if "census_block_group" in geo.columns else None
    if grp_col:
        agg = (geo.groupby(grp_col)
               .agg(call_count=(grp_col, "count"),
                    lat=("latitude",  "mean"),
                    lon=("longitude", "mean"))
               .reset_index())
    else:
        # Fallback: round lat/lon to 3 dp as proxy location key
        geo = geo.copy()
        geo["_loc"] = (geo["latitude"].round(3).astype(str) + "_" +
                       geo["longitude"].round(3).astype(str))
        agg = (geo.groupby("_loc")
               .agg(call_count=("_loc", "count"),
                    lat=("latitude",  "mean"),
                    lon=("longitude", "mean"))
               .reset_index())

    max_calls = agg["call_count"].max() or 1
    agg["bubble_size"] = 4 + 28 * (agg["call_count"] / max_calls)

    fig.add_trace(go.Scattermapbox(
        lat=agg["lat"],
        lon=agg["lon"],
        mode="markers",
        name="Call Density",
        marker=dict(
            size=agg["bubble_size"],
            color=agg["call_count"],
            colorscale="Viridis",
            cmin=agg["call_count"].min(),
            cmax=agg["call_count"].max(),
            colorbar=dict(title="Calls", x=0.01, xanchor="left",
                          bgcolor="rgba(26,26,46,0.8)",
                          tickfont=dict(color="#e0e0e0")),
            opacity=0.75,
        ),
        hovertemplate=(
            "<b>Calls:</b> %{marker.color:,}<br>"
            "<b>Lat:</b> %{lat:.4f}<br>"
            "<b>Lon:</b> %{lon:.4f}"
            "<extra>Call Density</extra>"
        ),
    ))

    # ── Layer 2: DBSCAN hotspot cluster centres ──
    labels = _dbscan_clusters(geo)
    geo = geo.copy()
    geo["cluster"] = labels.values

    cluster_ids = sorted(geo[geo["cluster"] >= 0]["cluster"].unique())
    for cid in cluster_ids:
        sub = geo[geo["cluster"] == cid]
        colour = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        centre_lat = sub["latitude"].mean()
        centre_lon = sub["longitude"].mean()

        # Convex-hull scatter ring to mark cluster area
        fig.add_trace(go.Scattermapbox(
            lat=sub["latitude"],
            lon=sub["longitude"],
            mode="markers",
            name=f"Cluster {cid}",
            marker=dict(size=6, color=colour, opacity=0.55),
            hovertemplate=(
                f"<b>Cluster {cid}</b><br>"
                f"Points: {len(sub):,}<br>"
                "<b>Lat:</b> %{lat:.4f} <b>Lon:</b> %{lon:.4f}"
                "<extra></extra>"
            ),
            showlegend=True,
        ))
        # Centroid star marker
        fig.add_trace(go.Scattermapbox(
            lat=[centre_lat],
            lon=[centre_lon],
            mode="markers+text",
            name=f"Cluster {cid} Centre",
            marker=dict(size=16, color=colour, opacity=1.0,
                        symbol="circle"),
            text=[f"H{cid}"],
            textfont=dict(size=9, color="white"),
            textposition="middle center",
            hovertemplate=(
                f"<b>Hotspot {cid}</b><br>"
                f"Centre: ({centre_lat:.4f}, {centre_lon:.4f})<br>"
                f"Points: {len(sub):,}"
                "<extra>Hotspot Centre</extra>"
            ),
            showlegend=False,
        ))

    clat, clon = _map_centre(geo)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=clat, lon=clon),
            zoom=10,
        ),
        height=540,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            bgcolor="rgba(26,26,46,0.85)",
            font=dict(color="#e0e0e0", size=11),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        title=dict(text="Call Density & DBSCAN Hotspot Clusters",
                   x=0.5, font=dict(size=16, color="#e0e0e0")),
    )
    return fig


# ── Map: density heatmap variant ─────────────────────────────────
def _build_heatmap(df: pd.DataFrame) -> go.Figure:
    """Density heatmap of all calls."""
    geo = _geo_df(df)
    fig = go.Figure()

    if geo.empty:
        fig.update_layout(template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)",
                          height=300,
                          annotations=[dict(text="No geo data",
                                           showarrow=False)])
        return fig

    fig.add_trace(go.Densitymapbox(
        lat=geo["latitude"],
        lon=geo["longitude"],
        colorscale="Hot",
        radius=14,
        opacity=0.75,
        hovertemplate="<b>Density:</b> %{z:.2f}<extra></extra>",
        name="Call Density",
    ))

    clat, clon = _map_centre(geo)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        mapbox=dict(style="open-street-map",
                    center=dict(lat=clat, lon=clon), zoom=10),
        height=300,
        margin=dict(l=0, r=0, t=35, b=0),
        title=dict(text="Call Density Heatmap",
                   x=0.5, font=dict(size=14, color="#e0e0e0")),
    )
    return fig


# ── Service-type breakdown by city ───────────────────────────────
def _build_city_bar(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: EMS vs Fire call count per city (top 12)."""
    if df.empty or "city_name" not in df.columns:
        return go.Figure()

    top_cities = (df["city_name"].value_counts()
                  .head(12).index.tolist())
    sub = df[df["city_name"].isin(top_cities)]

    agg = (sub.groupby(["city_name", "service_type"])
           .size()
           .reset_index(name="count"))

    fig = px.bar(
        agg, x="city_name", y="count", color="service_type",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
        labels={"city_name": "City", "count": "Calls",
                "service_type": "Service"},
        barmode="stack",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=60, l=10, r=10),
        title=dict(text="EMS vs Fire Calls by City (Top 12)",
                   x=0.5, font=dict(size=16, color="#e0e0e0")),
        xaxis=dict(tickangle=-35),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ── Response Equity scatter ───────────────────────────────────────
def _build_equity_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter: x = call volume per city (proxy for demand / burden),
             y = avg completeness_score (proxy for response data quality / compliance).
    Colour = service_type split.
    """
    if df.empty or "city_name" not in df.columns:
        return go.Figure()

    # Build city-level summary
    agg = (df.groupby(["city_name", "service_type"])
           .agg(call_count=("city_name", "count"),
                avg_completeness=("completeness_score", "mean"))
           .reset_index())
    agg["avg_completeness_pct"] = (agg["avg_completeness"] * 100).round(2)

    fig = px.scatter(
        agg,
        x="call_count",
        y="avg_completeness_pct",
        color="service_type",
        size="call_count",
        size_max=40,
        text="city_name",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
        labels={
            "call_count":           "Call Volume",
            "avg_completeness_pct": "Avg Data Completeness (%)",
            "service_type":         "Service",
            "city_name":            "City",
        },
    )

    # NFPA 90% compliance reference line
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="90% NFPA Target",
                  annotation_position="bottom right")

    fig.update_traces(textposition="top center",
                      textfont=dict(size=10, color="#e0e0e0"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=20, l=10, r=10),
        title=dict(text="Response Equity: Call Burden vs Data Completeness",
                   x=0.5, font=dict(size=16, color="#e0e0e0")),
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(title="Call Volume (log scale)", type="log"),
        yaxis=dict(title="Avg Completeness (%)", range=[0, 105]),
    )
    return fig


def _build_equity_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: avg completeness per city, colour-coded vs 90% target."""
    if df.empty or "city_name" not in df.columns:
        return go.Figure()

    city_comp = (df.groupby("city_name")["completeness_score"]
                 .mean()
                 .reset_index())
    city_comp.columns = ["City", "Score"]
    city_comp["Score_pct"] = (city_comp["Score"] * 100).round(2)
    city_comp = (city_comp
                 .sort_values("Score_pct", ascending=True)
                 .tail(15))

    bar_colors = [COLORS["success"] if s >= 90
                  else (COLORS["warning"] if s >= 70 else COLORS["danger"])
                  for s in city_comp["Score_pct"]]

    fig = go.Figure(go.Bar(
        x=city_comp["Score_pct"],
        y=city_comp["City"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}%" for s in city_comp["Score_pct"]],
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))
    fig.add_vline(x=90, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="90% NFPA Target",
                  annotation_position="top")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Data Completeness by City (Top 15)",
                   x=0.5, font=dict(size=16, color="#e0e0e0")),
        xaxis=dict(title="Avg Completeness %", range=[0, 105]),
        yaxis_title="",
    )
    return fig


def _build_call_type_geo(df: pd.DataFrame) -> go.Figure:
    """Bar: top incident categories with geographic spread (# unique CBGs)."""
    if df.empty or "call_type" not in df.columns:
        return go.Figure()

    cbg_col = "census_block_group" if "census_block_group" in df.columns else None
    if cbg_col:
        spread = (df.groupby("call_type")
                  .agg(total=("call_type", "count"),
                       cbg_spread=(cbg_col, "nunique"))
                  .reset_index()
                  .sort_values("total", ascending=False)
                  .head(12))
    else:
        spread = (df["call_type"].value_counts()
                  .head(12)
                  .reset_index())
        spread.columns = ["call_type", "total"]
        spread["cbg_spread"] = 0

    spread = spread.sort_values("cbg_spread", ascending=True)

    fig = go.Figure(go.Bar(
        x=spread["cbg_spread"],
        y=spread["call_type"],
        orientation="h",
        marker=dict(
            color=spread["cbg_spread"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="CBGs", tickfont=dict(color="#e0e0e0")),
        ),
        text=[f"{v:,} CBGs" for v in spread["cbg_spread"]],
        textposition="auto",
        textfont=dict(color="white", size=10),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Geographic Spread per Incident Type (# Unique CBGs)",
                   x=0.5, font=dict(size=16, color="#e0e0e0")),
        xaxis_title="Unique Census Block Groups",
        yaxis_title="",
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Empty-figure helper
# ═══════════════════════════════════════════════════════════════
def _empty_fig(msg: str = "No data matches filters") -> go.Figure:
    f = go.Figure()
    f.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=18, color="#e0e0e0"))],
        height=300,
    )
    return f


# ═══════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════

layout = dbc.Container([

    html.H2("🗺️ Geographic Analysis", className="my-3",
            style={"fontWeight": "700"}),
    html.P("Call-density maps, DBSCAN hotspot clusters, "
           "and response-equity analysis across Pittsburgh.",
           className="text-muted mb-4"),

    # ── KPI Row (via callback) ──
    html.Div(id="geo-kpi-row"),

    html.Hr(style={"borderColor": "#333"}),

    # ── Tabs: Map View / Response Equity ──
    dbc.Tabs(id="geo-tabs", active_tab="tab-map", children=[

        # ── Tab 1: Map ──
        dbc.Tab(label="📍 Hotspot Map", tab_id="tab-map", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(id="geo-density-map",
                                  config={"scrollZoom": True}),
                        className="mt-3",
                    ),
                    # Legend / info banner
                    dbc.Alert([
                        html.Strong("Legend: "),
                        "Bubble size & colour = call count per block-group centroid. ",
                        "Coloured rings = DBSCAN hotspot clusters (eps≈0.03°, min_samples=5). ",
                        "⭐ Labelled stars mark cluster centres (H0, H1, …).",
                    ], color="info", className="mt-2 mb-0 py-2",
                       style={"fontSize": "0.82rem"}),
                ], md=12),
            ]),
            html.Hr(style={"borderColor": "#333"}),
            dbc.Row([
                dbc.Col(dcc.Graph(id="geo-heatmap"), md=6),
                dbc.Col(dcc.Graph(id="geo-city-bar"), md=6),
            ], className="mt-3 mb-4"),
        ]),

        # ── Tab 2: Response Equity ──
        dbc.Tab(label="⚖️ Response Equity", tab_id="tab-equity", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="geo-equity-scatter"), md=12),
            ], className="mt-3"),
            html.Hr(style={"borderColor": "#333"}),
            dbc.Row([
                dbc.Col(dcc.Graph(id="geo-equity-bar"),    md=6),
                dbc.Col(dcc.Graph(id="geo-call-type-geo"), md=6),
            ], className="mb-4"),
            dbc.Alert([
                html.Strong("Note: "),
                "Data Completeness is used as a proxy for response data quality. "
                "Cities below the 90% NFPA target (red dashed line) may indicate "
                "under-resourced areas or reporting gaps requiring attention.",
            ], color="warning", className="mb-3 py-2",
               style={"fontSize": "0.82rem"}),
        ]),

    ]),

], fluid=True)


# ═══════════════════════════════════════════════════════════════
# Callbacks — react to global filter store
# ═══════════════════════════════════════════════════════════════

@callback(
    Output("geo-kpi-row",        "children"),
    Output("geo-density-map",    "figure"),
    Output("geo-heatmap",        "figure"),
    Output("geo-city-bar",       "figure"),
    Output("geo-equity-scatter", "figure"),
    Output("geo-equity-bar",     "figure"),
    Output("geo-call-type-geo",  "figure"),
    Input("global-filter-store", "data"),
)
def update_geography(filters):
    """Re-render all geography charts when global filters change."""
    df = _apply_filters(DF, filters)

    if df.empty:
        emp = _empty_fig()
        return (
            dbc.Alert("No data matches the current filters.", color="warning"),
            emp, emp, emp, emp, emp, emp,
        )

    return (
        _build_kpi_row(df),
        _build_density_map(df),
        _build_heatmap(df),
        _build_city_bar(df),
        _build_equity_scatter(df),
        _build_equity_bar(df),
        _build_call_type_geo(df),
    )
