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

Data sources (built by scripts/build_geography_aggregates.py):
  - data/processed/geo_density_agg.parquet
      per (year, service, call_type, census_block_group): call_count + lat/lon centroid
  - data/processed/geo_city_agg.parquet
      per (year, service, city_name): call_count + completeness sum/count
  - data/processed/geo_call_type_cbg_agg.parquet
      per (year, service, call_type): total_calls + unique_cbg_count
"""

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
_DENSITY_AGG = _REPO_ROOT / "data" / "processed" / "geo_density_agg.parquet"
_CITY_AGG = _REPO_ROOT / "data" / "processed" / "geo_city_agg.parquet"
_CALL_TYPE_AGG = _REPO_ROOT / "data" / "processed" / "geo_call_type_cbg_agg.parquet"

# ── Cached aggregate frames ──
_density_cache: pd.DataFrame | None = None
_city_cache: pd.DataFrame | None = None
_call_type_cache: pd.DataFrame | None = None


def _load_density_agg() -> pd.DataFrame:
    global _density_cache
    if _density_cache is None:
        if _DENSITY_AGG.exists():
            _density_cache = pd.read_parquet(_DENSITY_AGG)
        else:
            _density_cache = pd.DataFrame(
                columns=["year", "service", "call_type", "census_block_group",
                         "call_count", "latitude", "longitude"]
            )
    return _density_cache


def _load_city_agg() -> pd.DataFrame:
    global _city_cache
    if _city_cache is None:
        if _CITY_AGG.exists():
            _city_cache = pd.read_parquet(_CITY_AGG)
        else:
            _city_cache = pd.DataFrame(
                columns=["year", "service", "city_name", "call_count",
                         "completeness_sum", "completeness_count"]
            )
    return _city_cache


def _load_call_type_agg() -> pd.DataFrame:
    global _call_type_cache
    if _call_type_cache is None:
        if _CALL_TYPE_AGG.exists():
            _call_type_cache = pd.read_parquet(_CALL_TYPE_AGG)
        else:
            _call_type_cache = pd.DataFrame(
                columns=["year", "service", "call_type", "total_calls", "unique_cbg_count"]
            )
    return _call_type_cache


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
# Filter helpers
# ═══════════════════════════════════════════════════════════════

def _mask_for(df: pd.DataFrame, filters: dict, with_call_type: bool) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    filters = filters or {}
    years = filters.get("years") or []
    services = filters.get("services") or []
    call_types = filters.get("call_types") or []

    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["year"].isin(years)
    if services and "service" in df.columns:
        mask &= df["service"].isin(services)
    if with_call_type and call_types and "call_type" in df.columns:
        mask &= df["call_type"].isin(call_types)
    return mask


def _filtered_density(filters: dict) -> pd.DataFrame:
    df = _load_density_agg()
    if df.empty:
        return df
    return df[_mask_for(df, filters, with_call_type=True)]


def _cbg_centroids(filtered_density: pd.DataFrame) -> pd.DataFrame:
    """Roll up the (year,service,call_type,CBG) density to per-CBG totals."""
    if filtered_density.empty:
        return filtered_density
    return (
        filtered_density.groupby("census_block_group", observed=True)
        .agg(
            call_count=("call_count", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )


def _filtered_city(filters: dict) -> pd.DataFrame:
    df = _load_city_agg()
    if df.empty:
        return df
    # call_types filter is not honoured on city aggregate (no call_type dimension);
    # the filter still applies on the density side.
    return df[_mask_for(df, filters, with_call_type=False)]


def _filtered_call_type(filters: dict) -> pd.DataFrame:
    df = _load_call_type_agg()
    if df.empty:
        return df
    return df[_mask_for(df, filters, with_call_type=True)]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _dbscan_clusters(geo: pd.DataFrame, eps: float = 0.03,
                     min_samples: int = 5) -> pd.Series:
    """DBSCAN on CBG centroids (≤ ~1100 points). Returns cluster labels per row."""
    if geo.empty:
        return pd.Series([], dtype=int)
    try:
        from sklearn.cluster import DBSCAN
        coords = geo[["latitude", "longitude"]].values
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
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

def _build_kpi_row(filters: dict) -> dbc.Row:
    density = _filtered_density(filters)
    city = _filtered_city(filters)
    centroids = _cbg_centroids(density)

    n_cbg = len(centroids)
    n_geo = int(density["call_count"].sum()) if not density.empty else 0

    if not centroids.empty:
        labels = _dbscan_clusters(centroids)
        n_clusters = int(labels[labels >= 0].nunique())
    else:
        n_clusters = 0

    if not city.empty and city["completeness_count"].sum() > 0:
        per_city = city.groupby("city_name").agg(
            comp_sum=("completeness_sum", "sum"),
            comp_n=("completeness_count", "sum"),
        )
        per_city = per_city[per_city["comp_n"] > 0]
        per_city["mean"] = per_city["comp_sum"] / per_city["comp_n"]
        if len(per_city) > 1 and per_city["mean"].mean() > 0:
            equity = 1 - (per_city["mean"].std() / per_city["mean"].mean())
            equity = max(0.0, min(1.0, equity))
            equity_str = f"{equity:.2%}"
        else:
            equity_str = "N/A"
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
def _build_density_map(filters: dict) -> go.Figure:
    density = _filtered_density(filters)
    geo = _cbg_centroids(density)

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

    max_calls = geo["call_count"].max() or 1
    geo = geo.copy()
    geo["bubble_size"] = 4 + 28 * (geo["call_count"] / max_calls)

    fig.add_trace(go.Scattermapbox(
        lat=geo["latitude"],
        lon=geo["longitude"],
        mode="markers",
        name="Call Density",
        marker=dict(
            size=geo["bubble_size"],
            color=geo["call_count"],
            colorscale="Viridis",
            cmin=geo["call_count"].min(),
            cmax=geo["call_count"].max(),
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

    labels = _dbscan_clusters(geo)
    geo = geo.assign(cluster=labels.values if not labels.empty else -1)

    cluster_ids = sorted(geo[geo["cluster"] >= 0]["cluster"].unique())
    for cid in cluster_ids:
        sub = geo[geo["cluster"] == cid]
        colour = CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]
        centre_lat = sub["latitude"].mean()
        centre_lon = sub["longitude"].mean()

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
def _build_heatmap(filters: dict) -> go.Figure:
    density = _filtered_density(filters)
    geo = _cbg_centroids(density)
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
        z=geo["call_count"],
        colorscale="Hot",
        radius=14,
        opacity=0.75,
        hovertemplate="<b>Density:</b> %{z:.0f}<extra></extra>",
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
def _build_city_bar(filters: dict) -> go.Figure:
    df = _filtered_city(filters)
    if df.empty:
        return _empty_fig()

    by_city = (df.groupby("city_name")["call_count"].sum()
               .sort_values(ascending=False)
               .head(12))
    top_cities = by_city.index.tolist()
    sub = df[df["city_name"].isin(top_cities)]

    agg = (sub.groupby(["city_name", "service"])["call_count"]
           .sum()
           .reset_index(name="count"))

    fig = px.bar(
        agg, x="city_name", y="count", color="service",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
        labels={"city_name": "City", "count": "Calls",
                "service": "Service"},
        barmode="stack",
        category_orders={"city_name": top_cities},
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
def _build_equity_scatter(filters: dict) -> go.Figure:
    df = _filtered_city(filters)
    if df.empty:
        return _empty_fig()

    agg = (df.groupby(["city_name", "service"])
           .agg(call_count=("call_count", "sum"),
                comp_sum=("completeness_sum", "sum"),
                comp_n=("completeness_count", "sum"))
           .reset_index())
    agg = agg[agg["comp_n"] > 0].copy()
    if agg.empty:
        return _empty_fig()
    agg["avg_completeness"] = agg["comp_sum"] / agg["comp_n"]
    agg["avg_completeness_pct"] = (agg["avg_completeness"] * 100).round(2)

    fig = px.scatter(
        agg,
        x="call_count",
        y="avg_completeness_pct",
        color="service",
        size="call_count",
        size_max=40,
        text="city_name",
        color_discrete_map={"EMS": COLORS["ems"], "Fire": COLORS["fire"]},
        labels={
            "call_count":           "Call Volume",
            "avg_completeness_pct": "Avg Data Completeness (%)",
            "service":              "Service",
            "city_name":            "City",
        },
    )

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


def _build_equity_bar(filters: dict) -> go.Figure:
    df = _filtered_city(filters)
    if df.empty:
        return _empty_fig()

    per_city = (df.groupby("city_name")
                .agg(comp_sum=("completeness_sum", "sum"),
                     comp_n=("completeness_count", "sum"))
                .reset_index())
    per_city = per_city[per_city["comp_n"] > 0].copy()
    if per_city.empty:
        return _empty_fig()
    per_city["Score"] = per_city["comp_sum"] / per_city["comp_n"]
    per_city["Score_pct"] = (per_city["Score"] * 100).round(2)
    per_city = per_city.rename(columns={"city_name": "City"})
    per_city = (per_city[["City", "Score_pct"]]
                .sort_values("Score_pct", ascending=True)
                .tail(15))

    bar_colors = [COLORS["success"] if s >= 90
                  else (COLORS["warning"] if s >= 70 else COLORS["danger"])
                  for s in per_city["Score_pct"]]

    fig = go.Figure(go.Bar(
        x=per_city["Score_pct"],
        y=per_city["City"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.1f}%" for s in per_city["Score_pct"]],
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


def _build_call_type_geo(filters: dict) -> go.Figure:
    df = _filtered_call_type(filters)
    if df.empty:
        return _empty_fig()

    spread = (df.groupby("call_type")
              .agg(total=("total_calls", "sum"),
                   cbg_spread=("unique_cbg_count", "max"))
              .reset_index()
              .sort_values("total", ascending=False)
              .head(12))
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

    html.Div(id="geo-kpi-row"),

    html.Hr(style={"borderColor": "#333"}),

    dbc.Tabs(id="geo-tabs", active_tab="tab-map", children=[

        dbc.Tab(label="📍 Hotspot Map", tab_id="tab-map", children=[
            dbc.Row([
                dbc.Col([
                    html.Div(
                        dcc.Graph(id="geo-density-map",
                                  config={"scrollZoom": True}),
                        className="mt-3",
                    ),
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
# Callbacks — one per output, wired to the global filter store
# ═══════════════════════════════════════════════════════════════

@callback(Output("geo-kpi-row", "children"), Input("global-filter-store", "data"))
def _cb_kpi_row(filters):
    return _build_kpi_row(filters or {})


@callback(Output("geo-density-map", "figure"), Input("global-filter-store", "data"))
def _cb_density_map(filters):
    return _build_density_map(filters or {})


@callback(Output("geo-heatmap", "figure"), Input("global-filter-store", "data"))
def _cb_heatmap(filters):
    return _build_heatmap(filters or {})


@callback(Output("geo-city-bar", "figure"), Input("global-filter-store", "data"))
def _cb_city_bar(filters):
    return _build_city_bar(filters or {})


@callback(Output("geo-equity-scatter", "figure"), Input("global-filter-store", "data"))
def _cb_equity_scatter(filters):
    return _build_equity_scatter(filters or {})


@callback(Output("geo-equity-bar", "figure"), Input("global-filter-store", "data"))
def _cb_equity_bar(filters):
    return _build_equity_bar(filters or {})


@callback(Output("geo-call-type-geo", "figure"), Input("global-filter-store", "data"))
def _cb_call_type_geo(filters):
    return _build_call_type_geo(filters or {})
