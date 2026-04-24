"""
pages/forecast.py — Demand Forecasting dashboard page.

Owner: Deekshitha (C5)
Phase: 4

Features:
  - 4-quarter forecast with go.Scatter uncertainty bands (fill='tonexty')
  - Prophet/LightGBM/Ensemble dcc.RadioItems toggle
  - Wired to global filters (services, call_types, years)

Data source: data/processed/fact_dispatch_clean.parquet
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from prophet import Prophet  # type: ignore
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

try:
    from lightgbm import LGBMRegressor  # type: ignore
    _HAS_LGBM = True
except ImportError:
    LGBMRegressor = None  # type: ignore
    _HAS_LGBM = False

dash.register_page(__name__, path="/forecast", name="Forecast", order=3)

# ── Constants ──
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PARQUET = _REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"
_QUARTER_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
_HORIZON = 4
_PROPHET_WEIGHT = 0.4
_LGBM_WEIGHT = 0.6
_Z_95 = 1.96
_MIN_HISTORY = 6  # quarters required for lag_4 features

# ── Data load (one-shot at module import) ──
try:
    _DF = pd.read_parquet(
        _PARQUET,
        columns=["CALL_YEAR", "CALL_QUARTER", "service_type", "call_type"],
    )
except FileNotFoundError:
    _DF = pd.DataFrame(
        columns=["CALL_YEAR", "CALL_QUARTER", "service_type", "call_type"]
    )


# ── History aggregation ─────────────────────────────────────
def _quarterly_history(
    services: Optional[list[str]] = None,
    call_types: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Aggregate dispatch records into a (ds, y) quarterly time series."""
    df = _DF
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])
    if years:
        df = df[df["CALL_YEAR"].isin(years)]
    if services:
        df = df[df["service_type"].isin(services)]
    if call_types:
        df = df[df["call_type"].isin(call_types)]
    if df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    agg = (
        df.groupby(["CALL_YEAR", "CALL_QUARTER"], observed=True)
        .size()
        .reset_index(name="y")
    )
    agg["ds"] = pd.to_datetime(
        dict(
            year=agg["CALL_YEAR"].astype(int),
            month=agg["CALL_QUARTER"].map(_QUARTER_MONTH).astype(int),
            day=1,
        )
    )
    return agg.sort_values("ds")[["ds", "y"]].reset_index(drop=True)


def _next_quarter_starts(last_ds: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_ds + pd.DateOffset(months=3), periods=n, freq="QS")


def _format_quarter(ds: pd.Timestamp) -> str:
    return f"Q{ds.quarter} {ds.year}"


# ── Forecasters ─────────────────────────────────────────────
def _forecast_prophet(ts: pd.DataFrame) -> pd.DataFrame:
    """Trend + seasonality. Uses Prophet if installed, OLS+Fourier fallback otherwise."""
    if _HAS_PROPHET:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
        )
        model.fit(ts)
        future = model.make_future_dataframe(periods=_HORIZON, freq="QS")
        f = model.predict(future).tail(_HORIZON)
        return f[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)

    # Fallback: linear trend + quarterly Fourier seasonality fit by OLS.
    n = len(ts)
    x = np.arange(n, dtype=float)
    q = ts["ds"].dt.quarter.values
    A = np.column_stack([
        np.ones(n), x,
        np.sin(2 * np.pi * q / 4),
        np.cos(2 * np.pi * q / 4),
    ])
    coef, *_ = np.linalg.lstsq(A, ts["y"].values, rcond=None)
    sigma = float(np.std(ts["y"].values - A @ coef, ddof=1)) if n > 4 else 0.0

    next_ds = _next_quarter_starts(ts["ds"].iloc[-1], _HORIZON)
    nx = np.arange(n, n + _HORIZON, dtype=float)
    nq = next_ds.quarter.to_numpy()
    F = np.column_stack([
        np.ones(_HORIZON), nx,
        np.sin(2 * np.pi * nq / 4),
        np.cos(2 * np.pi * nq / 4),
    ])
    yhat = F @ coef
    return pd.DataFrame({
        "ds": next_ds,
        "yhat": yhat,
        "yhat_lower": yhat - _Z_95 * sigma,
        "yhat_upper": yhat + _Z_95 * sigma,
    })


def _forecast_lightgbm(ts: pd.DataFrame) -> pd.DataFrame:
    """Recursive multi-step gradient-boosted forecast on quarterly lag features."""
    df = ts.copy()
    df["t"] = np.arange(len(df))
    q = df["ds"].dt.quarter
    df["q_sin"] = np.sin(2 * np.pi * q / 4)
    df["q_cos"] = np.cos(2 * np.pi * q / 4)
    for lag in (1, 2, 4):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    train = df.dropna().reset_index(drop=True)
    feats = ["t", "q_sin", "q_cos", "lag_1", "lag_2", "lag_4"]

    if _HAS_LGBM:
        model = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            random_state=42, verbose=-1,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42,
        )
    model.fit(train[feats], train["y"])
    sigma = float(np.std(train["y"].values - model.predict(train[feats]), ddof=1))

    history = list(ts["y"].values)
    next_ds = _next_quarter_starts(ts["ds"].iloc[-1], _HORIZON)
    rows = []
    for i, ds in enumerate(next_ds):
        row = pd.DataFrame([{
            "t": len(ts) + i,
            "q_sin": np.sin(2 * np.pi * ds.quarter / 4),
            "q_cos": np.cos(2 * np.pi * ds.quarter / 4),
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_4": history[-4],
        }])[feats]
        yhat = float(max(0.0, model.predict(row)[0]))
        rows.append({
            "ds": ds, "yhat": yhat,
            "yhat_lower": max(0.0, yhat - _Z_95 * sigma),
            "yhat_upper": yhat + _Z_95 * sigma,
        })
        history.append(yhat)
    return pd.DataFrame(rows)


def _forecast_ensemble(ts: pd.DataFrame) -> pd.DataFrame:
    p = _forecast_prophet(ts).reset_index(drop=True)
    g = _forecast_lightgbm(ts).reset_index(drop=True)
    return pd.DataFrame({
        "ds": p["ds"],
        "yhat":       _PROPHET_WEIGHT * p["yhat"]       + _LGBM_WEIGHT * g["yhat"],
        "yhat_lower": _PROPHET_WEIGHT * p["yhat_lower"] + _LGBM_WEIGHT * g["yhat_lower"],
        "yhat_upper": _PROPHET_WEIGHT * p["yhat_upper"] + _LGBM_WEIGHT * g["yhat_upper"],
    })


_FORECASTERS = {
    "prophet": _forecast_prophet,
    "lightgbm": _forecast_lightgbm,
    "ensemble": _forecast_ensemble,
}


@lru_cache(maxsize=128)
def _cached_forecast(
    model_type: str,
    services_key: Optional[tuple],
    call_types_key: Optional[tuple],
    years_key: Optional[tuple],
) -> Optional[dict]:
    """Compute a real 4-quarter forecast from historical dispatch data."""
    ts = _quarterly_history(
        services=list(services_key) if services_key else None,
        call_types=list(call_types_key) if call_types_key else None,
        years=list(years_key) if years_key else None,
    )
    if len(ts) < _MIN_HISTORY:
        return None
    fn = _FORECASTERS.get(model_type, _forecast_ensemble)
    fcst = fn(ts).reset_index(drop=True)
    return {
        "quarters":         [_format_quarter(d) for d in fcst["ds"]],
        "forecast":         fcst["yhat"].clip(lower=0).round().tolist(),
        "lower_bound":      fcst["yhat_lower"].clip(lower=0).round().tolist(),
        "upper_bound":      fcst["yhat_upper"].clip(lower=0).round().tolist(),
        "model":            model_type,
        "history_quarters": [_format_quarter(d) for d in ts["ds"]],
        "history":          ts["y"].astype(int).tolist(),
    }


# ── Chart builders ───────────────────────────────────────────
def _create_forecast_chart(forecast_data: dict) -> go.Figure:
    """Real history overlay + 4-quarter forecast with uncertainty band."""
    quarters = forecast_data["quarters"]
    forecast = forecast_data["forecast"]
    lower = forecast_data["lower_bound"]
    upper = forecast_data["upper_bound"]
    hist_q = forecast_data.get("history_quarters", [])
    hist_y = forecast_data.get("history", [])

    # Trim history to a recent context window for readability.
    ctx = 12
    hist_q_show = hist_q[-ctx:]
    hist_y_show = hist_y[-ctx:]

    fig = go.Figure()

    # Historical actuals
    if hist_q_show:
        fig.add_trace(go.Scatter(
            x=hist_q_show, y=hist_y_show,
            mode="lines+markers",
            name="History (actual)",
            line=dict(color="rgb(120, 120, 120)", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))

        # Connect the last actual to the first forecast for visual continuity
        fig.add_trace(go.Scatter(
            x=[hist_q_show[-1], quarters[0]],
            y=[hist_y_show[-1], forecast[0]],
            mode="lines",
            line=dict(color="rgba(0, 100, 200, 0.4)", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 95% band
    fig.add_trace(go.Scatter(
        x=quarters, y=upper,
        mode="lines", line_color="rgba(0,0,0,0)",
        showlegend=False, name="Upper",
    ))
    fig.add_trace(go.Scatter(
        x=quarters, y=lower,
        mode="lines", line_color="rgba(0,0,0,0)",
        fill="tonexty", fillcolor="rgba(0, 100, 200, 0.2)",
        name="95% Confidence Interval",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=quarters, y=forecast,
        mode="lines+markers",
        line=dict(color="rgb(0, 100, 200)", width=3),
        marker=dict(size=10),
        name=f"Forecast ({forecast_data['model'].title()})",
        hovertemplate="<b>%{x}</b><br>Expected calls: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Quarterly Demand: History + 4-Quarter Forecast — {forecast_data['model'].title()}",
        xaxis_title="Quarter",
        yaxis_title="Total Calls",
        hovermode="x unified",
        height=450,
        margin=dict(l=50, r=50, t=60, b=50),
        template="plotly_white",
    )
    return fig


def _create_quarterly_stats(forecast_data: dict) -> dbc.Row:
    """4 stat tiles for the forecast quarters."""
    quarters = forecast_data["quarters"]
    forecast = forecast_data["forecast"]
    color_map = ["primary", "info", "success", "warning"]
    cards = []
    for i, (q, val) in enumerate(zip(quarters, forecast)):
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(q, className="card-title text-muted"),
                        html.H3(f"{val:,.0f}", className="card-text fw-bold"),
                        html.Small("calls forecasted", className="text-muted"),
                    ])
                ], className=f"border-start border-{color_map[i]} border-4")
            ], md=3, className="mb-3")
        )
    return dbc.Row(cards)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14))
    fig.update_layout(height=450, template="plotly_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ── Layout ──────────────────────────────────────────────────
_engine_label = (
    f"{'Prophet' if _HAS_PROPHET else 'Trend+Seasonality (OLS)'} · "
    f"{'LightGBM' if _HAS_LGBM else 'HistGradientBoosting'} · Ensemble"
)

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("🔮 Demand Forecast", className="my-3"),
            html.P([
                "4-quarter demand forecast computed from historical dispatch data. ",
                html.Small(f"Engines: {_engine_label}", className="text-muted"),
            ]),
        ], width=12),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Select Forecast Model:", className="fw-bold mb-2"),
            dcc.RadioItems(
                id="model-selector",
                options=[
                    {"label": " 🔵 Prophet (Trend + Seasonality)", "value": "prophet"},
                    {"label": " 🟢 LightGBM (Gradient Boosting on Lags)", "value": "lightgbm"},
                    {"label": " 🟠 Ensemble (40% Prophet + 60% LightGBM)", "value": "ensemble"},
                ],
                value="ensemble",
                inline=False,
                className="mb-3",
            ),
        ], md=12),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([dcc.Graph(id="forecast-chart", style={"height": "450px"})], width=12),
    ], className="mb-4"),

    dbc.Row([dbc.Col([html.H5("Quarterly Forecast Summary", className="mb-3")], width=12)]),
    dbc.Row(id="quarterly-stats", className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P(
                        "Forecast is computed at request time from "
                        "data/processed/fact_dispatch_clean.parquet, aggregated to "
                        "quarterly call counts and filtered by the global filter bar "
                        "(services, call types, years)."
                    ),
                    html.Ul([
                        html.Li(
                            "Prophet: yearly seasonality + trend with 95% uncertainty intervals "
                            "(falls back to OLS trend + quarterly Fourier seasonality if Prophet "
                            "is not installed)."
                        ),
                        html.Li(
                            "LightGBM: recursive multi-step regression on quarterly lag features "
                            "(lag_1, lag_2, lag_4) and quarter sin/cos seasonality "
                            "(falls back to HistGradientBoostingRegressor if LightGBM is not installed)."
                        ),
                        html.Li("Ensemble: weighted average (40% Prophet, 60% LightGBM)."),
                    ]),
                    html.P(
                        "Uncertainty bands are derived from in-sample residual standard deviation "
                        "(±1.96σ ≈ 95%) for the LightGBM and OLS paths; native intervals for Prophet."
                    ),
                    html.P("Target: MAPE < 15% on the held-out tail of the time series."),
                ], title="📖 Model Details & Methodology"),
            ], flush=True),
        ], width=12),
    ]),

    dcc.Store(id="forecast-data-store"),
], fluid=True, className="p-4")


# ── Callbacks ───────────────────────────────────────────────
def _norm_key(values) -> Optional[tuple]:
    """Normalize a list-ish filter value to a hashable tuple (or None)."""
    if not values:
        return None
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(values)


@callback(
    Output("forecast-data-store", "data"),
    Input("model-selector", "value"),
    Input("global-filter-store", "data"),
    prevent_initial_call=False,
)
def update_forecast_data(model_type, global_filters):
    filters = global_filters or {}
    return _cached_forecast(
        model_type or "ensemble",
        _norm_key(filters.get("services")),
        _norm_key(filters.get("call_types")),
        _norm_key(filters.get("years")),
    )


@callback(
    Output("forecast-chart", "figure"),
    Input("forecast-data-store", "data"),
)
def update_forecast_chart(forecast_data):
    if not forecast_data:
        return _empty_figure(
            f"Insufficient history (need ≥{_MIN_HISTORY} quarters). "
            "Broaden the global filters to include more years."
        )
    return _create_forecast_chart(forecast_data)


@callback(
    Output("quarterly-stats", "children"),
    Input("forecast-data-store", "data"),
)
def update_quarterly_stats(forecast_data):
    if not forecast_data:
        return dbc.Alert(
            f"No forecast available — the current filter selection yields "
            f"fewer than {_MIN_HISTORY} quarters of history.",
            color="warning",
        )
    return _create_quarterly_stats(forecast_data)
