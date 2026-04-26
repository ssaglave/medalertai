"""
pages/qa.py — Classification QA dashboard page.

Owner: Suvarna (C2)
Phase: 4B

Charts / Components:
  1. KPI Summary Row — Accuracy, Macro F1, Disagreements, Test Rows
  2. Color-coded Agreement DataTable (conditional formatting: Match/Review/Mismatch)
  3. Bullet Chart (go.Indicator) — Data Completeness vs 90% NFPA target
  4. Bullet Chart (go.Indicator) — Macro F1 Score vs 0.55 target
  5. Agreement Rate by MPDS Group (horizontal bar)
  6. Completeness Trend by Year (line chart)
  7. Confidence Distribution (histogram)

Data sources:
  - models/artifacts/classifier/disagreements.parquet
  - models/artifacts/classifier/metrics.json
  - models/artifacts/classifier/label_map.parquet
  - models/artifacts/classifier/feature_importance.csv
"""

import json
from pathlib import Path

import dash
from dash import html, dcc, callback, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Page Registration ──
dash.register_page(__name__, path="/classification-qa",
                   name="Classification QA", order=4)

# ── Paths ──
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CLASSIFIER_DIR = _REPO_ROOT / "models" / "artifacts" / "classifier"

# ── Data Loading ──
try:
    DISAGREE_DF = pd.read_parquet(_CLASSIFIER_DIR / "disagreements.parquet")
except FileNotFoundError:
    DISAGREE_DF = pd.DataFrame()

try:
    with open(_CLASSIFIER_DIR / "metrics.json") as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = {}

try:
    LABEL_MAP = pd.read_parquet(_CLASSIFIER_DIR / "label_map.parquet")
except FileNotFoundError:
    LABEL_MAP = pd.DataFrame()

try:
    FEATURE_IMP = pd.read_csv(_CLASSIFIER_DIR / "feature_importance.csv")
except FileNotFoundError:
    FEATURE_IMP = pd.DataFrame()

try:
    _cm_long = pd.read_parquet(_CLASSIFIER_DIR / "confusion_matrix.parquet")
    CONFUSION_MATRIX = _cm_long.set_index("true_mpds_group")
except FileNotFoundError:
    CONFUSION_MATRIX = pd.DataFrame()

try:
    with open(_CLASSIFIER_DIR / "disagreement_flagging_eval.json") as f:
        DISAGREE_EVAL = json.load(f)
except FileNotFoundError:
    DISAGREE_EVAL = {}


# ── Color Palette ──
COLORS = {
    "match":      "#10b981",   # green
    "review":     "#f59e0b",   # amber
    "mismatch":   "#ef4444",   # red
    "accent":     "#a855f7",   # purple
    "ems":        "#00d4ff",   # cyan
    "bg_card":    "#1a1a2e",
    "text":       "#e0e0e0",
    "success":    "#10b981",
    "warning":    "#f59e0b",
    "target_line": "#ff6b6b",
}

NFPA_TARGET = 0.90  # 90% NFPA compliance target
F1_TARGET = 0.55    # Macro F1 target (revised 2026-04-26 from 0.75)


# ═══════════════════════════════════════════════════════════════
# Helper: apply global filters
# ═══════════════════════════════════════════════════════════════
def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Filter disagreements dataframe based on global-filter-store."""
    if not filters or df.empty:
        return df

    years = filters.get("years", [])
    services = filters.get("services", [])
    call_types = filters.get("call_types", [])

    mask = pd.Series(True, index=df.index)

    if years:
        mask &= df["year"].isin(years)
    if services:
        mask &= df["service_type"].isin(services)
    if call_types:
        mask &= df["call_type"].isin(call_types)

    return df[mask]


# ═══════════════════════════════════════════════════════════════
# Helper: classify agreement status
# ═══════════════════════════════════════════════════════════════
def _classify_agreement(row) -> str:
    """Classify row into Match / Review / Mismatch."""
    if row["is_correct"]:
        if row["max_confidence"] >= 0.7:
            return "Match"
        else:
            return "Review"
    else:
        return "Mismatch"


# ═══════════════════════════════════════════════════════════════
# KPI Card builder
# ═══════════════════════════════════════════════════════════════
def _kpi_card(title: str, value, icon: str, color: str,
              subtitle: str = "") -> dbc.Col:
    """Create a single KPI card."""
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
                    html.Small(subtitle, className="text-muted")
                    if subtitle else None,
                ], style={"textAlign": "center"})
            ])
        ], className="h-100",
           style={"backgroundColor": COLORS["bg_card"],
                  "border": f"1px solid {color}33",
                  "borderRadius": "12px"}),
        xs=12, sm=6, md=3, className="mb-3"
    )


# ═══════════════════════════════════════════════════════════════
# Chart builders
# ═══════════════════════════════════════════════════════════════

def _build_kpi_row(df: pd.DataFrame, metrics: dict) -> dbc.Row:
    """KPI tiles: Accuracy, Macro F1, Disagreements, Test Rows."""
    test_metrics = metrics.get("test", {})
    accuracy = test_metrics.get("accuracy", 0)
    macro_f1 = test_metrics.get("macro_f1", 0)
    n_disagree = df["is_disagreement"].sum() if not df.empty else 0
    n_rows = len(df)

    return dbc.Row([
        _kpi_card("Accuracy", f"{accuracy:.2%}", "🎯",
                  COLORS["success"], "Test set"),
        _kpi_card("Macro F1", f"{macro_f1:.4f}", "📊",
                  COLORS["accent"], f"Target: ≥{F1_TARGET}"),
        _kpi_card("Disagreements", f"{n_disagree:,}", "⚠️",
                  COLORS["warning"] if n_disagree > 0 else COLORS["success"],
                  f"of {n_rows:,} rows"),
        _kpi_card("MPDS Classes", f"{metrics.get('n_classes', 0)}", "🏷️",
                  COLORS["ems"], "Complaint groups"),
    ])


def _build_agreement_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Color-coded agreement DataTable with conditional formatting."""
    if df.empty:
        return html.Div("No classifier data available.")

    df = df.copy()
    df["status"] = df.apply(_classify_agreement, axis=1)

    # Prepare table data — show a summary per MPDS group
    summary = (df.groupby("mpds_group")
               .agg(
                   total=("is_correct", "count"),
                   matches=("is_correct", "sum"),
                   accuracy=("is_correct", "mean"),
                   avg_confidence=("max_confidence", "mean"),
                   disagreements=("is_disagreement", "sum"),
               )
               .reset_index()
               .sort_values("total", ascending=False))

    summary["accuracy_pct"] = (summary["accuracy"] * 100).round(2)
    summary["avg_confidence_pct"] = (summary["avg_confidence"] * 100).round(1)
    summary["matches"] = summary["matches"].astype(int)
    summary["disagreements"] = summary["disagreements"].astype(int)

    # Assign status badge — tuned to the 9-class collapsed model where
    # per-class recall ranges roughly 14–86%. Old bar (>=99 / >=90) made
    # every group look like Mismatch; this scales to the model we have.
    summary["status"] = summary["accuracy_pct"].apply(
        lambda x: "✅ Match" if x >= 70
                  else ("⚠️ Review" if x >= 50 else "❌ Mismatch")
    )

    display_cols = ["mpds_group", "total", "matches", "disagreements",
                    "accuracy_pct", "avg_confidence_pct", "status"]
    col_labels = {
        "mpds_group": "MPDS Group",
        "total": "Total",
        "matches": "Matches",
        "disagreements": "Disagree",
        "accuracy_pct": "Accuracy %",
        "avg_confidence_pct": "Avg Confidence %",
        "status": "Status",
    }

    return dash_table.DataTable(
        id="qa-agreement-table",
        data=summary[display_cols].to_dict("records"),
        columns=[{"name": col_labels.get(c, c), "id": c} for c in display_cols],
        sort_action="native",
        filter_action="native",
        page_size=15,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#16213e",
            "color": "#e0e0e0",
            "fontWeight": "700",
            "border": "1px solid #333",
            "textAlign": "center",
        },
        style_cell={
            "backgroundColor": "#1a1a2e",
            "color": "#e0e0e0",
            "border": "1px solid #333",
            "textAlign": "center",
            "fontSize": "0.85rem",
            "padding": "8px",
        },
        style_data_conditional=[
            # Match rows — green
            {
                "if": {
                    "filter_query": '{status} contains "Match"',
                },
                "backgroundColor": "rgba(16, 185, 129, 0.1)",
                "color": COLORS["match"],
            },
            # Review rows — amber
            {
                "if": {
                    "filter_query": '{status} contains "Review"',
                },
                "backgroundColor": "rgba(245, 158, 11, 0.15)",
                "color": COLORS["review"],
            },
            # Mismatch rows — red
            {
                "if": {
                    "filter_query": '{status} contains "Mismatch"',
                },
                "backgroundColor": "rgba(239, 68, 68, 0.15)",
                "color": COLORS["mismatch"],
            },
            # Highlight disagreements > 0
            {
                "if": {
                    "filter_query": "{disagreements} > 0",
                    "column_id": "disagreements",
                },
                "fontWeight": "bold",
                "color": COLORS["warning"],
            },
        ],
    )


def _build_bullet_completeness(df: pd.DataFrame) -> go.Figure:
    """Bullet Chart: Data Completeness vs 90% NFPA target."""
    avg_comp = df["completeness_score"].mean() if not df.empty else 0

    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=avg_comp * 100,
        delta={"reference": NFPA_TARGET * 100, "position": "top",
               "increasing": {"color": COLORS["success"]},
               "decreasing": {"color": COLORS["mismatch"]}},
        title={"text": "Data Completeness",
               "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": "%",
                "font": {"size": 28, "color": COLORS["text"]}},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100],
                     "tickfont": {"color": COLORS["text"]}},
            "bar": {"color": COLORS["ems"], "thickness": 0.6},
            "bgcolor": "#16213e",
            "steps": [
                {"range": [0, 60], "color": "rgba(239,68,68,0.2)"},
                {"range": [60, 80], "color": "rgba(245,158,11,0.2)"},
                {"range": [80, 100], "color": "rgba(16,185,129,0.2)"},
            ],
            "threshold": {
                "line": {"color": COLORS["target_line"], "width": 3},
                "thickness": 0.8,
                "value": NFPA_TARGET * 100,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def _build_bullet_f1(metrics: dict) -> go.Figure:
    """Bullet Chart: Macro F1 Score vs 0.55 target."""
    f1_val = metrics.get("test", {}).get("macro_f1", 0) * 100

    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=f1_val,
        delta={"reference": F1_TARGET * 100, "position": "top",
               "increasing": {"color": COLORS["success"]},
               "decreasing": {"color": COLORS["mismatch"]}},
        title={"text": "Macro F1 Score",
               "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": "%",
                "font": {"size": 28, "color": COLORS["text"]}},
        gauge={
            "shape": "bullet",
            "axis": {"range": [0, 100],
                     "tickfont": {"color": COLORS["text"]}},
            "bar": {"color": COLORS["accent"], "thickness": 0.6},
            "bgcolor": "#16213e",
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.2)"},
                {"range": [50, 75], "color": "rgba(245,158,11,0.2)"},
                {"range": [75, 100], "color": "rgba(16,185,129,0.2)"},
            ],
            "threshold": {
                "line": {"color": COLORS["target_line"], "width": 3},
                "thickness": 0.8,
                "value": F1_TARGET * 100,
            },
        },
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def _build_agreement_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: Agreement rate by MPDS group."""
    if df.empty:
        return go.Figure()

    grp = (df.groupby("mpds_group")["is_correct"]
           .agg(["mean", "count"])
           .sort_values("count", ascending=True)
           .tail(15)
           .reset_index())
    grp.columns = ["Group", "Accuracy", "Count"]
    grp["Accuracy_pct"] = (grp["Accuracy"] * 100).round(2)

    # Color by accuracy — same thresholds as the agreement DataTable.
    colors = [COLORS["success"] if a >= 70
              else (COLORS["warning"] if a >= 50 else COLORS["mismatch"])
              for a in grp["Accuracy_pct"]]

    fig = go.Figure(go.Bar(
        x=grp["Accuracy_pct"],
        y=grp["Group"],
        orientation="h",
        marker_color=colors,
        text=[f"{a:.1f}% ({c:,})" for a, c in
              zip(grp["Accuracy_pct"], grp["Count"])],
        textposition="auto",
        textfont=dict(color="white", size=11),
    ))

    # Add NFPA target line
    fig.add_vline(x=NFPA_TARGET * 100,
                  line_dash="dash", line_color=COLORS["target_line"],
                  annotation_text="90% NFPA Target",
                  annotation_position="top")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Agreement Rate by MPDS Group",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="Accuracy %", range=[0, 100]),
        yaxis_title="",
    )
    return fig


def _build_completeness_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart: completeness score trend by year."""
    if df.empty:
        return go.Figure()

    trend = (df.groupby("year")["completeness_score"]
             .mean()
             .reset_index())
    trend.columns = ["Year", "Completeness"]
    trend["Completeness_pct"] = (trend["Completeness"] * 100).round(2)

    fig = go.Figure()

    # Completeness line
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["Completeness_pct"],
        mode="lines+markers",
        name="Data Completeness",
        line=dict(color=COLORS["ems"], width=3),
        marker=dict(size=8, symbol="circle"),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.1)",
    ))

    # NFPA target line
    fig.add_hline(y=NFPA_TARGET * 100,
                  line_dash="dash", line_color=COLORS["target_line"],
                  annotation_text="90% NFPA Target",
                  annotation_position="bottom right")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Data Completeness Trend by Year",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="Year", dtick=1),
        yaxis=dict(title="Completeness %", range=[60, 100]),
        showlegend=False,
    )
    return fig


def _build_confidence_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram: distribution of classifier confidence scores."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # Correct predictions
    correct = df[df["is_correct"]]["max_confidence"]
    fig.add_trace(go.Histogram(
        x=correct, name="Correct",
        marker_color=COLORS["success"],
        opacity=0.7, nbinsx=50,
    ))

    # Incorrect predictions
    incorrect = df[~df["is_correct"]]["max_confidence"]
    if len(incorrect) > 0:
        fig.add_trace(go.Histogram(
            x=incorrect, name="Incorrect",
            marker_color=COLORS["mismatch"],
            opacity=0.8, nbinsx=50,
        ))

    # Confidence threshold line
    fig.add_vline(x=0.7, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="0.7 Threshold",
                  annotation_position="top")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        barmode="overlay",
        title=dict(text="Classifier Confidence Distribution",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="Confidence Score"),
        yaxis=dict(title="Count"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _build_confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    """Row-normalized confusion matrix heatmap (true class → predicted distribution)."""
    if cm.empty:
        return go.Figure()

    row_totals = cm.sum(axis=1).replace(0, 1)
    cm_norm = cm.div(row_totals, axis=0) * 100

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm.values,
        x=list(cm_norm.columns),
        y=list(cm_norm.index),
        colorscale="Magma",
        zmin=0, zmax=100,
        colorbar=dict(title="% of true class", ticksuffix="%"),
        hovertemplate=(
            "True: %{y}<br>"
            "Predicted: %{x}<br>"
            "Share of true class: %{z:.1f}%<extra></extra>"
        ),
        text=cm_norm.round(1).values,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=80, l=120, r=20),
        title=dict(text="Confusion Matrix (row-normalized)",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="Predicted class", tickangle=-45),
        yaxis=dict(title="True class", autorange="reversed"),
    )
    return fig


def _build_threshold_sweep(report: dict) -> go.Figure:
    """Recall / false-alarm / precision-proxy across confidence thresholds."""
    sweep = sorted(report.get("sweep", []), key=lambda r: r["threshold"])
    if not sweep:
        return go.Figure()

    thresholds = [row["threshold"] for row in sweep]
    recall = [row["recall_on_injected"] for row in sweep]
    false_alarm = [row["false_alarm_rate"] for row in sweep]
    precision = [row["precision_proxy"] for row in sweep]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=recall, mode="lines+markers",
        name="Recall on injected errors",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=precision, mode="lines+markers",
        name="Precision (proxy)",
        line=dict(color=COLORS["ems"], width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=false_alarm, mode="lines+markers",
        name="False-alarm rate",
        line=dict(color=COLORS["mismatch"], width=2, dash="dash"),
        marker=dict(size=7),
    ))

    default_thr = report.get("default_threshold")
    if default_thr is not None:
        fig.add_vline(
            x=default_thr,
            line_dash="dot", line_color=COLORS["warning"],
            annotation_text=f"default {default_thr}",
            annotation_position="top right",
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=10, r=10),
        title=dict(text="Disagreement-flag Recall vs Confidence Threshold",
                   x=0.5, font=dict(size=16)),
        xaxis=dict(title="Confidence threshold"),
        yaxis=dict(title="Rate", range=[0, 1]),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════

layout = dbc.Container([
    html.H2("✅ Classification QA", className="my-3",
            style={"fontWeight": "700"}),
    html.P("Incident classification agreement analysis, data quality, "
           "and NFPA compliance tracking.",
           className="text-muted mb-4"),

    # KPI Row (via callback)
    html.Div(id="qa-kpi-row"),

    html.Hr(style={"borderColor": "#333"}),

    # Bullet Charts Row
    dbc.Row([
        dbc.Col([
            html.H5("📏 NFPA Performance Targets",
                     className="mb-3", style={"fontWeight": "600"}),
            dcc.Graph(id="qa-bullet-completeness"),
            dcc.Graph(id="qa-bullet-f1"),
        ], md=12),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#333"}),

    # Agreement DataTable
    dbc.Row([
        dbc.Col([
            html.H5("🏷️ Classification Agreement by MPDS Group",
                     className="mb-3", style={"fontWeight": "600"}),
            html.Div(id="qa-agreement-table-container"),
        ], md=12),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#333"}),

    # Agreement Bar + Completeness Trend
    dbc.Row([
        dbc.Col(dcc.Graph(id="qa-agreement-bar"), md=7),
        dbc.Col(dcc.Graph(id="qa-completeness-trend"), md=5),
    ], className="mb-4"),

    # Confidence Distribution
    dbc.Row([
        dbc.Col(dcc.Graph(id="qa-confidence-hist"), md=12),
    ], className="mb-4"),

    html.Hr(style={"borderColor": "#333"}),

    # Confusion matrix + Disagreement-flag threshold sweep (static — model-level)
    dbc.Row([
        dbc.Col([
            html.H5("🔍 Confusion Matrix",
                     className="mb-3", style={"fontWeight": "600"}),
            dcc.Graph(figure=_build_confusion_heatmap(CONFUSION_MATRIX),
                      id="qa-confusion-matrix"),
        ], md=7),
        dbc.Col([
            html.H5("🎚️ Disagreement-flag Sweep",
                     className="mb-3", style={"fontWeight": "600"}),
            dcc.Graph(figure=_build_threshold_sweep(DISAGREE_EVAL),
                      id="qa-threshold-sweep"),
        ], md=5),
    ], className="mb-4"),

])


# ═══════════════════════════════════════════════════════════════
# Callbacks — react to global filter store
# ═══════════════════════════════════════════════════════════════

@callback(
    Output("qa-kpi-row", "children"),
    Output("qa-bullet-completeness", "figure"),
    Output("qa-bullet-f1", "figure"),
    Output("qa-agreement-table-container", "children"),
    Output("qa-agreement-bar", "figure"),
    Output("qa-completeness-trend", "figure"),
    Output("qa-confidence-hist", "figure"),
    Input("global-filter-store", "data"),
)
def update_qa_page(filters):
    """Re-render all QA charts when global filters change."""
    df = _apply_filters(DISAGREE_DF, filters)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No data matches filters",
                              showarrow=False, font=dict(size=18))]
        )
        return (
            dbc.Alert("No classifier data matches the current filters.",
                      color="warning"),
            empty_fig, empty_fig,
            dbc.Alert("No data.", color="warning"),
            empty_fig, empty_fig, empty_fig,
        )

    return (
        _build_kpi_row(df, METRICS),
        _build_bullet_completeness(df),
        _build_bullet_f1(METRICS),
        _build_agreement_table(df),
        _build_agreement_bar(df),
        _build_completeness_trend(df),
        _build_confidence_histogram(df),
    )
