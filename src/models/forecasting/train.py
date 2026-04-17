"""
forecasting/train.py — Demand Forecasting Model.

Owner: Sanika (C3)
Phase: 2

Responsibilities:
  - Prophet univariate + LightGBM ensemble
  - 4-quarter horizon, walk-forward CV
  - Target: MAPE < 15%
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.forecasting.ensemble import train_and_serialize_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.forecasting")

warnings.filterwarnings("ignore")

FORECASTING_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "forecasting"


def load_temporal_data() -> pd.DataFrame:
    """Load and aggregate dispatch data by month to form a univariate time series."""
    path = PROCESSED_DATA_DIR / "fact_dispatch_clean.parquet"
    if not path.exists():
        log.error("Clean data not found: %s", path)
        log.error("Ensure Phase 1 ingestion is complete.")
        sys.exit(1)
        
    df = pd.read_parquet(path)
    
    if "date" in df.columns:
        df["ds"] = pd.to_datetime(df["date"])
    else:
        # Fallback using 'year' and 'month' from feature engineering if available
        if "year" in df.columns and "month" in df.columns:
            df["ds"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
                format="%Y-%m-%d", errors="coerce"
            )
        else:
            log.error("Cannot construct 'ds'; required time columns missing.")
            sys.exit(1)
            
    df = df.dropna(subset=["ds"])
    
    # Monthly aggregation
    ts = df.groupby(df["ds"].dt.to_period("M")).size().reset_index(name="y")
    ts["ds"] = ts["ds"].dt.to_timestamp()
    
    log.info("Aggregated %d monthly timepoints.", len(ts))
    return ts.sort_values("ds")


def train_prophet_model(ts: pd.DataFrame):
    """Train Prophet univariate model."""
    log.info("Training Prophet univariate model on %d time points...", len(ts))
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(ts)
    return model


def evaluate_walk_forward(model) -> dict:
    """Execute walk-forward cross validation for 4-quarter horizon."""
    log.info("Running walk-forward cross validation (4-quarter horizon)...")
    try:
        # 4 quarters ≈ 365 days. 
        # Using initial=2 years (730 days) if data permits.
        df_cv = cross_validation(model, initial="730 days", period="180 days", horizon="365 days")
        df_p = performance_metrics(df_cv)
        
        avg_mape = df_p["mape"].mean()
        log.info("Walk-forward CV complete. Average MAPE: %.2f%%", avg_mape * 100)
    except Exception as e:
        log.warning("Walk-forward CV failed (potentially insufficient data length): %s", e)
        avg_mape = 0.0

    target_met = bool(0 < avg_mape < 0.15)
    if target_met:
        log.info("  ✅ PASSED MAPE target (<15%)")
    else:
        log.warning("  ⚠️  DID NOT MEET target (<15%)" if avg_mape > 0 else "")

    return {
        "avg_mape": float(avg_mape),
        "target_met": target_met
    }


def save_artifacts(model, metrics: dict, output_dir: Path):
    """Serialize artifacts for Dash components and C4 ensemble."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if _HAS_PROPHET:
        from prophet.serialize import model_to_json
        
        # Serialize Prophet in native json format
        model_path = output_dir / "prophet_model.json"
        with open(model_path, "w") as fout:
            json.dump(model_to_json(model), fout)
        log.info("Saved Prophet model: %s", model_path)
        
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    log.info("Saved metrics: %s", metrics_path)


def log_to_mlflow(metrics: dict):
    """Log results to MLflow tracking."""
    try:
        import mlflow
        mlflow.set_experiment("medalertai-forecasting")
        with mlflow.start_run(run_name="prophet_univariate"):
            mlflow.log_metric("avg_mape", metrics["avg_mape"])
            log.info("MLflow run logged successfully.")
    except Exception as exc:
        log.warning("MLflow logging failed (non-fatal): %s", exc)


def run_training(use_mlflow: bool = True, skip_ensemble: bool = False):
    start = time.time()
    log.info("=== MedAlertAI Forecasting Pipeline (Sanika) ===")
    
    if not _HAS_PROPHET:
        log.warning("Prophet not installed. Run `pip install prophet` first.")
        # Cannot proceed without Prophet
        sys.exit(1)

    ts = load_temporal_data()
    
    if len(ts) < 24:
        log.warning("Data length (%d months) may be too short for 4-quarter CV.", len(ts))
        
    model = train_prophet_model(ts)
    metrics = evaluate_walk_forward(model)
    
    save_artifacts(model, metrics, FORECASTING_ARTIFACTS_DIR)

    if not skip_ensemble:
        try:
            ensemble_summary = train_and_serialize_ensemble(
                ts=ts,
                prophet_model=model,
                output_dir=FORECASTING_ARTIFACTS_DIR,
            )
            metrics["ensemble"] = ensemble_summary
            save_artifacts(model, metrics, FORECASTING_ARTIFACTS_DIR)
        except Exception as exc:
            log.warning("C4 ensemble serialization failed (non-fatal): %s", exc)
    
    if use_mlflow:
        log_to_mlflow(metrics)
        
    log.info("=== Forecasting Pipeline complete in %.1f seconds ===", time.time() - start)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MedAlertAI — Phase 2 Forecasting Training")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Skip C4 LightGBM ensemble serialization",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_training(use_mlflow=not args.no_mlflow, skip_ensemble=args.skip_ensemble)
