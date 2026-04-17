"""
Forecast ensemble utilities for Phase 2.

Owner: Srileakhana (C4)
Phase: 2

Responsibilities:
  - Train a LightGBM-compatible monthly demand forecaster.
  - Combine Prophet and LightGBM forecasts into a single ensemble output.
  - Serialize forecasting artifacts for Dash callbacks and evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

from config.settings import FORECAST_HORIZON_QUARTERS, MODEL_ARTIFACTS_DIR, PROJECT_ROOT

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except (ImportError, OSError):
    LGBMRegressor = None
    _HAS_LGBM = False


log = logging.getLogger("medalertai.forecasting.ensemble")

FORECASTING_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "forecasting"
LIGHTGBM_MODEL_PATH = FORECASTING_ARTIFACTS_DIR / "lightgbm_forecaster.joblib"
ENSEMBLE_MODEL_PATH = FORECASTING_ARTIFACTS_DIR / "ensemble_model.joblib"
ENSEMBLE_FORECAST_PATH = FORECASTING_ARTIFACTS_DIR / "ensemble_forecast.parquet"
FORECAST_MANIFEST_PATH = FORECASTING_ARTIFACTS_DIR / "artifact_manifest.json"

DEFAULT_LAGS = (1, 2, 3, 6, 12)
DEFAULT_ROLLING_WINDOWS = (3, 6)
HORIZON_MONTHS = FORECAST_HORIZON_QUARTERS * 3


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _validate_time_series(ts: pd.DataFrame, allow_missing_y: bool = False) -> pd.DataFrame:
    required = {"ds", "y"}
    missing = required - set(ts.columns)
    if missing:
        raise ValueError(f"Forecast time series missing required columns: {sorted(missing)}")

    clean = ts.loc[:, ["ds", "y"]].copy()
    clean["ds"] = pd.to_datetime(clean["ds"], errors="coerce")
    clean["y"] = pd.to_numeric(clean["y"], errors="coerce")
    required_non_null = ["ds"] if allow_missing_y else ["ds", "y"]
    clean = clean.dropna(subset=required_non_null).sort_values("ds").reset_index(drop=True)

    if clean.empty:
        raise ValueError("Forecast time series has no valid rows after cleaning.")
    if not allow_missing_y and clean["y"].isna().any():
        raise ValueError("Forecast time series has missing target values.")

    return clean


def build_lightgbm_features(
    ts: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    dropna: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create deterministic monthly lag and seasonality features."""
    features = _validate_time_series(ts, allow_missing_y=not dropna)
    features["month"] = features["ds"].dt.month.astype("int16")
    features["quarter"] = features["ds"].dt.quarter.astype("int16")
    features["year"] = features["ds"].dt.year.astype("int16")
    features["time_index"] = np.arange(len(features), dtype=np.int32)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

    feature_columns = [
        "month",
        "quarter",
        "year",
        "time_index",
        "month_sin",
        "month_cos",
    ]

    for lag in lags:
        col = f"lag_{lag}"
        features[col] = features["y"].shift(lag)
        feature_columns.append(col)

    for window in rolling_windows:
        col = f"rolling_mean_{window}"
        features[col] = features["y"].shift(1).rolling(window=window).mean()
        feature_columns.append(col)

    if dropna:
        features = features.dropna(subset=feature_columns + ["y"]).reset_index(drop=True)

    return features, feature_columns


def _build_regressor(random_state: int = 42):
    if _HAS_LGBM:
        return LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=random_state,
    )


def train_lightgbm_forecaster(ts: pd.DataFrame, random_state: int = 42) -> tuple[object, dict]:
    """Train the LightGBM forecasting component and return model metadata."""
    feature_frame, feature_columns = build_lightgbm_features(ts)
    if len(feature_frame) < 6:
        raise ValueError(
            "Need at least 18 monthly observations to train the LightGBM forecaster "
            "with the default lag features."
        )

    holdout_size = min(max(3, int(len(feature_frame) * 0.2)), max(1, len(feature_frame) - 3))
    train_df = feature_frame.iloc[:-holdout_size]
    val_df = feature_frame.iloc[-holdout_size:]

    model = _build_regressor(random_state=random_state)
    model.fit(train_df[feature_columns], train_df["y"])

    val_pred = np.clip(model.predict(val_df[feature_columns]), 0, None)
    mape = mean_absolute_percentage_error(val_df["y"], val_pred)

    model.fit(feature_frame[feature_columns], feature_frame["y"])

    metadata = {
        "engine": "LightGBM" if _HAS_LGBM else "HistGradientBoostingRegressor",
        "feature_columns": feature_columns,
        "validation": {
            "mape": float(mape),
            "n_validation_rows": int(len(val_df)),
        },
        "n_training_rows": int(len(feature_frame)),
    }
    return model, metadata


def forecast_lightgbm(
    model,
    history: pd.DataFrame,
    feature_columns: list[str],
    periods: int = HORIZON_MONTHS,
) -> pd.DataFrame:
    """Recursively forecast monthly demand with the trained LightGBM component."""
    history_clean = _validate_time_series(history)
    working = history_clean.copy()
    forecasts = []

    for _ in range(periods):
        next_ds = working["ds"].max() + pd.offsets.MonthBegin(1)
        candidate = pd.concat(
            [working, pd.DataFrame({"ds": [next_ds], "y": [np.nan]})],
            ignore_index=True,
        )
        feature_frame, _ = build_lightgbm_features(candidate, dropna=False)
        row = feature_frame.iloc[[-1]]

        missing_features = [col for col in feature_columns if pd.isna(row.iloc[0][col])]
        if missing_features:
            raise ValueError(
                "Unable to create recursive forecast row; missing features: "
                f"{missing_features}"
            )

        yhat = float(np.clip(model.predict(row[feature_columns])[0], 0, None))
        forecasts.append({"ds": next_ds, "lightgbm_yhat": yhat})
        working = pd.concat(
            [working, pd.DataFrame({"ds": [next_ds], "y": [yhat]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


@dataclass
class ForecastEnsemble:
    """Weighted combiner for Prophet and LightGBM monthly forecasts."""

    prophet_weight: float = 0.5
    lightgbm_weight: float = 0.5

    def __post_init__(self) -> None:
        total = self.prophet_weight + self.lightgbm_weight
        if total <= 0:
            raise ValueError("Forecast ensemble weights must sum to a positive value.")
        self.prophet_weight = self.prophet_weight / total
        self.lightgbm_weight = self.lightgbm_weight / total

    @property
    def weights(self) -> dict[str, float]:
        return {
            "prophet": float(self.prophet_weight),
            "lightgbm": float(self.lightgbm_weight),
        }

    def combine(self, prophet_forecast: pd.DataFrame, lightgbm_forecast: pd.DataFrame) -> pd.DataFrame:
        prophet = prophet_forecast.copy()
        lightgbm = lightgbm_forecast.copy()
        prophet["ds"] = pd.to_datetime(prophet["ds"])
        lightgbm["ds"] = pd.to_datetime(lightgbm["ds"])

        if "yhat" not in prophet.columns:
            raise ValueError("Prophet forecast must include a 'yhat' column.")
        if "lightgbm_yhat" not in lightgbm.columns:
            if "yhat" in lightgbm.columns:
                lightgbm = lightgbm.rename(columns={"yhat": "lightgbm_yhat"})
            else:
                raise ValueError("LightGBM forecast must include 'lightgbm_yhat' or 'yhat'.")

        merged = prophet[["ds", "yhat"]].rename(columns={"yhat": "prophet_yhat"}).merge(
            lightgbm[["ds", "lightgbm_yhat"]],
            on="ds",
            how="inner",
        )

        if merged.empty:
            raise ValueError("No overlapping dates between Prophet and LightGBM forecasts.")

        merged["ensemble_yhat"] = (
            self.prophet_weight * merged["prophet_yhat"]
            + self.lightgbm_weight * merged["lightgbm_yhat"]
        )
        merged["yhat"] = merged["ensemble_yhat"]

        if {"yhat_lower", "yhat_upper"}.issubset(prophet.columns):
            intervals = prophet[["ds", "yhat_lower", "yhat_upper"]].copy()
            merged = merged.merge(intervals, on="ds", how="left")

        return merged


def build_prophet_future_forecast(prophet_model, periods: int = HORIZON_MONTHS) -> pd.DataFrame:
    """Create a monthly Prophet forecast frame for the next forecast horizon."""
    future = prophet_model.make_future_dataframe(periods=periods, freq="MS", include_history=False)
    return prophet_model.predict(future)


def save_ensemble_artifacts(
    lightgbm_model,
    lightgbm_metadata: dict,
    ensemble: ForecastEnsemble,
    ensemble_forecast: pd.DataFrame,
    output_dir: Path = FORECASTING_ARTIFACTS_DIR,
) -> dict:
    """Serialize LightGBM, ensemble metadata, and dashboard-ready forecasts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lightgbm_model_path = output_dir / LIGHTGBM_MODEL_PATH.name
    ensemble_model_path = output_dir / ENSEMBLE_MODEL_PATH.name
    forecast_path = output_dir / ENSEMBLE_FORECAST_PATH.name
    metadata_path = output_dir / "lightgbm_metadata.json"
    manifest_path = output_dir / FORECAST_MANIFEST_PATH.name

    joblib.dump(lightgbm_model, lightgbm_model_path)
    joblib.dump(ensemble, ensemble_model_path)
    ensemble_forecast.to_parquet(forecast_path, index=False)
    metadata_path.write_text(json.dumps(lightgbm_metadata, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "forecasting": {
            "prophet_model": "prophet_model.json",
            "lightgbm_model": lightgbm_model_path.name,
            "ensemble_model": ensemble_model_path.name,
            "ensemble_forecast": forecast_path.name,
            "lightgbm_metadata": metadata_path.name,
            "metrics": "metrics.json",
        },
        "ensemble_weights": ensemble.weights,
        "dashboard_output_columns": list(ensemble_forecast.columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log.info("Saved LightGBM forecaster: %s", _display_path(lightgbm_model_path))
    log.info("Saved ensemble combiner: %s", _display_path(ensemble_model_path))
    log.info("Saved ensemble forecast: %s", _display_path(forecast_path))
    log.info("Saved forecast artifact manifest: %s", _display_path(manifest_path))
    return manifest


def train_and_serialize_ensemble(
    ts: pd.DataFrame,
    prophet_model,
    periods: int = HORIZON_MONTHS,
    output_dir: Path = FORECASTING_ARTIFACTS_DIR,
) -> dict:
    """Train the C4 ensemble layer after the Prophet model is available."""
    lightgbm_model, metadata = train_lightgbm_forecaster(ts)
    lightgbm_forecast = forecast_lightgbm(
        lightgbm_model,
        history=ts,
        feature_columns=metadata["feature_columns"],
        periods=periods,
    )
    prophet_forecast = build_prophet_future_forecast(prophet_model, periods=periods)
    ensemble = ForecastEnsemble()
    ensemble_forecast = ensemble.combine(prophet_forecast, lightgbm_forecast)
    manifest = save_ensemble_artifacts(
        lightgbm_model=lightgbm_model,
        lightgbm_metadata=metadata,
        ensemble=ensemble,
        ensemble_forecast=ensemble_forecast,
        output_dir=output_dir,
    )
    return {
        "manifest": manifest,
        "lightgbm_metadata": metadata,
        "ensemble_rows": int(len(ensemble_forecast)),
    }
