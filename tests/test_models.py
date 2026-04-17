"""
test_models.py — ML model tests.

Owners: Suvarna (C2), Sanika (C3)
Phase: 5

Targets:
  - Classifier: macro F1 > 0.75
  - Forecaster: MAPE < 15%
  - Clustering: Silhouette > 0.4, Recall@20 > 0.7
"""
import pytest
import pandas as pd

from src.models.forecasting.ensemble import (
    ForecastEnsemble,
    build_lightgbm_features,
    forecast_lightgbm,
    save_ensemble_artifacts,
    train_lightgbm_forecaster,
)


def test_placeholder():
    """Placeholder test — replace with real tests."""
    assert True


def _sample_monthly_series(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=periods, freq="MS")
    values = [100 + (idx * 2) + ((idx % 12) * 3) for idx in range(periods)]
    return pd.DataFrame({"ds": dates, "y": values})


def test_build_lightgbm_features_creates_lag_columns():
    features, feature_columns = build_lightgbm_features(_sample_monthly_series())

    assert "lag_1" in feature_columns
    assert "rolling_mean_3" in feature_columns
    assert not features[feature_columns].isna().any().any()


def test_lightgbm_forecaster_trains_and_predicts_future_months():
    ts = _sample_monthly_series()

    model, metadata = train_lightgbm_forecaster(ts)
    forecast = forecast_lightgbm(model, ts, metadata["feature_columns"], periods=4)

    assert len(forecast) == 4
    assert list(forecast.columns) == ["ds", "lightgbm_yhat"]
    assert (forecast["lightgbm_yhat"] >= 0).all()


def test_forecast_ensemble_combines_prophet_and_lightgbm_outputs():
    dates = pd.date_range("2024-07-01", periods=4, freq="MS")
    prophet = pd.DataFrame({"ds": dates, "yhat": [100, 110, 120, 130]})
    lightgbm = pd.DataFrame({"ds": dates, "lightgbm_yhat": [90, 100, 110, 120]})

    combined = ForecastEnsemble(prophet_weight=0.6, lightgbm_weight=0.4).combine(
        prophet,
        lightgbm,
    )

    assert combined["ensemble_yhat"].round(2).tolist() == [96.0, 106.0, 116.0, 126.0]
    assert combined["yhat"].equals(combined["ensemble_yhat"])


def test_save_ensemble_artifacts_writes_dashboard_manifest(tmp_path):
    ts = _sample_monthly_series()
    model, metadata = train_lightgbm_forecaster(ts)
    forecast = pd.DataFrame({
        "ds": pd.date_range("2024-07-01", periods=2, freq="MS"),
        "prophet_yhat": [100.0, 110.0],
        "lightgbm_yhat": [95.0, 105.0],
        "ensemble_yhat": [97.5, 107.5],
        "yhat": [97.5, 107.5],
    })

    manifest = save_ensemble_artifacts(
        lightgbm_model=model,
        lightgbm_metadata=metadata,
        ensemble=ForecastEnsemble(),
        ensemble_forecast=forecast,
        output_dir=tmp_path,
    )

    assert (tmp_path / "lightgbm_forecaster.joblib").exists()
    assert (tmp_path / "ensemble_model.joblib").exists()
    assert (tmp_path / "ensemble_forecast.parquet").exists()
    assert manifest["forecasting"]["ensemble_forecast"] == "ensemble_forecast.parquet"
