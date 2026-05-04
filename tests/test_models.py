"""
test_models.py — ML model tests.

Owners: Suvarna (C2), Sanika (C3)
Phase: 5

Targets (real-data plan targets, revised):
  - Classifier: macro F1 > 0.55 (regression floor 0.42)
  - Forecaster: MAPE < 15%
  - Clustering: Silhouette > 0.35, Recall@20 > 0.25

The clustering and classifier asserts in *this* file run on synthetic
in-test data engineered to clear the original (pre-revision) targets
of 0.4 / 0.7 — they validate the function logic, not the real-data
targets, so the assertion thresholds here do not change.
"""
import pytest
import pandas as pd
import numpy as np

from src.models.forecasting.train import train_prophet_model, evaluate_walk_forward
from src.models.clustering.train import train_dbscan, train_isolation_forest

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

from src.models.forecasting.ensemble import (
    ForecastEnsemble,
    build_lightgbm_features,
    forecast_lightgbm,
    save_ensemble_artifacts,
    train_lightgbm_forecaster,
)


@pytest.mark.skipif(not _HAS_PROPHET, reason="Prophet not installed")
def test_train_prophet_model():
    ts = _sample_monthly_series(periods=36)
    model = train_prophet_model(ts)
    assert model is not None
    assert hasattr(model, "predict")


@pytest.mark.skipif(not _HAS_PROPHET, reason="Prophet not installed")
def test_evaluate_walk_forward():
    ts = _sample_monthly_series(periods=72)
    model = train_prophet_model(ts)
    # The dummy data is extremely predictable, so MAPE should be very low
    metrics = evaluate_walk_forward(model)
    assert "avg_mape" in metrics
    assert "target_met" in metrics
    assert metrics["target_met"] is True
    assert metrics["avg_mape"] < 0.15


def test_train_dbscan_silhouette():
    np.random.seed(42)
    df1 = pd.DataFrame({
        "latitude": np.random.normal(40.0, 0.001, 20),
        "longitude": np.random.normal(-80.0, 0.001, 20)
    })
    df2 = pd.DataFrame({
        "latitude": np.random.normal(40.5, 0.001, 20),
        "longitude": np.random.normal(-80.5, 0.001, 20)
    })
    hotspot_df = pd.concat([df1, df2]).reset_index(drop=True)
    res = train_dbscan(hotspot_df)
    
    assert res["silhouette_score"] > 0.4
    assert res["target_met"] is True
    assert res["n_clusters"] >= 2


def test_train_isolation_forest_recall():
    np.random.seed(42)
    df_reg = pd.DataFrame({
        "latitude": np.random.normal(40.0, 0.001, 100),
        "longitude": np.random.normal(-80.0, 0.001, 100),
        "hour": 12,
        "day_of_week": 1,
        "month": 1,
        "priority_code": "E3",
        "incident_id": range(100)
    })
    df_anom = pd.DataFrame({
        "latitude": np.random.normal(45.0, 0.001, 10),
        "longitude": np.random.normal(-85.0, 0.001, 10),
        "hour": 3,
        "day_of_week": 6,
        "month": 6,
        "priority_code": "E1",
        "incident_id": range(100, 110)
    })
    full_df = pd.concat([df_reg, df_anom]).reset_index(drop=True)
    res = train_isolation_forest(full_df)
    
    assert "recall_20" in res
    assert res["recall_20"] > 0.7
    assert res["target_met"] is True


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
