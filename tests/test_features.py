import pandas as pd
from src.data.feature_engineering import (
    add_cyclical_features,
    add_geo_target_encoding,
    add_lag_features
)


def test_cyclical_features():
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=10, freq="h")
    })

    result = add_cyclical_features(df, "time")

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns


def test_geo_encoding():
    df = pd.DataFrame({
        "location": ["A", "A", "B"],
        "target": [1, 2, 3]
    })

    result = add_geo_target_encoding(df, "location", "target")

    assert "location_target_enc" in result.columns


def test_lag_features():
    df = pd.DataFrame({
        "value": [10, 20, 30, 40]
    })

    result = add_lag_features(df, "value")

    assert "value_lag_1" in result.columns