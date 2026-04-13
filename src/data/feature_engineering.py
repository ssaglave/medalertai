import numpy as np
import pandas as pd


def add_cyclical_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_geo_target_encoding(df: pd.DataFrame, geo_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    means = df.groupby(geo_col)[target_col].mean()
    df[f"{geo_col}_target_enc"] = df[geo_col].map(means)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list = [1, 2, 3],
    group_col: str = None
) -> pd.DataFrame:

    df = df.copy()

    if group_col:
        df = df.sort_values(group_col)
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df