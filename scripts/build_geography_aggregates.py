"""
scripts/build_geography_aggregates.py
-------------------------------------
Pre-aggregates fact_dispatch_clean.parquet for the Geographic Analysis
dashboard page so it doesn't have to load 134 MB and run DBSCAN on
3.3 M points on every callback.

Outputs (all under data/processed/):
    geo_density_agg.parquet
        per (year, service, call_type, census_block_group): call_count + lat/lon centroid
    geo_city_agg.parquet
        per (year, service, city_name): call_count + sum/count of completeness_score
    geo_call_type_cbg_agg.parquet
        per (year, service, call_type): total_calls + unique_cbg_count

Run after the fact table is rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"
DENSITY_OUT = REPO_ROOT / "data" / "processed" / "geo_density_agg.parquet"
CITY_OUT = REPO_ROOT / "data" / "processed" / "geo_city_agg.parquet"
CALL_TYPE_OUT = REPO_ROOT / "data" / "processed" / "geo_call_type_cbg_agg.parquet"

NEEDED_COLS = [
    "CALL_YEAR",
    "service_type",
    "census_block_group",
    "city_name",
    "call_type",
    "latitude",
    "longitude",
    "completeness_score",
]


def _load() -> pd.DataFrame:
    df = pd.read_parquet(SOURCE, columns=NEEDED_COLS)
    df = df.rename(columns={"CALL_YEAR": "year", "service_type": "service"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df["service"] = df["service"].astype("string").str.strip()
    df["call_type"] = df["call_type"].astype("string").fillna("Unknown").str.strip()
    df["city_name"] = df["city_name"].astype("string").str.strip()
    df["census_block_group"] = df["census_block_group"].astype("string")
    return df.dropna(subset=["year"])


def build_density_agg(df: pd.DataFrame) -> pd.DataFrame:
    geo = df.dropna(subset=["latitude", "longitude", "census_block_group", "call_type"])
    return (
        geo.groupby(
            ["year", "service", "call_type", "census_block_group"], observed=True
        )
        .agg(
            call_count=("latitude", "size"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )


def build_city_agg(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["city_name"])
    return (
        sub.groupby(["year", "service", "city_name"], observed=True)
        .agg(
            call_count=("city_name", "size"),
            completeness_sum=("completeness_score", "sum"),
            completeness_count=("completeness_score", "count"),
        )
        .reset_index()
    )


def build_call_type_cbg_agg(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["call_type"])
    return (
        sub.groupby(["year", "service", "call_type"], observed=True)
        .agg(
            total_calls=("call_type", "size"),
            unique_cbg_count=("census_block_group", "nunique"),
        )
        .reset_index()
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source parquet not found: {SOURCE}")

    print(f"Reading {SOURCE} ...")
    df = _load()
    print(f"  loaded {len(df):,} rows")

    density = build_density_agg(df)
    city = build_city_agg(df)
    call_type = build_call_type_cbg_agg(df)

    DENSITY_OUT.parent.mkdir(parents=True, exist_ok=True)
    density.to_parquet(DENSITY_OUT, index=False)
    city.to_parquet(CITY_OUT, index=False)
    call_type.to_parquet(CALL_TYPE_OUT, index=False)

    print(f"Wrote {DENSITY_OUT}     ({len(density):,} rows)")
    print(f"Wrote {CITY_OUT}        ({len(city):,} rows)")
    print(f"Wrote {CALL_TYPE_OUT}   ({len(call_type):,} rows)")


if __name__ == "__main__":
    main()
