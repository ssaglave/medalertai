"""
scripts/build_temporal_aggregates.py
------------------------------------
Pre-aggregates the 134 MB fact_dispatch_clean.parquet into two small parquets
that the Temporal Analysis dashboard page can load instantly.

Outputs:
    data/processed/temporal_heatmap_agg.parquet
        columns: year, quarter, service, call_count
    data/processed/temporal_slope_agg.parquet
        columns: year, service, call_type, call_count

Run once after the fact table is rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"
HEATMAP_OUT = REPO_ROOT / "data" / "processed" / "temporal_heatmap_agg.parquet"
SLOPE_OUT = REPO_ROOT / "data" / "processed" / "temporal_slope_agg.parquet"

NEEDED_COLS = ["CALL_YEAR", "CALL_QUARTER", "service_type", "call_type"]


def _load_source() -> pd.DataFrame:
    df = pd.read_parquet(SOURCE, columns=NEEDED_COLS)
    df = df.rename(
        columns={
            "CALL_YEAR": "year",
            "CALL_QUARTER": "quarter",
            "service_type": "service",
        }
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df["quarter"] = (
        df["quarter"]
        .astype("string")
        .str.upper()
        .str.replace("QUARTER ", "Q", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"})
    )
    df["service"] = df["service"].astype("string").str.strip()
    df["call_type"] = df["call_type"].astype("string").fillna("Unknown").str.strip()
    return df.dropna(subset=["year", "quarter"])


def build_heatmap_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year", "quarter", "service"], observed=True)
        .size()
        .reset_index(name="call_count")
    )


def build_slope_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["year", "service", "call_type"], observed=True)
        .size()
        .reset_index(name="call_count")
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source parquet not found: {SOURCE}")

    print(f"Reading {SOURCE} ...")
    df = _load_source()
    print(f"  loaded {len(df):,} rows")

    heatmap_agg = build_heatmap_agg(df)
    slope_agg = build_slope_agg(df)

    HEATMAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    heatmap_agg.to_parquet(HEATMAP_OUT, index=False)
    slope_agg.to_parquet(SLOPE_OUT, index=False)

    print(f"Wrote {HEATMAP_OUT}  ({len(heatmap_agg):,} rows)")
    print(f"Wrote {SLOPE_OUT}  ({len(slope_agg):,} rows)")


if __name__ == "__main__":
    main()
