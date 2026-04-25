"""
scripts/build_overview_aggregates.py
------------------------------------
Pre-aggregates fact_dispatch_clean.parquet for the Overview dashboard
page so it doesn't have to load 134 MB on import.

Output:
    data/processed/overview_agg.parquet
        per (year, quarter, service, priority_level, call_type):
            call_count, with_coords_count, high_completeness_count

priority_level is the 5-bucket mapping used by the page (Life Threatening,
ALS, BLS, Non-Emergency, Other), pre-computed here so callbacks never
have to apply the regex mapping at request time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet"
OUT = REPO_ROOT / "data" / "processed" / "overview_agg.parquet"

NEEDED_COLS = [
    "CALL_YEAR",
    "CALL_QUARTER",
    "service_type",
    "call_type",
    "priority_description",
    "longitude",
    "completeness_score",
]

HIGH_COMPLETENESS_THRESHOLD = 0.75


def _map_priority_level(desc: object) -> str:
    if pd.isna(desc):
        return "Other"
    d = str(desc).lower()
    if "life threatening" in d:
        return "Life Threatening"
    if "advanced life support" in d or "als" in d:
        return "ALS"
    if "basic life support" in d or "bls" in d:
        return "BLS"
    if any(k in d for k in ["assist", "admin", "non emergency",
                            "no immediate threat", "mark out"]):
        return "Non-Emergency"
    return "Other"


def _load() -> pd.DataFrame:
    df = pd.read_parquet(SOURCE, columns=NEEDED_COLS)
    df = df.rename(columns={"CALL_YEAR": "year",
                            "CALL_QUARTER": "quarter",
                            "service_type": "service"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int16")
    df["quarter"] = (
        df["quarter"].astype("string").str.upper()
        .str.replace("QUARTER ", "Q", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"})
    )
    df["service"] = df["service"].astype("string").str.strip()
    df["call_type"] = df["call_type"].astype("string").fillna("Unknown").str.strip()
    df["priority_level"] = df["priority_description"].map(_map_priority_level)
    df["with_coords"] = df["longitude"].notna()
    df["high_completeness"] = df["completeness_score"].fillna(0) >= HIGH_COMPLETENESS_THRESHOLD
    return df.dropna(subset=["year", "quarter"])


def build_overview_agg(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["year", "quarter", "service", "priority_level", "call_type"],
            observed=True,
        )
        .agg(
            call_count=("call_type", "size"),
            with_coords_count=("with_coords", "sum"),
            high_completeness_count=("high_completeness", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source parquet not found: {SOURCE}")

    print(f"Reading {SOURCE} ...")
    df = _load()
    print(f"  loaded {len(df):,} rows")

    agg = build_overview_agg(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"Wrote {OUT}  ({len(agg):,} rows)")


if __name__ == "__main__":
    main()
