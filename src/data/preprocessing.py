"""
src/data/preprocessing.py
--------------------------
Phase 1 — NEMSIS-aligned preprocessing pipeline
Owner: Greeshma

Reads raw EMS_Data.csv and Fire_Data.csv from data/raw/,
normalises columns to the shared schema defined in config/contracts.py,
and writes a clean Parquet file to data/processed/fact_dispatch_clean.parquet.

Usage (from repo root):
    python -m src.data.preprocessing          # processes both files
    python -m src.data.preprocessing --dry-run # validate only, no writes
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.preprocessing")

# ---------------------------------------------------------------------------
# Paths  (resolve relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _REPO_ROOT / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "fact_dispatch_clean.parquet"

EMS_CSV = RAW_DIR / "EMS_Data.csv"
FIRE_CSV = RAW_DIR / "Fire_Data.csv"

# ---------------------------------------------------------------------------
# NEMSIS / WPRDC column mapping
# Raw CSV header  →  canonical name used everywhere in the project
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP: dict[str, str] = {
    # identifiers
    "CAD_INCIDENT_ID":             "call_id_hash",
    "INCIDENT_ID":                 "call_id_hash",
    # service type  (EMS file already has this; Fire file gets it injected)
    "SERVICE_TYPE":                "service",
    # priority
    "PRIORITY":                    "priority",
    "PRIORITY_DESC":               "priority_desc",
    "INITIAL_PRIORITY":            "priority",
    "INITIAL_PRIORITY_DESC":       "priority_desc",
    # call description
    "DESCRIPTION_SHORT":           "description_short",
    "CALL_TYPE":                   "description_short",
    "INCIDENT_TYPE":               "description_short",
    # temporal
    "CALL_CREATE_TIME":            "call_create_time",
    "INCIDENT_DATE":               "call_create_time",
    "CREATE_TIME_INCIDENT":        "call_create_time",
    # geography
    "CITY_CODE":                   "city_code",
    "CITY_NAME":                   "city_name",
    "GEOID":                       "geoid",
    "CENSUS_BLOCK_GROUP_CENTER__X": "census_block_group_center__x",
    "CENSUS_BLOCK_GROUP_CENTER__Y": "census_block_group_center__y",
    "LONGITUDE":                   "census_block_group_center__x",
    "LATITUDE":                    "census_block_group_center__y",
    # response
    "RESPONSE_TIME_SEC":           "response_time_sec",
    "UNIT_DISPATCH_TIME":          "unit_dispatch_time",
    "UNIT_ON_SCENE_TIME":          "unit_on_scene_time",
}

# Columns that MUST be present in the final output (subset of contracts.py)
REQUIRED_COLUMNS: list[str] = [
    "call_id_hash",
    "service",
    "priority",
    "priority_desc",
    "call_quarter",
    "call_year",
    "description_short",
    "city_code",
    "city_name",
    "geoid",
    "census_block_group_center__x",
    "census_block_group_center__y",
]

# Canonical dtypes for the Parquet schema
PARQUET_DTYPES: dict[str, str] = {
    "call_id_hash":                 "string",
    "service":                      "category",
    "priority":                     "string",
    "priority_desc":                "string",
    "call_quarter":                 "int8",
    "call_year":                    "int16",
    "description_short":            "string",
    "city_code":                    "string",
    "city_name":                    "string",
    "geoid":                        "string",
    "census_block_group_center__x": "float32",
    "census_block_group_center__y": "float32",
    "response_time_sec":            "float32",
}


# ===========================================================================
# Helper utilities
# ===========================================================================

def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, uppercase, then apply rename map."""
    df.columns = [c.strip().upper() for c in df.columns]
    return df.rename(columns=COLUMN_RENAME_MAP)


def _parse_datetime(df: pd.DataFrame, col: str = "call_create_time") -> pd.DataFrame:
    """Robustly parse the incident datetime column."""
    if col not in df.columns:
        log.warning("Datetime column '%s' not found — skipping temporal features.", col)
        df["call_year"] = pd.NA
        df["call_quarter"] = pd.NA
        return df

    dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    null_count = dt.isna().sum()
    if null_count:
        log.warning("  %d rows have unparseable datetime values (set to NaT).", null_count)

    df[col] = dt
    df["call_year"] = dt.dt.year.astype("Int16")
    df["call_quarter"] = dt.dt.quarter.astype("Int8")
    return df


def _clean_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise priority codes and fill missing descriptions."""
    if "priority" in df.columns:
        df["priority"] = (
            df["priority"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
        )
    else:
        df["priority"] = pd.NA

    if "priority_desc" in df.columns:
        df["priority_desc"] = (
            df["priority_desc"]
            .astype(str)
            .str.strip()
            .str.title()
            .replace({"Nan": pd.NA, "None": pd.NA, "": pd.NA})
        )
    else:
        df["priority_desc"] = pd.NA

    return df


def _clean_description(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and strip call-type descriptions."""
    if "description_short" in df.columns:
        df["description_short"] = (
            df["description_short"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": pd.NA, "NONE": pd.NA, "": pd.NA})
        )
    else:
        log.warning("'description_short' not found — filling with NA.")
        df["description_short"] = pd.NA
    return df


def _clean_geography(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise city names, coerce coordinate columns."""
    if "city_name" in df.columns:
        df["city_name"] = df["city_name"].astype(str).str.strip().str.title()

    if "city_code" in df.columns:
        df["city_code"] = df["city_code"].astype(str).str.strip().str.upper()

    if "geoid" in df.columns:
        df["geoid"] = df["geoid"].astype(str).str.strip().str.zfill(12)

    for coord_col in ("census_block_group_center__x", "census_block_group_center__y"):
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce").astype("float32")
        else:
            df[coord_col] = np.nan

    return df


def _clean_response_times(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce response_time_sec; derive it from dispatch/on-scene if missing."""
    if "response_time_sec" not in df.columns:
        # Try to derive from unit timestamps
        if "unit_dispatch_time" in df.columns and "unit_on_scene_time" in df.columns:
            t0 = pd.to_datetime(df["unit_dispatch_time"], errors="coerce")
            t1 = pd.to_datetime(df["unit_on_scene_time"], errors="coerce")
            df["response_time_sec"] = (t1 - t0).dt.total_seconds().astype("float32")
            log.info("  Derived response_time_sec from unit timestamps.")
        else:
            df["response_time_sec"] = np.nan
    else:
        df["response_time_sec"] = pd.to_numeric(
            df["response_time_sec"], errors="coerce"
        ).astype("float32")

    # Clamp obviously invalid values: < 0 or > 2 hours
    mask_invalid = (df["response_time_sec"] < 0) | (df["response_time_sec"] > 7200)
    n_invalid = mask_invalid.sum()
    if n_invalid:
        log.warning("  Clamping %d out-of-range response_time_sec values to NaN.", n_invalid)
        df.loc[mask_invalid, "response_time_sec"] = np.nan

    return df


def _ensure_required_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Add any missing required columns as NA so downstream code never KeyErrors."""
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            log.warning("[%s] Required column '%s' missing — filling with NA.", source_label, col)
            df[col] = pd.NA
    return df


def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to canonical Parquet-friendly dtypes."""
    for col, dtype in PARQUET_DTYPES.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "category":
                df[col] = df[col].astype("category")
            elif dtype == "string":
                df[col] = df[col].astype(pd.StringDtype())
            elif dtype in ("float32", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
            elif dtype in ("int8", "int16", "int32", "int64"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(f"Int{dtype[3:]}")
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not cast column '%s' to %s: %s", col, dtype, exc)
    return df


def _compute_completeness_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'completeness_score' column (0.0–1.0) per row,
    based on how many of the REQUIRED_COLUMNS are non-null.
    """
    req = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df["completeness_score"] = (
        df[req].notna().sum(axis=1) / len(req)
    ).round(3).astype("float32")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and log how many were removed."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.info("  Dropped %d exact duplicate rows.", removed)
    return df


# ===========================================================================
# Per-file loaders
# ===========================================================================

def _read_csv(path: Path, service_label: str) -> pd.DataFrame:
    """Load a raw CSV, inject the service label, normalise column names."""
    log.info("Reading %s …", path.name)
    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8",
        on_bad_lines="warn",
    )
    log.info("  Raw shape: %s", df.shape)

    df = _normalise_column_names(df)

    # Inject service label if not already present
    if "service" not in df.columns:
        df["service"] = service_label
    else:
        df["service"] = df["service"].fillna(service_label)

    return df


def preprocess_ems(path: Path = EMS_CSV) -> pd.DataFrame:
    """Full preprocessing pipeline for EMS data."""
    log.info("--- EMS preprocessing ---")
    df = _read_csv(path, service_label="EMS")
    df = _parse_datetime(df)
    df = _clean_priority(df)
    df = _clean_description(df)
    df = _clean_geography(df)
    df = _clean_response_times(df)
    df = _ensure_required_columns(df, "EMS")
    df = _drop_duplicates(df)
    df = _compute_completeness_score(df)
    df = _apply_dtypes(df)
    log.info("  Clean EMS shape: %s", df.shape)
    return df


def preprocess_fire(path: Path = FIRE_CSV) -> pd.DataFrame:
    """Full preprocessing pipeline for Fire data."""
    log.info("--- Fire preprocessing ---")
    df = _read_csv(path, service_label="Fire")
    df = _parse_datetime(df)
    df = _clean_priority(df)
    df = _clean_description(df)
    df = _clean_geography(df)
    df = _clean_response_times(df)
    df = _ensure_required_columns(df, "Fire")
    df = _drop_duplicates(df)
    df = _compute_completeness_score(df)
    df = _apply_dtypes(df)
    log.info("  Clean Fire shape: %s", df.shape)
    return df


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(
    ems_path: Path = EMS_CSV,
    fire_path: Path = FIRE_CSV,
    output_path: Path = OUTPUT_PARQUET,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline for both EMS and Fire files,
    combine them, and write the result to Parquet.

    Parameters
    ----------
    ems_path   : Path to EMS_Data.csv
    fire_path  : Path to Fire_Data.csv
    output_path: Destination Parquet file
    dry_run    : If True, validate and return the DataFrame without writing

    Returns
    -------
    pd.DataFrame  –  the fully cleaned, combined dispatch table
    """
    frames: list[pd.DataFrame] = []

    # --- EMS ---
    if ems_path.exists():
        frames.append(preprocess_ems(ems_path))
    else:
        log.error("EMS file not found: %s — run scripts/download_data.py first.", ems_path)

    # --- Fire ---
    if fire_path.exists():
        frames.append(preprocess_fire(fire_path))
    else:
        log.error("Fire file not found: %s — run scripts/download_data.py first.", fire_path)

    if not frames:
        log.error("No data files found. Aborting.")
        sys.exit(1)

    # --- Combine ---
    log.info("Combining EMS and Fire frames …")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    log.info("  Combined shape: %s", combined.shape)

    # --- Validate required columns (pre-rename) ---
    missing = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing:
        log.error("Final DataFrame is missing required columns: %s", missing)
        sys.exit(1)

    # --- Report completeness ---
    avg_completeness = combined["completeness_score"].mean()
    log.info("  Average row completeness: %.1f%%", avg_completeness * 100)
    null_summary = combined[REQUIRED_COLUMNS].isna().sum()
    log.info("  Null counts per required column:\n%s", null_summary.to_string())

    # --- Rename columns to NEMSIS v3 canonical names ---
    # This ensures the Parquet uses the same column names expected by
    # contracts.py, schemas.py, and all downstream consumers (dashboard, ML).
    from config.contracts import COLUMN_MAPPING
    rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in combined.columns}
    combined = combined.rename(columns=rename_map)
    log.info("  Renamed %d columns to NEMSIS v3 canonical names.", len(rename_map))

    # --- Convert quarter from int (1–4) to string ("Q1"–"Q4") ---
    _quarter_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    if "quarter" in combined.columns:
        combined["quarter"] = (
            pd.to_numeric(combined["quarter"], errors="coerce")
            .map(_quarter_map)
            .astype("string")
        )
        log.info("  Converted quarter values to Q1–Q4 string format.")

    if dry_run:
        log.info("Dry-run mode: skipping Parquet write.")
        return combined

    # --- Write Parquet ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Writing clean Parquet to %s …", output_path)
    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(
        table,
        output_path,
        compression="snappy",
        row_group_size=500_000,
    )
    size_mb = output_path.stat().st_size / 1_048_576
    log.info("  Done. File size: %.1f MB", size_mb)
    log.info("  Rows written: %d", len(combined))

    return combined


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAlertAI — Phase 1 preprocessing pipeline (Greeshma)"
    )
    parser.add_argument(
        "--ems", type=Path, default=EMS_CSV,
        help="Path to EMS_Data.csv (default: data/raw/EMS_Data.csv)"
    )
    parser.add_argument(
        "--fire", type=Path, default=FIRE_CSV,
        help="Path to Fire_Data.csv (default: data/raw/Fire_Data.csv)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PARQUET,
        help="Output Parquet path (default: data/processed/fact_dispatch_clean.parquet)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and log stats without writing the Parquet file"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        ems_path=args.ems,
        fire_path=args.fire,
        output_path=args.output,
        dry_run=args.dry_run,
    )
