"""
src/data/splits.py

Phase 2 - Training splits + feature set contracts
Owner: Greeshma

Reads the clean Parquet produced in Phase 1, defines the feature contracts
for the three ML tracks, and writes train/validation/test split files under
data/processed/splits/.

Usage from the repo root:
    python -m src.data.splits
    python -m src.data.splits --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.splits")


INPUT_PARQUET = PROCESSED_DATA_DIR / "fact_dispatch_clean.parquet"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
MIN_CLASSIFIER_LABEL_COUNT = 8
RARE_CLASS_LABEL = "Other / Rare Call Type"


# Phase 2 canonical column names. These match config/contracts.py and the
# current fact_dispatch_clean.parquet output.
CLASSIFIER_FEATURE_COLS: list[str] = [
    "priority_code",
    "priority_description",
    "quarter",
    "year",
    "city_code",
    "service_type",
    "longitude",
    "latitude",
    "completeness_score",
]
CLASSIFIER_TARGET_COL = "call_type"

FORECASTER_FEATURE_COLS: list[str] = [
    "year",
    "quarter",
    "service_type",
    "city_code",
    "priority_code",
]
FORECASTER_TARGET_COL = "call_count"

CLUSTERING_FEATURE_COLS: list[str] = [
    "longitude",
    "latitude",
    "quarter",
    "service_type",
    "priority_code",
]


# Compatibility aliases let the script work with both the Phase 1 raw-style
# columns and the canonical schema in config/contracts.py.
COLUMN_ALIASES: dict[str, list[str]] = {
    "priority_code": ["priority_code", "priority", "PRIORITY"],
    "priority_description": ["priority_description", "priority_desc", "PRIORITY_DESC"],
    "quarter": ["quarter", "CALL_QUARTER", "call_quarter"],
    "year": ["year", "CALL_YEAR", "call_year"],
    "call_type": ["call_type", "description_short", "DESCRIPTION_SHORT"],
    "service_type": ["service_type", "SERVICE", "service"],
    "longitude": ["longitude", "census_block_group_center__x"],
    "latitude": ["latitude", "census_block_group_center__y"],
    "city_code": ["city_code", "CITY_CODE"],
    "city_name": ["city_name", "CITY_NAME"],
    "completeness_score": ["completeness_score"],
}


def _load_clean_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.error("Clean Parquet not found: %s", path)
        log.error("Run Phase 1 first: python -m src.data.preprocessing")
        sys.exit(1)

    log.info("Loading clean Parquet from %s", path)
    df = pd.read_parquet(path)
    log.info("Loaded shape: %s", df.shape)
    return df


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Phase 2 canonical column names filled from aliases."""
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias == canonical:
                continue
            if alias in out.columns:
                if canonical in out.columns:
                    missing = out[canonical].isna()
                    if missing.any():
                        out.loc[missing, canonical] = out.loc[missing, alias].to_numpy()
                else:
                    out[canonical] = out[alias]

    if "quarter" in out.columns:
        out["quarter"] = out["quarter"].astype("string").str.upper().str.replace("QUARTER ", "Q", regex=False)
        numeric_quarter = pd.to_numeric(out["quarter"], errors="coerce")
        out.loc[numeric_quarter.notna(), "quarter"] = "Q" + numeric_quarter.dropna().astype("int64").astype(str)

    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int16")

    return out


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    size_kb = path.stat().st_size / 1024
    log.info("Wrote %s (%d rows, %.1f KB)", _display_path(path), len(df), size_kb)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("Wrote %s", _display_path(path))


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _log_split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label: str) -> None:
    total = len(train) + len(val) + len(test)
    if total == 0:
        log.warning("[%s] No rows available for splitting.", label)
        return
    log.info(
        "[%s] train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        label,
        len(train),
        len(train) / total * 100,
        len(val),
        len(val) / total * 100,
        len(test),
        len(test) / total * 100,
    )


def _safe_three_way_split(
    df: pd.DataFrame,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df[stratify_col] if stratify_col else None
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=VAL_RATIO + TEST_RATIO,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
    except ValueError as exc:
        log.warning("Falling back to random train split because stratification failed: %s", exc)
        train_df, temp_df = train_test_split(
            df,
            test_size=VAL_RATIO + TEST_RATIO,
            random_state=RANDOM_SEED,
        )

    temp_stratify = temp_df[stratify_col] if stratify_col else None
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
            stratify=temp_stratify,
        )
    except ValueError as exc:
        log.warning("Falling back to random validation/test split because stratification failed: %s", exc)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
        )
    return train_df, val_df, test_df


def build_classifier_splits(df: pd.DataFrame, out_dir: Path) -> None:
    log.info("Building classifier splits")
    df_model = _canonicalize_columns(df)
    _require_columns(df_model, CLASSIFIER_FEATURE_COLS + [CLASSIFIER_TARGET_COL], "classifier")

    keep_cols = CLASSIFIER_FEATURE_COLS + [CLASSIFIER_TARGET_COL]
    df_model = df_model[keep_cols].dropna(subset=[CLASSIFIER_TARGET_COL]).copy()
    df_model[CLASSIFIER_TARGET_COL] = df_model[CLASSIFIER_TARGET_COL].astype("string").str.strip()
    df_model = df_model[df_model[CLASSIFIER_TARGET_COL] != ""].copy()

    counts = df_model[CLASSIFIER_TARGET_COL].value_counts()
    rare_labels = counts[counts < MIN_CLASSIFIER_LABEL_COUNT].index
    if len(rare_labels) > 0:
        rare_rows = df_model[CLASSIFIER_TARGET_COL].isin(rare_labels).sum()
        log.info("Grouping %d rare labels (%d rows) into '%s'", len(rare_labels), rare_rows, RARE_CLASS_LABEL)
        df_model.loc[df_model[CLASSIFIER_TARGET_COL].isin(rare_labels), CLASSIFIER_TARGET_COL] = RARE_CLASS_LABEL

    categories = sorted(df_model[CLASSIFIER_TARGET_COL].dropna().unique().tolist())
    label_to_code = {label: idx for idx, label in enumerate(categories)}
    df_model["label_code"] = df_model[CLASSIFIER_TARGET_COL].map(label_to_code).astype("int16")

    train_df, val_df, test_df = _safe_three_way_split(df_model, stratify_col="label_code")
    _log_split_summary(train_df, val_df, test_df, "classifier")

    split_dir = out_dir / "classifier"
    _write_split(train_df, split_dir / "train.parquet")
    _write_split(val_df, split_dir / "val.parquet")
    _write_split(test_df, split_dir / "test.parquet")

    label_map = pd.DataFrame(
        [{"label_code": code, CLASSIFIER_TARGET_COL: label} for label, code in label_to_code.items()]
    ).sort_values("label_code")
    _write_split(label_map, split_dir / "label_map.parquet")


def build_forecaster_splits(df: pd.DataFrame, out_dir: Path) -> None:
    log.info("Building forecaster splits")
    df_model = _canonicalize_columns(df)
    _require_columns(df_model, FORECASTER_FEATURE_COLS, "forecaster")

    df_model = df_model[FORECASTER_FEATURE_COLS].dropna(subset=["year", "quarter"]).copy()
    df_agg = (
        df_model.groupby(FORECASTER_FEATURE_COLS, observed=True)
        .size()
        .reset_index(name=FORECASTER_TARGET_COL)
        .sort_values(["year", "quarter", "service_type", "city_code", "priority_code"])
        .reset_index(drop=True)
    )
    log.info("Aggregated forecaster shape: %s", df_agg.shape)

    periods = (
        df_agg[["year", "quarter"]]
        .drop_duplicates()
        .sort_values(["year", "quarter"])
        .apply(tuple, axis=1)
        .tolist()
    )
    if len(periods) >= 4:
        val_period = periods[-2]
        test_period = periods[-1]
        val_mask = (df_agg["year"] == val_period[0]) & (df_agg["quarter"] == val_period[1])
        test_mask = (df_agg["year"] == test_period[0]) & (df_agg["quarter"] == test_period[1])
        train_df = df_agg[~val_mask & ~test_mask].copy()
        val_df = df_agg[val_mask].copy()
        test_df = df_agg[test_mask].copy()
    else:
        log.warning("Too few periods for chronological split; falling back to random split.")
        train_df, val_df, test_df = _safe_three_way_split(df_agg)

    _log_split_summary(train_df, val_df, test_df, "forecaster")
    split_dir = out_dir / "forecaster"
    _write_split(train_df, split_dir / "train.parquet")
    _write_split(val_df, split_dir / "val.parquet")
    _write_split(test_df, split_dir / "test.parquet")


def build_clustering_splits(df: pd.DataFrame, out_dir: Path) -> None:
    log.info("Building clustering splits")
    df_model = _canonicalize_columns(df)
    _require_columns(df_model, CLUSTERING_FEATURE_COLS, "clustering")

    df_model = df_model[CLUSTERING_FEATURE_COLS].dropna(subset=["longitude", "latitude"]).copy()
    fit_df, eval_df = train_test_split(df_model, test_size=0.20, random_state=RANDOM_SEED)
    log.info("[clustering] fit=%d, eval=%d", len(fit_df), len(eval_df))

    split_dir = out_dir / "clustering"
    _write_split(fit_df, split_dir / "fit.parquet")
    _write_split(eval_df, split_dir / "eval.parquet")


def export_feature_contracts(out_dir: Path) -> None:
    contracts_path = out_dir / "feature_contracts.md"
    lines = [
        "# MedAlertAI - Feature Set Contracts (Phase 2)",
        "",
        "Owner: Greeshma",
        "Generated by `python -m src.data.splits`.",
        "",
        "Do not change these contracts without reviewing with the Phase 2 model owners.",
        "",
        "## MPDS Classifier",
        "",
        f"Target column: `{CLASSIFIER_TARGET_COL}`",
        "",
        "| Feature | Description |",
        "|---|---|",
        "| `priority_code` | Dispatch priority code. |",
        "| `priority_description` | Human-readable priority description. |",
        "| `quarter` | Quarter label (`Q1` through `Q4`). |",
        "| `year` | Dispatch year. |",
        "| `city_code` | Municipality code. |",
        "| `service_type` | EMS or Fire. |",
        "| `longitude` | Incident/block-group longitude. |",
        "| `latitude` | Incident/block-group latitude. |",
        "| `completeness_score` | Phase 1 row completeness score. |",
        "",
        "Files: `data/processed/splits/classifier/train.parquet`, `val.parquet`, `test.parquet`, `label_map.parquet`.",
        "",
        "## Demand Forecaster",
        "",
        f"Target column: `{FORECASTER_TARGET_COL}` after aggregation.",
        "",
        "| Feature | Description |",
        "|---|---|",
        "| `year` | Dispatch year. |",
        "| `quarter` | Quarter label (`Q1` through `Q4`). |",
        "| `service_type` | EMS or Fire. |",
        "| `city_code` | Municipality code. |",
        "| `priority_code` | Dispatch priority code. |",
        "",
        "Files: `data/processed/splits/forecaster/train.parquet`, `val.parquet`, `test.parquet`.",
        "",
        "## Hotspot Clustering",
        "",
        "Target column: none; this is an unsupervised dataset.",
        "",
        "| Feature | Description |",
        "|---|---|",
        "| `longitude` | Incident/block-group longitude. |",
        "| `latitude` | Incident/block-group latitude. |",
        "| `quarter` | Quarter label (`Q1` through `Q4`). |",
        "| `service_type` | EMS or Fire. |",
        "| `priority_code` | Dispatch priority code. |",
        "",
        "Files: `data/processed/splits/clustering/fit.parquet`, `eval.parquet`.",
        "",
    ]
    contracts_path.parent.mkdir(parents=True, exist_ok=True)
    contracts_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", _display_path(contracts_path))


def export_split_manifest(out_dir: Path) -> None:
    manifest = {
        "owner": "Greeshma",
        "phase": "Phase 2 - ML Models",
        "input": str(INPUT_PARQUET.relative_to(PROJECT_ROOT)),
        "random_seed": RANDOM_SEED,
        "ratios": {"train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO},
        "classifier": {
            "features": CLASSIFIER_FEATURE_COLS,
            "target": CLASSIFIER_TARGET_COL,
            "rare_label_policy": {
                "minimum_rows": MIN_CLASSIFIER_LABEL_COUNT,
                "replacement_label": RARE_CLASS_LABEL,
            },
        },
        "forecaster": {
            "features": FORECASTER_FEATURE_COLS,
            "target": FORECASTER_TARGET_COL,
            "split_policy": "chronological by year and quarter; final period is test, previous period is validation",
        },
        "clustering": {
            "features": CLUSTERING_FEATURE_COLS,
            "target": None,
            "split_policy": "80/20 random fit/evaluation split",
        },
    }
    _write_json(manifest, out_dir / "split_manifest.json")


def run_splits(
    input_path: Path = INPUT_PARQUET,
    output_dir: Path = SPLITS_DIR,
    dry_run: bool = False,
) -> None:
    df = _canonicalize_columns(_load_clean_parquet(input_path))
    required = sorted(
        set(CLASSIFIER_FEATURE_COLS + [CLASSIFIER_TARGET_COL] + FORECASTER_FEATURE_COLS + CLUSTERING_FEATURE_COLS)
    )
    _require_columns(df, required, "phase 2 splits")

    if dry_run:
        log.info("Dry-run complete. Required Phase 2 columns are present.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    build_classifier_splits(df, output_dir)
    build_forecaster_splits(df, output_dir)
    build_clustering_splits(df, output_dir)
    export_feature_contracts(output_dir)
    export_split_manifest(output_dir)
    log.info("All Phase 2 split artifacts written to %s", output_dir)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedAlertAI Phase 2 training splits")
    parser.add_argument("--input", type=Path, default=INPUT_PARQUET)
    parser.add_argument("--output", type=Path, default=SPLITS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_splits(input_path=args.input, output_dir=args.output, dry_run=args.dry_run)
