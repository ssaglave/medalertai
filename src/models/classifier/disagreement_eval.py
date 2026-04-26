"""
disagreement_eval.py — Recall harness for the MPDS disagreement flag.

Owner: Suvarna (C2)
Phase: 5

The classifier's `flag_disagreements()` mechanism marks a row when the model
predicts MPDS class A with confidence > threshold but the row's label is B.
On its own, that count tells us nothing about whether the *flag* is any good
at catching real labeling errors.

This module measures that. We:
  1. Take the test split as ground truth (assume its labels are correct).
  2. Inject a known fraction of synthetic label errors (perturb mpds_group to
     a random *different* class).
  3. Run the flag and compute, for a sweep of confidence thresholds:
       - recall  = perturbed rows flagged / perturbed rows
       - false alarm rate = unperturbed rows flagged / unperturbed rows
       - precision (proxy) = perturbed rows flagged / total rows flagged

The result is persisted as `disagreement_flagging_eval.json` for the QA tab
and the Phase 5 tests.

Usage (from repo root):
    python -m src.models.classifier.disagreement_eval
    python -m src.models.classifier.disagreement_eval --frac 0.05 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.classifier.disagreement_eval")

CLASSIFIER_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "classifier"
TEST_SPLIT_PATH = PROCESSED_DATA_DIR / "splits" / "classifier" / "test.parquet"

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_PERTURB_FRAC = 0.05


def inject_label_errors(
    y_true_codes: np.ndarray,
    n_classes: int,
    frac: float = DEFAULT_PERTURB_FRAC,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a random `frac` of label codes to a different class.

    Returns (perturbed_codes, perturbed_mask) where perturbed_mask is a
    boolean array marking which rows had their label swapped.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1); got {frac}")
    if n_classes < 2:
        raise ValueError("Need >= 2 classes to perturb labels")

    rng = np.random.default_rng(seed)
    n = len(y_true_codes)
    n_perturb = max(1, int(round(frac * n)))

    perturb_idx = rng.choice(n, size=n_perturb, replace=False)
    perturbed = y_true_codes.copy()

    for i in perturb_idx:
        original = perturbed[i]
        # Pick a different class uniformly at random
        new_label = rng.integers(0, n_classes)
        while new_label == original:
            new_label = rng.integers(0, n_classes)
        perturbed[i] = new_label

    perturbed_mask = np.zeros(n, dtype=bool)
    perturbed_mask[perturb_idx] = True
    return perturbed, perturbed_mask


def flag_with_threshold(
    y_pred: np.ndarray,
    y_label: np.ndarray,
    max_confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Replicate `flag_disagreements` flag computation.

    A row is flagged when the model is confident (max_confidence > threshold)
    AND the prediction differs from the (possibly-perturbed) label.
    """
    return (max_confidence > threshold) & (y_pred != y_label)


def evaluate_flagging(
    y_pred: np.ndarray,
    max_confidence: np.ndarray,
    y_perturbed_label: np.ndarray,
    perturbed_mask: np.ndarray,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> list[dict]:
    """Sweep thresholds and compute recall / false-alarm / precision-proxy."""
    n_perturbed = int(perturbed_mask.sum())
    n_unperturbed = int((~perturbed_mask).sum())
    rows = []

    for thr in thresholds:
        flagged = flag_with_threshold(y_pred, y_perturbed_label, max_confidence, thr)
        flagged_perturbed = int((flagged & perturbed_mask).sum())
        flagged_unperturbed = int((flagged & ~perturbed_mask).sum())
        n_flagged_total = flagged_perturbed + flagged_unperturbed

        recall = flagged_perturbed / n_perturbed if n_perturbed else 0.0
        false_alarm = (
            flagged_unperturbed / n_unperturbed if n_unperturbed else 0.0
        )
        precision = (
            flagged_perturbed / n_flagged_total if n_flagged_total else 0.0
        )

        rows.append({
            "threshold": round(thr, 2),
            "n_perturbed": n_perturbed,
            "n_unperturbed": n_unperturbed,
            "n_flagged_total": n_flagged_total,
            "n_flagged_perturbed": flagged_perturbed,
            "n_flagged_unperturbed": flagged_unperturbed,
            "recall_on_injected": round(recall, 4),
            "false_alarm_rate": round(false_alarm, 4),
            "precision_proxy": round(precision, 4),
        })

    return rows


def _load_artifacts() -> tuple[object, object, pd.DataFrame]:
    """Load the saved pipeline, label encoder, and label map."""
    pipeline_path = CLASSIFIER_ARTIFACTS_DIR / "classifier_pipeline.joblib"
    le_path = CLASSIFIER_ARTIFACTS_DIR / "label_encoder.joblib"
    label_map_path = CLASSIFIER_ARTIFACTS_DIR / "label_map.parquet"

    for path in (pipeline_path, le_path, label_map_path):
        if not path.exists():
            log.error("Artifact not found: %s", path)
            log.error("Run `python -m src.models.classifier.train` first.")
            sys.exit(1)

    pipeline = joblib.load(pipeline_path)
    label_encoder = joblib.load(le_path)
    label_map = pd.read_parquet(label_map_path)
    log.info("Loaded pipeline, label encoder, and %d-class label map", len(label_map))
    return pipeline, label_encoder, label_map


def _prepare_test_features(
    test_df: pd.DataFrame,
    label_encoder,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the MPDS mapper, encode labels, return feature DF + true codes.

    If the model was trained with a collapsed label set (top-N + Other), any
    raw MPDS class outside the encoder's known classes is mapped to 'Other'
    so the harness can run against the same label space the model saw.
    """
    from src.models.classifier.train import (
        ALL_FEATURES,
        CATEGORICAL_FEATURES,
        OTHER_LABEL,
        TARGET_COL,
        add_mpds_target,
    )

    df = add_mpds_target(test_df.copy())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    known_classes = set(label_encoder.classes_)
    if OTHER_LABEL in known_classes:
        df[TARGET_COL] = df[TARGET_COL].where(
            df[TARGET_COL].isin(known_classes), OTHER_LABEL,
        )

    y_codes = label_encoder.transform(df[TARGET_COL]).astype(np.int32)
    return df[ALL_FEATURES], y_codes


def run_disagreement_evaluation(
    frac: float = DEFAULT_PERTURB_FRAC,
    seed: int = 42,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    output_path: Optional[Path] = None,
) -> dict:
    """End-to-end harness: load artifacts, inject errors, sweep thresholds, save JSON."""
    start = time.time()
    log.info("=== Disagreement-flagging recall harness ===")

    if not TEST_SPLIT_PATH.exists():
        log.error("Test split not found: %s", TEST_SPLIT_PATH)
        sys.exit(1)

    pipeline, label_encoder, label_map = _load_artifacts()
    test_df = pd.read_parquet(TEST_SPLIT_PATH)
    log.info("Loaded test split — %d rows", len(test_df))

    X_test_df, y_true_codes = _prepare_test_features(test_df, label_encoder)

    log.info("Running pipeline.predict / predict_proba on %d rows …", len(X_test_df))
    y_pred = pipeline.predict(X_test_df)
    y_proba = pipeline.predict_proba(X_test_df)
    max_confidence = y_proba.max(axis=1)

    n_classes = len(label_map)
    y_perturbed, perturbed_mask = inject_label_errors(
        y_true_codes, n_classes=n_classes, frac=frac, seed=seed,
    )
    log.info(
        "Injected %d label perturbations (frac=%.3f, seed=%d)",
        int(perturbed_mask.sum()), frac, seed,
    )

    sweep = evaluate_flagging(
        y_pred=y_pred,
        max_confidence=max_confidence,
        y_perturbed_label=y_perturbed,
        perturbed_mask=perturbed_mask,
        thresholds=thresholds,
    )

    # Default operating point used by `flag_disagreements`
    default_threshold = 0.7
    default_row = next(
        (row for row in sweep if abs(row["threshold"] - default_threshold) < 1e-9),
        None,
    )

    report = {
        "test_split_rows": int(len(test_df)),
        "n_classes": int(n_classes),
        "perturb_frac": frac,
        "seed": seed,
        "thresholds": list(thresholds),
        "sweep": sweep,
        "default_threshold": default_threshold,
        "default": default_row,
        "duration_sec": round(time.time() - start, 1),
    }

    if output_path is None:
        output_path = CLASSIFIER_ARTIFACTS_DIR / "disagreement_flagging_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("Saved harness report: %s", output_path.relative_to(PROJECT_ROOT))

    log.info("Threshold sweep:")
    log.info("  %-9s %-8s %-12s %-9s", "thr", "recall", "false_alarm", "precision")
    for row in sweep:
        log.info(
            "  %-9.2f %-8.4f %-12.4f %-9.4f",
            row["threshold"],
            row["recall_on_injected"],
            row["false_alarm_rate"],
            row["precision_proxy"],
        )
    return report


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAlertAI — Phase 5 disagreement-flag recall harness (Suvarna)"
    )
    parser.add_argument(
        "--frac", type=float, default=DEFAULT_PERTURB_FRAC,
        help=f"Fraction of test rows to perturb (default: {DEFAULT_PERTURB_FRAC})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible perturbations (default: 42)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Confidence thresholds to sweep (default: 0.5 0.6 0.7 0.8 0.9)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_disagreement_evaluation(
        frac=args.frac,
        seed=args.seed,
        thresholds=tuple(args.thresholds),
    )
