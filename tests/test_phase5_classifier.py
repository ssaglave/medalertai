"""
Phase 5 classifier evaluation tests.

Owner: Suvarna (C2)

Scope:
  - Confusion matrix artifact (named, square, sums match support)
  - Per-class metrics JSON (covers every label, well-formed entries)
  - Disagreement-flagging recall harness (logic on synthetic input,
    artifact sanity on the saved sweep)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import MODEL_ARTIFACTS_DIR
from src.models.classifier.disagreement_eval import (
    DEFAULT_THRESHOLDS,
    evaluate_flagging,
    flag_with_threshold,
    inject_label_errors,
)

CLASSIFIER_DIR = MODEL_ARTIFACTS_DIR / "classifier"
CM_PATH = CLASSIFIER_DIR / "confusion_matrix.parquet"
PER_CLASS_PATH = CLASSIFIER_DIR / "per_class_metrics.json"
METRICS_PATH = CLASSIFIER_DIR / "metrics.json"
DISAGREE_EVAL_PATH = CLASSIFIER_DIR / "disagreement_flagging_eval.json"


# ---------------------------------------------------------------------------
# Confusion matrix artifact
# ---------------------------------------------------------------------------
class TestConfusionMatrix:
    @pytest.mark.slow
    def test_confusion_matrix_artifact_exists(self):
        assert CM_PATH.exists(), f"Missing confusion matrix artifact: {CM_PATH}"

    @pytest.mark.slow
    def test_confusion_matrix_is_square_and_named(self):
        cm = pd.read_parquet(CM_PATH)
        # Stored as long-form: first column is the true class label,
        # remaining columns are the predicted classes.
        assert cm.columns[0] == "true_mpds_group"
        class_columns = list(cm.columns[1:])
        assert len(cm) == len(class_columns), (
            f"Confusion matrix is not square: {len(cm)} rows vs {len(class_columns)} pred cols"
        )
        assert sorted(cm["true_mpds_group"].tolist()) == sorted(class_columns), (
            "Row labels must match column labels"
        )

    @pytest.mark.slow
    def test_confusion_matrix_row_sums_match_per_class_support(self):
        cm = pd.read_parquet(CM_PATH).set_index("true_mpds_group")
        per_class = json.loads(PER_CLASS_PATH.read_text(encoding="utf-8"))

        for cls, stats in per_class.items():
            row_sum = int(cm.loc[cls].sum())
            assert row_sum == stats["support"], (
                f"Row sum for {cls!r} ({row_sum}) != per-class support ({stats['support']})"
            )


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------
class TestPerClassMetrics:
    @pytest.mark.slow
    def test_per_class_metrics_exists(self):
        assert PER_CLASS_PATH.exists(), f"Missing per-class metrics: {PER_CLASS_PATH}"

    @pytest.mark.slow
    def test_covers_all_label_map_classes(self):
        per_class = json.loads(PER_CLASS_PATH.read_text(encoding="utf-8"))
        label_map = pd.read_parquet(CLASSIFIER_DIR / "label_map.parquet")
        expected = set(label_map["mpds_group"].tolist())
        assert set(per_class.keys()) == expected, (
            f"per_class_metrics.json missing classes: {expected - set(per_class.keys())}"
        )

    @pytest.mark.slow
    def test_each_entry_well_formed(self):
        per_class = json.loads(PER_CLASS_PATH.read_text(encoding="utf-8"))
        required = {"precision", "recall", "f1", "support"}
        for cls, stats in per_class.items():
            assert required.issubset(stats.keys()), f"{cls!r} missing keys"
            assert 0.0 <= stats["precision"] <= 1.0
            assert 0.0 <= stats["recall"] <= 1.0
            assert 0.0 <= stats["f1"] <= 1.0
            assert stats["support"] >= 0

    @pytest.mark.slow
    def test_metrics_json_has_test_per_class_block(self):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        assert "test_per_class" in metrics
        assert "validation_per_class" in metrics
        assert len(metrics["test_per_class"]) == metrics["n_classes"]


# ---------------------------------------------------------------------------
# Macro-F1 regression floor
# ---------------------------------------------------------------------------
# Plan target is 0.55 (revised down from 0.75 because the WPRDC dataset has
# no hour-of-day timestamps and `call_type` must be excluded for leakage —
# the achievable ceiling on dispatch metadata alone is ~0.45). This floor
# guards against further regressions while leaving headroom toward the
# revised plan target. Raise it whenever the saved metrics improve.
F1_REGRESSION_FLOOR = 0.42
F1_PLAN_TARGET = 0.55


class TestMacroF1Floor:
    @pytest.mark.slow
    def test_test_macro_f1_above_regression_floor(self):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        f1 = metrics["test"]["macro_f1"]
        assert f1 >= F1_REGRESSION_FLOOR, (
            f"Test macro F1 {f1:.4f} dropped below regression floor "
            f"{F1_REGRESSION_FLOOR:.2f}. Plan target is {F1_PLAN_TARGET:.2f}."
        )


# ---------------------------------------------------------------------------
# Disagreement-flag harness — pure logic on synthetic data
# ---------------------------------------------------------------------------
class TestDisagreementHarnessLogic:
    def test_inject_label_errors_perturbs_requested_fraction(self):
        y_true = np.arange(1000) % 5  # 5 classes
        perturbed, mask = inject_label_errors(y_true, n_classes=5, frac=0.1, seed=0)

        assert mask.sum() == 100
        # Perturbed positions must differ from original
        assert (perturbed[mask] != y_true[mask]).all()
        # Untouched positions must match original
        assert (perturbed[~mask] == y_true[~mask]).all()

    def test_inject_label_errors_is_seeded(self):
        y_true = np.arange(500) % 3
        a, mask_a = inject_label_errors(y_true, n_classes=3, frac=0.2, seed=7)
        b, mask_b = inject_label_errors(y_true, n_classes=3, frac=0.2, seed=7)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mask_a, mask_b)

    def test_inject_label_errors_rejects_invalid_frac(self):
        with pytest.raises(ValueError):
            inject_label_errors(np.array([0, 1, 2]), n_classes=3, frac=0.0)
        with pytest.raises(ValueError):
            inject_label_errors(np.array([0, 1, 2]), n_classes=3, frac=1.0)

    def test_flag_with_threshold_requires_confidence_and_mismatch(self):
        y_pred = np.array([0, 1, 0, 1])
        y_label = np.array([0, 0, 1, 1])
        conf = np.array([0.9, 0.9, 0.5, 0.95])

        flagged = flag_with_threshold(y_pred, y_label, conf, threshold=0.7)
        # Position 0: match → not flagged
        # Position 1: mismatch + confident → flagged
        # Position 2: mismatch but not confident → not flagged
        # Position 3: match → not flagged
        np.testing.assert_array_equal(flagged, [False, True, False, False])

    def test_evaluate_flagging_recall_drops_with_threshold(self):
        # Build a controlled scenario with known recall behavior:
        # 4 perturbed rows, 6 unperturbed; model is "right" with varying conf.
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_perturbed = np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0])
        mask = np.array([True, True, True, True, False, False, False, False, False, False])
        conf = np.array([0.55, 0.65, 0.75, 0.85, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])

        sweep = evaluate_flagging(y_pred, conf, y_perturbed, mask, thresholds=(0.5, 0.7, 0.9))
        recalls = [row["recall_on_injected"] for row in sweep]

        assert recalls[0] >= recalls[1] >= recalls[2], (
            f"Recall should be non-increasing across thresholds; got {recalls}"
        )
        assert recalls[0] == 1.0
        # All unperturbed rows match y_pred=0, so false alarms must be 0.
        assert all(row["false_alarm_rate"] == 0.0 for row in sweep)


# ---------------------------------------------------------------------------
# Disagreement-flag harness — saved artifact sanity
# ---------------------------------------------------------------------------
class TestDisagreementEvalArtifact:
    @pytest.mark.slow
    def test_artifact_exists(self):
        assert DISAGREE_EVAL_PATH.exists(), (
            f"Missing harness output: {DISAGREE_EVAL_PATH}. "
            "Run `python -m src.models.classifier.disagreement_eval`."
        )

    @pytest.mark.slow
    def test_artifact_has_expected_top_level_keys(self):
        report = json.loads(DISAGREE_EVAL_PATH.read_text(encoding="utf-8"))
        for key in ("test_split_rows", "n_classes", "perturb_frac",
                    "seed", "thresholds", "sweep", "default_threshold", "default"):
            assert key in report, f"Missing key in harness report: {key}"

    @pytest.mark.slow
    def test_artifact_sweep_covers_all_default_thresholds(self):
        report = json.loads(DISAGREE_EVAL_PATH.read_text(encoding="utf-8"))
        sweep_thresholds = sorted(round(row["threshold"], 2) for row in report["sweep"])
        expected = sorted(round(t, 2) for t in DEFAULT_THRESHOLDS)
        assert sweep_thresholds == expected

    @pytest.mark.slow
    def test_artifact_recall_is_monotonic_non_increasing(self):
        report = json.loads(DISAGREE_EVAL_PATH.read_text(encoding="utf-8"))
        ordered = sorted(report["sweep"], key=lambda r: r["threshold"])
        recalls = [row["recall_on_injected"] for row in ordered]
        for lo, hi in zip(recalls, recalls[1:]):
            assert hi <= lo + 1e-6, f"Recall not non-increasing across thresholds: {recalls}"

    @pytest.mark.slow
    def test_artifact_default_recall_above_floor(self):
        """Sanity floor — flag must catch at least 10% of injected errors at thr=0.7.

        This is a low bar deliberately: macro-F1 is below target, so model
        confidence on correct predictions is limited. The bar should rise
        once classifier improvements land in step 4.
        """
        report = json.loads(DISAGREE_EVAL_PATH.read_text(encoding="utf-8"))
        default = report["default"]
        assert default is not None, "default operating point not recorded"
        assert default["recall_on_injected"] >= 0.10, (
            f"Disagreement-flag recall at thr=0.7 = {default['recall_on_injected']:.4f}, "
            "below the 0.10 sanity floor"
        )
