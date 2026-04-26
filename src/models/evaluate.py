"""
evaluate.py — Unified model evaluation harness.

Owner: Deekshitha (C5)
Phase: 2

Responsibilities:
  - MLflow tracking instrumentation for all 3 models
  - Metric evaluation (F1, MAPE, Silhouette, Recall@20)
  - CI test stubs for each model
  - Unified `run_evaluation()` entry point that loads serialized
    artifacts from `models/artifacts/` and verifies Phase 2 targets.

Phase 2 Metric Targets:
  ┌────────────────┬──────────────────────────────┐
  │ Model          │ Target                       │
  ├────────────────┼──────────────────────────────┤
  │ Classifier     │ Macro F1 > 0.55              │
  │ Forecaster     │ MAPE < 15 %                  │
  │ Clustering     │ Silhouette > 0.4             │
  │ Anomaly        │ Recall@20 > 0.7              │
  └────────────────┴──────────────────────────────┘

Dependencies:
  - C2 (Suvarna):   models/artifacts/classifier/metrics.json
  - C3 (Sanika):    models/artifacts/forecasting/metrics.json
  - C3 (Sanika):    models/artifacts/clustering/metrics.json
  - C4 (Srileakhana): models/artifacts/forecasting/lightgbm_metadata.json
  - C1 (Greeshma):  data/processed/splits/ (for re-evaluation if needed)

Usage (from repo root):
    python -m src.models.evaluate
    python -m src.models.evaluate --no-mlflow
    python -m src.models.evaluate --model classifier
    python -m src.models.evaluate --model forecasting
    python -m src.models.evaluate --model clustering
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.evaluate")


# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
CLASSIFIER_ARTIFACTS = MODEL_ARTIFACTS_DIR / "classifier"
FORECASTING_ARTIFACTS = MODEL_ARTIFACTS_DIR / "forecasting"
CLUSTERING_ARTIFACTS = MODEL_ARTIFACTS_DIR / "clustering"

MLFLOW_EXPERIMENT = "medalertai-evaluation"


# ---------------------------------------------------------------------------
# Metric Targets (Phase 2)
# ---------------------------------------------------------------------------
class MetricTarget(Enum):
    """Phase 2 evaluation metric targets from implementation_plan.md."""

    CLASSIFIER_MACRO_F1 = ("macro_f1", 0.55, "greater")
    FORECASTER_MAPE = ("avg_mape", 0.15, "less")
    CLUSTERING_SILHOUETTE = ("dbscan_silhouette", 0.4, "greater")
    ANOMALY_RECALL_20 = ("iso_forest_recall_20", 0.7, "greater")

    def __init__(self, metric_key: str, threshold: float, direction: str):
        self.metric_key = metric_key
        self.threshold = threshold
        self.direction = direction

    def evaluate(self, value: float) -> bool:
        """Return True if the metric meets or exceeds the target."""
        if self.direction == "greater":
            return value >= self.threshold
        return value < self.threshold

    @property
    def description(self) -> str:
        op = ">" if self.direction == "greater" else "<"
        return f"{self.metric_key} {op} {self.threshold}"


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    """Container for a single model's evaluation outcome."""

    model_name: str
    metrics: dict = field(default_factory=dict)
    targets: dict = field(default_factory=dict)  # metric_key -> (value, threshold, passed)
    passed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "targets": {
                k: {"value": v[0], "threshold": v[1], "passed": v[2]}
                for k, v in self.targets.items()
            },
            "passed": self.passed,
            "error": self.error,
        }


# ===========================================================================
# Classifier Evaluation
# ===========================================================================

def evaluate_classifier() -> EvaluationResult:
    """
    Evaluate the MPDS classifier (C2 — Suvarna).

    Reads serialized metrics from `models/artifacts/classifier/metrics.json`
    and optionally re-evaluates from the saved pipeline on the test split.

    Target: macro F1 > 0.55
    """
    result = EvaluationResult(model_name="classifier")
    metrics_path = CLASSIFIER_ARTIFACTS / "metrics.json"

    if not metrics_path.exists():
        result.error = f"Classifier metrics not found: {metrics_path}"
        log.error(result.error)
        return result

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result.metrics = metrics

        # Extract test-set macro F1 (the primary target)
        test_metrics = metrics.get("test", {})
        macro_f1 = test_metrics.get("macro_f1", 0.0)

        target = MetricTarget.CLASSIFIER_MACRO_F1
        passed = target.evaluate(macro_f1)
        result.targets[target.metric_key] = (macro_f1, target.threshold, passed)

        # Also capture supplementary metrics
        for key in ("accuracy", "weighted_f1", "macro_precision", "macro_recall"):
            val = test_metrics.get(key)
            if val is not None:
                result.targets[key] = (val, None, True)  # informational only

        result.passed = passed

        _log_evaluation_result("Classifier", target, macro_f1, passed)

        # Try to load and report disagreement stats
        disagree_path = CLASSIFIER_ARTIFACTS / "disagreements.parquet"
        if disagree_path.exists():
            disagree_df = pd.read_parquet(disagree_path)
            n_flagged = disagree_df["is_disagreement"].sum() if "is_disagreement" in disagree_df.columns else 0
            result.metrics["n_disagreements"] = int(n_flagged)
            result.metrics["n_test_rows"] = len(disagree_df)
            log.info("  Disagreements flagged: %d / %d test rows", n_flagged, len(disagree_df))

    except Exception as exc:
        result.error = f"Classifier evaluation failed: {exc}"
        log.error(result.error)

    return result


def evaluate_classifier_from_artifacts() -> EvaluationResult:
    """
    Re-evaluate classifier from the saved pipeline & test split.
    This is the 'live' test stub used in CI.
    """
    result = EvaluationResult(model_name="classifier_live")

    try:
        import joblib
        from sklearn.metrics import f1_score, accuracy_score

        pipeline_path = CLASSIFIER_ARTIFACTS / "classifier_pipeline.joblib"
        test_path = PROCESSED_DATA_DIR / "splits" / "classifier" / "test.parquet"

        if not pipeline_path.exists():
            result.error = f"Pipeline not found: {pipeline_path}"
            log.error(result.error)
            return result

        if not test_path.exists():
            result.error = f"Test split not found: {test_path}"
            log.error(result.error)
            return result

        pipeline = joblib.load(pipeline_path)
        test_df = pd.read_parquet(test_path)

        # Use the same feature list from the classifier module
        from src.models.classifier.train import ALL_FEATURES, LABEL_CODE_COL, CATEGORICAL_FEATURES

        for col in CATEGORICAL_FEATURES:
            if col in test_df.columns:
                test_df[col] = test_df[col].astype(str)

        X_test = test_df[ALL_FEATURES]
        y_test = test_df[LABEL_CODE_COL].values.astype(np.int32)

        y_pred = pipeline.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)

        result.metrics = {"macro_f1": round(macro_f1, 4), "accuracy": round(accuracy, 4)}

        target = MetricTarget.CLASSIFIER_MACRO_F1
        passed = target.evaluate(macro_f1)
        result.targets[target.metric_key] = (macro_f1, target.threshold, passed)
        result.passed = passed

        _log_evaluation_result("Classifier (live)", target, macro_f1, passed)

    except Exception as exc:
        result.error = f"Live classifier evaluation failed: {exc}"
        log.error(result.error)

    return result


# ===========================================================================
# Forecasting Evaluation
# ===========================================================================

def evaluate_forecaster() -> EvaluationResult:
    """
    Evaluate the demand forecaster (C3 — Sanika + C4 — Srileakhana ensemble).

    Reads metrics from `models/artifacts/forecasting/metrics.json`.

    Target: MAPE < 15%
    """
    result = EvaluationResult(model_name="forecasting")
    metrics_path = FORECASTING_ARTIFACTS / "metrics.json"

    if not metrics_path.exists():
        result.error = f"Forecasting metrics not found: {metrics_path}"
        log.error(result.error)
        return result

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result.metrics = metrics

        avg_mape = metrics.get("avg_mape", 1.0)

        target = MetricTarget.FORECASTER_MAPE
        passed = target.evaluate(avg_mape)
        result.targets[target.metric_key] = (avg_mape, target.threshold, passed)

        # Check ensemble metrics if available (from C4)
        ensemble_info = metrics.get("ensemble", {})
        if ensemble_info:
            lgbm_meta = ensemble_info.get("lightgbm_metadata", {})
            lgbm_val = lgbm_meta.get("validation", {})
            lgbm_mape = lgbm_val.get("mape")
            if lgbm_mape is not None:
                result.targets["lightgbm_mape"] = (lgbm_mape, 0.15, lgbm_mape < 0.15)
                log.info("  LightGBM component MAPE: %.4f", lgbm_mape)

            result.metrics["ensemble_rows"] = ensemble_info.get("ensemble_rows", 0)

        result.passed = passed

        _log_evaluation_result("Forecaster", target, avg_mape, passed)

    except Exception as exc:
        result.error = f"Forecasting evaluation failed: {exc}"
        log.error(result.error)

    return result


# ===========================================================================
# Clustering Evaluation
# ===========================================================================

def evaluate_clustering() -> EvaluationResult:
    """
    Evaluate hotspot detection & anomaly detection (C3 — Sanika).

    Reads metrics from `models/artifacts/clustering/metrics.json`.

    Targets:
      - Silhouette > 0.4
      - Recall@20 > 0.7
    """
    result = EvaluationResult(model_name="clustering")
    metrics_path = CLUSTERING_ARTIFACTS / "metrics.json"

    if not metrics_path.exists():
        result.error = f"Clustering metrics not found: {metrics_path}"
        log.error(result.error)
        return result

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result.metrics = metrics

        # --- DBSCAN Silhouette ---
        silhouette = metrics.get("dbscan_silhouette", 0.0)
        sil_target = MetricTarget.CLUSTERING_SILHOUETTE
        sil_passed = sil_target.evaluate(silhouette)
        result.targets[sil_target.metric_key] = (silhouette, sil_target.threshold, sil_passed)

        n_clusters = metrics.get("dbscan_clusters", 0)
        result.metrics["dbscan_clusters"] = n_clusters

        _log_evaluation_result("DBSCAN Silhouette", sil_target, silhouette, sil_passed)

        # --- Isolation Forest Recall@20 ---
        recall_20 = metrics.get("iso_forest_recall_20", 0.0)
        recall_target = MetricTarget.ANOMALY_RECALL_20
        recall_passed = recall_target.evaluate(recall_20)
        result.targets[recall_target.metric_key] = (recall_20, recall_target.threshold, recall_passed)

        _log_evaluation_result("Isolation Forest Recall@20", recall_target, recall_20, recall_passed)

        # Overall pass requires both sub-targets
        result.passed = sil_passed and recall_passed

        # Load hotspot parquet summary if available
        hotspot_path = CLUSTERING_ARTIFACTS / "hotspots.parquet"
        if hotspot_path.exists():
            hotspot_df = pd.read_parquet(hotspot_path)
            result.metrics["n_block_groups"] = len(hotspot_df)
            if "cluster" in hotspot_df.columns:
                result.metrics["n_hotspot_block_groups"] = int((hotspot_df["cluster"] >= 0).sum())

    except Exception as exc:
        result.error = f"Clustering evaluation failed: {exc}"
        log.error(result.error)

    return result


# ===========================================================================
# MLflow unified logger (C5 instrumentation)
# ===========================================================================

def log_evaluation_to_mlflow(
    results: list[EvaluationResult],
    experiment_name: str = MLFLOW_EXPERIMENT,
) -> None:
    """
    Instrument all 3 model evaluations into a single MLflow experiment.

    This is the C5 (Deekshitha) unified MLflow tracking layer.
    Each model gets its own MLflow run within the shared experiment.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        log.info("Logging %d evaluations to MLflow experiment: %s", len(results), experiment_name)

        for res in results:
            if res.error:
                log.warning("Skipping MLflow log for %s (error: %s)", res.model_name, res.error)
                continue

            with mlflow.start_run(run_name=f"eval_{res.model_name}"):
                # Log all flat metrics
                for key, value in res.metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    elif isinstance(value, dict):
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, (int, float)):
                                mlflow.log_metric(f"{key}_{sub_key}", sub_val)

                # Log target evaluations
                for target_key, (val, threshold, passed) in res.targets.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(f"target_{target_key}", val)
                    if threshold is not None:
                        mlflow.log_param(f"threshold_{target_key}", threshold)
                    mlflow.log_param(f"passed_{target_key}", passed)

                mlflow.log_param("overall_passed", res.passed)
                mlflow.log_param("model_name", res.model_name)

                # Log metrics JSON as artifact if available
                metrics_json = CLASSIFIER_ARTIFACTS if "classifier" in res.model_name else (
                    FORECASTING_ARTIFACTS if "forecast" in res.model_name else CLUSTERING_ARTIFACTS
                )
                metrics_file = metrics_json / "metrics.json"
                if metrics_file.exists():
                    mlflow.log_artifact(str(metrics_file))

            log.info("  MLflow run logged for: %s (passed=%s)", res.model_name, res.passed)

    except ImportError:
        log.warning("MLflow not installed. Skipping MLflow logging.")
    except Exception as exc:
        log.warning("MLflow logging failed (non-fatal): %s", exc)


# ===========================================================================
# Summary report
# ===========================================================================

def generate_evaluation_report(results: list[EvaluationResult]) -> dict:
    """Generate a comprehensive evaluation summary report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "Phase 2 — ML Models",
        "owner": "Deekshitha (C5)",
        "models": {},
        "overall_status": "PASS",
    }

    all_passed = True
    for res in results:
        report["models"][res.model_name] = res.to_dict()
        if not res.passed:
            all_passed = False

    report["overall_status"] = "PASS" if all_passed else "FAIL"

    # Summary table for logging
    log.info("")
    log.info("=" * 70)
    log.info("  MedAlertAI Phase 2 — Evaluation Summary (C5 — Deekshitha)")
    log.info("=" * 70)
    log.info("")
    log.info("  %-20s %-30s %-10s", "Model", "Metric", "Status")
    log.info("  %s", "-" * 60)

    for res in results:
        if res.error:
            log.info("  %-20s %-30s %-10s", res.model_name, "ERROR: " + res.error[:30], "❌")
            continue
        for key, (val, threshold, passed) in res.targets.items():
            if threshold is not None:
                metric_str = f"{key}={val:.4f} (target: {threshold})"
            else:
                metric_str = f"{key}={val:.4f}"
            status = "✅" if passed else "⚠️"
            log.info("  %-20s %-30s %-10s", res.model_name, metric_str, status)

    log.info("")
    log.info("  Overall: %s", "✅ ALL TARGETS MET" if all_passed else "⚠️  SOME TARGETS NOT MET")
    log.info("=" * 70)
    log.info("")

    return report


def save_evaluation_report(report: dict, output_path: Optional[Path] = None) -> Path:
    """Write the evaluation report to JSON."""
    if output_path is None:
        output_path = MODEL_ARTIFACTS_DIR / "evaluation_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    log.info("Evaluation report saved: %s", _display_path(output_path))
    return output_path


# ===========================================================================
# Helpers
# ===========================================================================

def _log_evaluation_result(
    label: str,
    target: MetricTarget,
    value: float,
    passed: bool,
) -> None:
    """Log a single metric evaluation result."""
    if passed:
        log.info("  ✅ %s: %.4f — PASSED (%s)", label, value, target.description)
    else:
        log.warning("  ⚠️  %s: %.4f — BELOW target (%s)", label, value, target.description)


def _display_path(path: Path) -> Path:
    """Return a path relative to the project root for cleaner logging."""
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_evaluation(
    models: Optional[list[str]] = None,
    use_mlflow: bool = True,
) -> dict:
    """
    Run the unified Phase 2 evaluation harness.

    Parameters
    ----------
    models : list[str] or None
        Which models to evaluate. Options: 'classifier', 'forecasting', 'clustering'.
        If None, evaluates all.
    use_mlflow : bool
        Whether to log results to MLflow.

    Returns
    -------
    dict
        The full evaluation report.
    """
    start = time.time()
    log.info("=" * 70)
    log.info("  MedAlertAI Phase 2 — Unified Evaluation Harness")
    log.info("  Owner: Deekshitha (C5)")
    log.info("=" * 70)

    if models is None:
        models = ["classifier", "forecasting", "clustering"]

    results: list[EvaluationResult] = []

    if "classifier" in models:
        log.info("\n--- Evaluating Classifier (C2 — Suvarna) ---")
        results.append(evaluate_classifier())

    if "forecasting" in models:
        log.info("\n--- Evaluating Forecaster (C3 — Sanika + C4 — Srileakhana) ---")
        results.append(evaluate_forecaster())

    if "clustering" in models:
        log.info("\n--- Evaluating Clustering (C3 — Sanika) ---")
        results.append(evaluate_clustering())

    # Generate report
    report = generate_evaluation_report(results)

    # Save report
    save_evaluation_report(report)

    # MLflow instrumentation
    if use_mlflow:
        log.info("\n--- Logging to MLflow ---")
        log_evaluation_to_mlflow(results)

    elapsed = time.time() - start
    log.info("Evaluation harness completed in %.1f seconds.", elapsed)

    return report


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAlertAI — Phase 2 Unified Evaluation Harness (Deekshitha / C5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["classifier", "forecasting", "clustering"],
        help="Evaluate a specific model only. Default: all models.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    models_to_eval = [args.model] if args.model else None
    report = run_evaluation(
        models=models_to_eval,
        use_mlflow=not args.no_mlflow,
    )

    # Exit with non-zero if any target failed
    if report.get("overall_status") != "PASS":
        sys.exit(1)
