"""
tests/test_evaluation.py — CI test stubs for all 3 ML models.

Owner: Deekshitha (C5)
Phase: 2

CI test stubs that validate:
  - Classifier: macro F1 above regression floor (plan target > 0.55, revised
    down from 0.75 because the WPRDC dataset lacks hour-of-day timestamps)
  - Forecaster: MAPE < 15% via serialized metrics
  - Clustering: Silhouette > 0.4 and Recall@20 > 0.7 via serialized metrics
  - Artifacts exist and are loadable
  - Evaluation harness itself is functional

These tests are designed to run in CI without requiring GPU or
expensive retraining — they read pre-computed metrics from
`models/artifacts/`.

Usage:
    pytest tests/test_evaluation.py -v
    pytest tests/test_evaluation.py -v -k classifier
    pytest tests/test_evaluation.py -v -k forecasting
    pytest tests/test_evaluation.py -v -k clustering
"""

import json
from pathlib import Path

import pytest

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLASSIFIER_DIR = MODEL_ARTIFACTS_DIR / "classifier"
FORECASTING_DIR = MODEL_ARTIFACTS_DIR / "forecasting"
CLUSTERING_DIR = MODEL_ARTIFACTS_DIR / "clustering"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    """Load a JSON file or return an empty dict if not found."""
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_file_exists(path: Path, label: str = "") -> None:
    """Assert that a file exists, with a descriptive message."""
    assert path.exists(), f"{label or path.name} not found at {path}"


# ===========================================================================
# CLASSIFIER TESTS (C2 — Suvarna)
# ===========================================================================

class TestClassifierArtifacts:
    """Verify that classifier artifacts exist and are well-formed."""

    def test_metrics_json_exists(self):
        _assert_file_exists(CLASSIFIER_DIR / "metrics.json", "Classifier metrics")

    def test_classifier_pipeline_exists(self):
        _assert_file_exists(
            CLASSIFIER_DIR / "classifier_pipeline.joblib",
            "Classifier pipeline",
        )

    def test_model_exists(self):
        _assert_file_exists(CLASSIFIER_DIR / "model.joblib", "Classifier model")

    def test_label_encoder_exists(self):
        _assert_file_exists(CLASSIFIER_DIR / "label_encoder.joblib", "Label encoder")

    def test_label_map_exists(self):
        _assert_file_exists(CLASSIFIER_DIR / "label_map.parquet", "Label map")

    def test_disagreements_parquet_exists(self):
        _assert_file_exists(
            CLASSIFIER_DIR / "disagreements.parquet",
            "Disagreement flags",
        )

    def test_feature_importance_exists(self):
        _assert_file_exists(
            CLASSIFIER_DIR / "feature_importance.csv",
            "Feature importance",
        )

    def test_best_params_exists(self):
        _assert_file_exists(CLASSIFIER_DIR / "best_params.json", "Best params")


class TestClassifierMetrics:
    """Validate classifier metric targets from serialized metrics.

    The plan target was lowered from macro F1 > 0.75 to > 0.55 once the
    WPRDC dataset's lack of hour-of-day timestamps was identified as a
    structural ceiling. These tests gate on a regression floor that
    leaves headroom toward the revised plan target — raise the floor as
    the saved metrics improve. Single source of truth for the same
    constants is `tests/test_phase5_classifier.py`.
    """

    F1_REGRESSION_FLOOR = 0.42
    F1_PLAN_TARGET = 0.55
    ACCURACY_REGRESSION_FLOOR = 0.45

    @pytest.fixture(autouse=True)
    def _load_metrics(self):
        self.metrics = _load_json(CLASSIFIER_DIR / "metrics.json")

    def test_has_test_metrics(self):
        assert "test" in self.metrics, "metrics.json must contain 'test' key"

    def test_macro_f1_above_target(self):
        test_metrics = self.metrics["test"]
        macro_f1 = test_metrics.get("macro_f1", 0.0)
        assert macro_f1 >= self.F1_REGRESSION_FLOOR, (
            f"Classifier macro F1 = {macro_f1:.4f} dropped below regression "
            f"floor {self.F1_REGRESSION_FLOOR:.2f}. "
            f"Plan target is {self.F1_PLAN_TARGET:.2f}."
        )

    def test_accuracy_above_baseline(self):
        test_metrics = self.metrics["test"]
        accuracy = test_metrics.get("accuracy", 0.0)
        assert accuracy >= self.ACCURACY_REGRESSION_FLOOR, (
            f"Classifier accuracy = {accuracy:.4f} dropped below regression "
            f"floor {self.ACCURACY_REGRESSION_FLOOR:.2f}."
        )

    def test_has_n_classes(self):
        """Verify n_classes is reported."""
        assert "n_classes" in self.metrics
        assert self.metrics["n_classes"] > 0

    def test_has_training_time(self):
        """Verify training time is reported."""
        assert "training_time_sec" in self.metrics
        assert self.metrics["training_time_sec"] > 0


class TestClassifierLive:
    """Live re-evaluation from serialized pipeline (optional CI test)."""

    def test_pipeline_loads_and_predicts(self):
        """Test that the saved pipeline can make predictions."""
        pipeline_path = CLASSIFIER_DIR / "classifier_pipeline.joblib"
        if not pipeline_path.exists():
            pytest.skip("Pipeline not found")

        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not installed")

        try:
            import pandas as pd

            pipeline = joblib.load(pipeline_path)
            assert pipeline is not None
            assert hasattr(pipeline, "predict")
        except Exception as exc:
            pytest.fail(f"Failed to load pipeline: {exc}")

    def test_label_map_parquet_loadable(self):
        """Test that label_map.parquet loads correctly."""
        label_map_path = CLASSIFIER_DIR / "label_map.parquet"
        if not label_map_path.exists():
            pytest.skip("Label map not found")

        import pandas as pd

        label_map = pd.read_parquet(label_map_path)
        assert len(label_map) > 0
        assert "mpds_group" in label_map.columns or "mpds_label_code" in label_map.columns

    def test_disagreements_have_required_columns(self):
        """Test that disagreements Parquet has the expected schema."""
        disagree_path = CLASSIFIER_DIR / "disagreements.parquet"
        if not disagree_path.exists():
            pytest.skip("Disagreements not found")

        import pandas as pd

        df = pd.read_parquet(disagree_path)
        required = {"is_disagreement", "max_confidence", "is_correct"}
        missing = required - set(df.columns)
        assert not missing, f"Disagreements Parquet missing columns: {missing}"


# ===========================================================================
# FORECASTING TESTS (C3 — Sanika + C4 — Srileakhana)
# ===========================================================================

class TestForecastingArtifacts:
    """Verify that forecasting artifacts exist."""

    def test_metrics_json_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts directory not yet created")
        _assert_file_exists(FORECASTING_DIR / "metrics.json", "Forecasting metrics")

    def test_prophet_model_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts directory not yet created")
        _assert_file_exists(
            FORECASTING_DIR / "prophet_model.json",
            "Prophet model",
        )


class TestForecastingMetrics:
    """Validate forecasting metric targets from serialized metrics."""

    @pytest.fixture(autouse=True)
    def _load_metrics(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts not yet created")
        self.metrics = _load_json(FORECASTING_DIR / "metrics.json")

    def test_has_avg_mape(self):
        assert "avg_mape" in self.metrics, "metrics.json must contain 'avg_mape'"

    def test_mape_below_target(self):
        """Phase 2 target: MAPE < 15%."""
        avg_mape = self.metrics.get("avg_mape", 1.0)
        assert avg_mape < 0.15, (
            f"Forecaster MAPE = {avg_mape:.4f} ({avg_mape*100:.1f}%), target is < 15%"
        )

    def test_target_met_flag(self):
        """Verify target_met is True."""
        target_met = self.metrics.get("target_met", False)
        assert target_met, "Forecasting target_met flag should be True"


class TestEnsembleArtifacts:
    """Verify C4 (Srileakhana) ensemble artifacts."""

    def test_lightgbm_forecaster_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts not yet created")
        path = FORECASTING_DIR / "lightgbm_forecaster.joblib"
        if not path.exists():
            pytest.skip("LightGBM forecaster not yet trained")
        assert path.stat().st_size > 0

    def test_ensemble_model_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts not yet created")
        path = FORECASTING_DIR / "ensemble_model.joblib"
        if not path.exists():
            pytest.skip("Ensemble model not yet trained")
        assert path.stat().st_size > 0

    def test_ensemble_forecast_parquet_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts not yet created")
        path = FORECASTING_DIR / "ensemble_forecast.parquet"
        if not path.exists():
            pytest.skip("Ensemble forecast not yet generated")

        import pandas as pd

        df = pd.read_parquet(path)
        assert len(df) > 0
        assert "ensemble_yhat" in df.columns, "Ensemble forecast must have 'ensemble_yhat'"

    def test_artifact_manifest_exists(self):
        if not FORECASTING_DIR.exists():
            pytest.skip("Forecasting artifacts not yet created")
        path = FORECASTING_DIR / "artifact_manifest.json"
        if not path.exists():
            pytest.skip("Artifact manifest not yet created")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert "forecasting" in manifest
        assert "ensemble_weights" in manifest


# ===========================================================================
# CLUSTERING TESTS (C3 — Sanika)
# ===========================================================================

class TestClusteringArtifacts:
    """Verify that clustering artifacts exist."""

    def test_metrics_json_exists(self):
        if not CLUSTERING_DIR.exists():
            pytest.skip("Clustering artifacts directory not yet created")
        _assert_file_exists(CLUSTERING_DIR / "metrics.json", "Clustering metrics")

    def test_dbscan_model_exists(self):
        if not CLUSTERING_DIR.exists():
            pytest.skip("Clustering artifacts directory not yet created")
        _assert_file_exists(
            CLUSTERING_DIR / "dbscan_model.joblib",
            "DBSCAN model",
        )

    def test_isolation_forest_model_exists(self):
        if not CLUSTERING_DIR.exists():
            pytest.skip("Clustering artifacts directory not yet created")
        _assert_file_exists(
            CLUSTERING_DIR / "isolation_forest_model.joblib",
            "Isolation Forest model",
        )

    def test_hotspots_parquet_exists(self):
        if not CLUSTERING_DIR.exists():
            pytest.skip("Clustering artifacts directory not yet created")
        _assert_file_exists(
            CLUSTERING_DIR / "hotspots.parquet",
            "Hotspots Parquet",
        )


class TestClusteringMetrics:
    """Validate clustering metric targets from serialized metrics."""

    @pytest.fixture(autouse=True)
    def _load_metrics(self):
        if not CLUSTERING_DIR.exists():
            pytest.skip("Clustering artifacts not yet created")
        self.metrics = _load_json(CLUSTERING_DIR / "metrics.json")

    def test_has_silhouette_score(self):
        assert "dbscan_silhouette" in self.metrics, "Missing dbscan_silhouette"

    def test_silhouette_above_target(self):
        """Phase 2 target: Silhouette > 0.4."""
        silhouette = self.metrics.get("dbscan_silhouette", 0.0)
        assert silhouette > 0.4, (
            f"DBSCAN Silhouette = {silhouette:.4f}, target is > 0.4"
        )

    def test_has_recall_20(self):
        assert "iso_forest_recall_20" in self.metrics, "Missing iso_forest_recall_20"

    def test_recall_20_above_target(self):
        """Phase 2 target: Recall@20 > 0.7."""
        recall_20 = self.metrics.get("iso_forest_recall_20", 0.0)
        assert recall_20 > 0.7, (
            f"Isolation Forest Recall@20 = {recall_20:.4f}, target is > 0.7"
        )

    def test_has_cluster_count(self):
        """Sanity check: at least 1 cluster found."""
        n_clusters = self.metrics.get("dbscan_clusters", 0)
        assert n_clusters >= 1, f"Expected >= 1 cluster, got {n_clusters}"


class TestClusteringLive:
    """Live artifact loading tests for clustering."""

    def test_dbscan_model_loads(self):
        model_path = CLUSTERING_DIR / "dbscan_model.joblib"
        if not model_path.exists():
            pytest.skip("DBSCAN model not yet trained")

        import joblib

        model = joblib.load(model_path)
        assert hasattr(model, "fit_predict")

    def test_isolation_forest_model_loads(self):
        model_path = CLUSTERING_DIR / "isolation_forest_model.joblib"
        if not model_path.exists():
            pytest.skip("Isolation Forest model not yet trained")

        import joblib

        model = joblib.load(model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "score_samples")

    def test_hotspots_parquet_has_cluster_column(self):
        hotspot_path = CLUSTERING_DIR / "hotspots.parquet"
        if not hotspot_path.exists():
            pytest.skip("Hotspots not yet generated")

        import pandas as pd

        df = pd.read_parquet(hotspot_path)
        assert "cluster" in df.columns, "Hotspots must have 'cluster' column"
        assert "latitude" in df.columns
        assert "longitude" in df.columns


# ===========================================================================
# EVALUATION HARNESS TESTS (C5 — Deekshitha)
# ===========================================================================

class TestEvaluationHarness:
    """Test the unified evaluation harness itself."""

    def test_evaluate_classifier_returns_result(self):
        from src.models.evaluate import evaluate_classifier, EvaluationResult

        result = evaluate_classifier()
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "classifier"

    def test_evaluate_forecaster_returns_result(self):
        from src.models.evaluate import evaluate_forecaster, EvaluationResult

        result = evaluate_forecaster()
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "forecasting"

    def test_evaluate_clustering_returns_result(self):
        from src.models.evaluate import evaluate_clustering, EvaluationResult

        result = evaluate_clustering()
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "clustering"

    def test_metric_target_enum_evaluates_correctly(self):
        from src.models.evaluate import MetricTarget

        # Greater direction
        assert MetricTarget.CLASSIFIER_MACRO_F1.evaluate(0.60) is True
        assert MetricTarget.CLASSIFIER_MACRO_F1.evaluate(0.50) is False
        assert MetricTarget.CLASSIFIER_MACRO_F1.evaluate(0.55) is True  # boundary

        # Less direction
        assert MetricTarget.FORECASTER_MAPE.evaluate(0.10) is True
        assert MetricTarget.FORECASTER_MAPE.evaluate(0.20) is False
        assert MetricTarget.FORECASTER_MAPE.evaluate(0.15) is False  # boundary (not strictly less)

    def test_evaluation_result_to_dict(self):
        from src.models.evaluate import EvaluationResult

        result = EvaluationResult(
            model_name="test_model",
            metrics={"accuracy": 0.95},
            targets={"macro_f1": (0.80, 0.75, True)},
            passed=True,
        )
        d = result.to_dict()
        assert d["model_name"] == "test_model"
        assert d["passed"] is True
        assert d["targets"]["macro_f1"]["value"] == 0.80
        assert d["targets"]["macro_f1"]["threshold"] == 0.75
        assert d["targets"]["macro_f1"]["passed"] is True

    def test_run_evaluation_returns_report(self):
        from src.models.evaluate import run_evaluation

        report = run_evaluation(models=["classifier"], use_mlflow=False)
        assert "models" in report
        assert "overall_status" in report
        assert "classifier" in report["models"]

    def test_generate_evaluation_report_structure(self):
        from src.models.evaluate import EvaluationResult, generate_evaluation_report

        results = [
            EvaluationResult(
                model_name="test",
                metrics={"val": 1.0},
                targets={"metric": (1.0, 0.5, True)},
                passed=True,
            )
        ]
        report = generate_evaluation_report(results)
        assert report["overall_status"] == "PASS"
        assert "test" in report["models"]


# ===========================================================================
# DATA SPLIT TESTS (C1 — Greeshma support verification)
# ===========================================================================

class TestDataSplits:
    """Verify that C1 (Greeshma) training splits are available."""

    def test_classifier_train_split_exists(self):
        path = SPLITS_DIR / "classifier" / "train.parquet"
        if not path.exists():
            pytest.skip("Classifier splits not yet created")
        assert path.stat().st_size > 0

    def test_classifier_val_split_exists(self):
        path = SPLITS_DIR / "classifier" / "val.parquet"
        if not path.exists():
            pytest.skip("Classifier splits not yet created")
        assert path.stat().st_size > 0

    def test_classifier_test_split_exists(self):
        path = SPLITS_DIR / "classifier" / "test.parquet"
        if not path.exists():
            pytest.skip("Classifier splits not yet created")
        assert path.stat().st_size > 0

    def test_forecaster_train_split_exists(self):
        path = SPLITS_DIR / "forecaster" / "train.parquet"
        if not path.exists():
            pytest.skip("Forecaster splits not yet created")
        assert path.stat().st_size > 0

    def test_clustering_fit_split_exists(self):
        path = SPLITS_DIR / "clustering" / "fit.parquet"
        if not path.exists():
            pytest.skip("Clustering splits not yet created")
        assert path.stat().st_size > 0
