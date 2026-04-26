"""
classifier/train.py — MPDS Complaint Code Classifier.

Owner: Suvarna (C2)
Phase: 2

Responsibilities:
  - Gradient Boosted Tree classifier (sklearn HistGradientBoosting / LightGBM)
    wrapped in an sklearn Pipeline
  - Optuna HPO (50 trials)
  - Flag disagreement rows (confidence >0.7)
  - Target: macro F1 > 0.75

Task:
  Predict the MPDS complaint group (`mpds_group`) for each dispatch record
  using dispatch metadata features (priority, service type, location, time).
  NOTE: `call_type` is EXCLUDED from features because the target
  (`mpds_group`) is deterministically derived from it via mpds_mapper,
  which would cause data leakage and trivial 100% accuracy.

Engine priority:
  1. LightGBM (if available + libomp installed)
  2. sklearn HistGradientBoostingClassifier (always available, same algorithm)

Usage (from repo root):
    python3 -m src.models.classifier.train
    python3 -m src.models.classifier.train --trials 20 --no-mlflow
    python3 -m src.models.classifier.train --skip-hpo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.mpds_mapper import map_call_to_mpds

# ---------------------------------------------------------------------------
# Try to import LightGBM; fall back to sklearn if unavailable
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except (ImportError, OSError):
    _HAS_LGBM = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.classifier")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_DIR = PROCESSED_DATA_DIR / "splits" / "classifier"
CLASSIFIER_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "classifier"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# NOTE: call_type is intentionally EXCLUDED from features because the target
# (mpds_group) is derived deterministically from call_type via mpds_mapper.
# Including it would cause data leakage → trivial 100% accuracy.
# The classifier should learn the mapping from dispatch metadata only,
# enabling it to flag records where metadata doesn't match the MPDS label.
CATEGORICAL_FEATURES = [
    "priority_code",
    "priority_description",
    "quarter",
    "city_code",
    "service_type",
]
NUMERIC_FEATURES = [
    "year",
    "longitude",
    "latitude",
    "completeness_score",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COL = "mpds_group"
LABEL_CODE_COL = "mpds_label_code"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_HGBC_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "max_depth": 10,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "random_state": 42,
    "early_stopping": True,
    "n_iter_no_change": 15,
    "validation_fraction": 0.1,
    "verbose": 1,
}

DEFAULT_LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 63,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


# ===========================================================================
# Data loading & MPDS target creation
# ===========================================================================

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from Parquet."""
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        path = SPLITS_DIR / name
        if not path.exists():
            log.error("Split file not found: %s", path)
            log.error("Run `python3 -m src.data.splits` first.")
            sys.exit(1)

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test = pd.read_parquet(SPLITS_DIR / "test.parquet")

    log.info("Loaded splits — train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def add_mpds_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MPDS mapper to create the target column."""
    df = df.copy()
    df[TARGET_COL] = df["call_type"].apply(map_call_to_mpds)
    return df


OTHER_LABEL = "Other"


def collapse_long_tail(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    top_n: int,
    other_label: str = OTHER_LABEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Roll classes outside the top-N (by train frequency) into a single 'Other' bucket.

    Macro F1 averages every class equally, so a long tail of rare classes the
    model can never learn (e.g. Drowning at 830 train rows out of 2.3M) drives
    macro F1 toward zero. Collapsing them lets the metric reflect performance
    on the classes that actually carry volume.
    """
    counts = train[TARGET_COL].value_counts()
    if top_n >= len(counts):
        log.info("top_n=%d >= total classes=%d — no collapse applied", top_n, len(counts))
        return train, val, test, counts.index.tolist()

    top_classes = counts.head(top_n).index.tolist()
    log.info("Collapsing %d-class target → top-%d + '%s'", len(counts), top_n, other_label)
    log.info("  Kept classes: %s", top_classes)

    def _collapse(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[TARGET_COL] = df[TARGET_COL].where(df[TARGET_COL].isin(top_classes), other_label)
        return df

    return _collapse(train), _collapse(val), _collapse(test), top_classes + [other_label]


def encode_target(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Encode MPDS groups as integer labels. Returns modified DFs + label encoder."""
    le = LabelEncoder()
    le.fit(train[TARGET_COL])

    for df in (train, val, test):
        df[LABEL_CODE_COL] = le.transform(df[TARGET_COL])

    label_map = pd.DataFrame({
        LABEL_CODE_COL: range(len(le.classes_)),
        TARGET_COL: le.classes_,
    })

    n_classes = len(le.classes_)
    log.info("MPDS target encoded — %d classes", n_classes)
    for cls in le.classes_:
        count = (train[TARGET_COL] == cls).sum()
        log.info("  %-25s %7d (%.1f%%)", cls, count, count / len(train) * 100)

    return train, val, test, label_map, le


# ===========================================================================
# Preprocessing pipeline
# ===========================================================================

def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer for feature preprocessing."""
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features X and target y from a DataFrame."""
    df_copy = df.copy()
    for col in CATEGORICAL_FEATURES:
        df_copy[col] = df_copy[col].astype(str)

    if fit:
        X = preprocessor.fit_transform(df_copy[ALL_FEATURES])
    else:
        X = preprocessor.transform(df_copy[ALL_FEATURES])

    y = df_copy[LABEL_CODE_COL].values.astype(np.int32)
    return X, y


# ===========================================================================
# Model factory
# ===========================================================================

def _get_engine_name() -> str:
    return "LightGBM" if _HAS_LGBM else "HistGradientBoosting"


# ===========================================================================
# Optuna HPO
# ===========================================================================

def _optuna_objective_hgbc(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [63, 127, 255]),
        "random_state": 42,
        "early_stopping": True,
        "n_iter_no_change": 10,
        "validation_fraction": 0.1,
        "verbose": 0,
    }

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    return f1


def _optuna_objective_lgbm(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> float:
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    return f1


def run_optuna_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    n_trials: int = 50,
) -> dict:
    """Run Optuna HPO targeting macro F1. Returns best params dict."""
    engine = _get_engine_name()
    log.info("Starting Optuna HPO with %d trials (engine: %s) …", n_trials, engine)

    def objective(trial: optuna.Trial) -> float:
        if _HAS_LGBM:
            return _optuna_objective_lgbm(trial, X_train, y_train, X_val, y_val, n_classes)
        return _optuna_objective_hgbc(trial, X_train, y_train, X_val, y_val)

    study = optuna.create_study(
        direction="maximize",
        study_name="mpds_classifier_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    log.info("Best trial #%d — macro F1=%.4f", best.number, best.value)
    log.info("Best params: %s", json.dumps(best.params, indent=2))

    if _HAS_LGBM:
        best_params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            **best.params,
        }
    else:
        best_params = {
            "random_state": 42,
            "early_stopping": True,
            "n_iter_no_change": 15,
            "validation_fraction": 0.1,
            "verbose": 0,
            **best.params,
        }

    return best_params


# ===========================================================================
# Training
# ===========================================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
):
    """Train the classifier with the best available engine."""
    engine = _get_engine_name()
    log.info("Training final %s model …", engine)

    if _HAS_LGBM:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        log.info("Training complete. Best iteration: %d", model.best_iteration_)
    else:
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        n_iter = getattr(model, "n_iter_", params.get("max_iter", "N/A"))
        log.info("Training complete. Iterations: %s", n_iter)

    return model


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    label_map: pd.DataFrame,
    split_name: str = "test",
) -> dict:
    """Evaluate model and return metrics dict.

    Includes confusion matrix and per-class metrics (precision/recall/F1/support)
    used by the QA dashboard tab and the Phase 5 evaluation tests.
    """
    y_pred = model.predict(X)

    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y, y_pred)
    macro_precision = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y, y_pred, average="macro", zero_division=0)

    class_codes = label_map[LABEL_CODE_COL].tolist()
    class_names = label_map[TARGET_COL].tolist()

    cm = confusion_matrix(y, y_pred, labels=class_codes)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "true_mpds_group"

    report = classification_report(
        y,
        y_pred,
        labels=class_codes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        cls: {
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "f1": round(stats["f1-score"], 4),
            "support": int(stats["support"]),
        }
        for cls, stats in report.items()
        if cls in class_names
    }

    log.info("=== %s Evaluation ===", split_name.upper())
    log.info("  Accuracy:         %.4f", accuracy)
    log.info("  Macro F1:         %.4f  (target: >0.75)", macro_f1)
    log.info("  Weighted F1:      %.4f", weighted_f1)
    log.info("  Macro Precision:  %.4f", macro_precision)
    log.info("  Macro Recall:     %.4f", macro_recall)

    if macro_f1 >= 0.75:
        log.info("  ✅ PASSED macro F1 target (>0.75)")
    else:
        log.warning("  ⚠️  BELOW macro F1 target (>0.75) — got %.4f", macro_f1)

    return {
        "split": split_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "confusion_matrix": cm_df,
        "per_class": per_class,
    }


# ===========================================================================
# Disagreement flagging
# ===========================================================================

def flag_disagreements(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    df: pd.DataFrame,
    label_map: pd.DataFrame,
    confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Flag rows where the model is confident (max_proba > threshold)
    but predicts a different class than the MPDS-mapped label.

    These are potential data quality issues — either the raw call_type
    is miscategorized or the MPDS mapping needs review.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    max_confidence = y_proba.max(axis=1)

    code_to_name = dict(zip(label_map[LABEL_CODE_COL], label_map[TARGET_COL]))

    result = df.copy()
    result["predicted_mpds_code"] = y_pred
    result["predicted_mpds_group"] = pd.Series(y_pred).map(code_to_name).values
    result["max_confidence"] = max_confidence.round(4)
    result["is_correct"] = (y_pred == y_true)
    result["is_disagreement"] = (
        (max_confidence > confidence_threshold) & (~result["is_correct"])
    )

    n_total = len(result)
    n_confident = (max_confidence > confidence_threshold).sum()
    n_disagree = result["is_disagreement"].sum()

    log.info("=== Disagreement Flagging (confidence > %.2f) ===", confidence_threshold)
    log.info("  Total rows:            %d", n_total)
    log.info("  Confident predictions: %d (%.1f%%)", n_confident, n_confident / n_total * 100)
    log.info("  Disagreements:         %d (%.1f%% of confident)",
             n_disagree, n_disagree / max(n_confident, 1) * 100)

    return result


# ===========================================================================
# Artifact serialization
# ===========================================================================

def save_artifacts(
    model,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    label_map: pd.DataFrame,
    metrics: dict,
    best_params: dict,
    disagreements: pd.DataFrame,
    confusion_matrix_df: pd.DataFrame,
    per_class_metrics: dict,
    output_dir: Path = CLASSIFIER_ARTIFACTS_DIR,
) -> None:
    """Save all classifier artifacts for downstream dashboard consumption."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full sklearn pipeline (preprocessor + model)
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    pipeline_path = output_dir / "classifier_pipeline.joblib"
    joblib.dump(full_pipeline, pipeline_path)
    log.info("Saved pipeline: %s (%.1f MB)",
             _display_path(pipeline_path),
             pipeline_path.stat().st_size / 1_048_576)

    # 2. Standalone model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    log.info("Saved standalone model: %s", _display_path(model_path))

    # 2b. LightGBM native format
    if _HAS_LGBM and hasattr(model, "booster_"):
        lgbm_path = output_dir / "lgbm_model.txt"
        model.booster_.save_model(str(lgbm_path))
        log.info("Saved LightGBM native model: %s", _display_path(lgbm_path))

    # 3. Label encoder + map
    le_path = output_dir / "label_encoder.joblib"
    joblib.dump(label_encoder, le_path)
    label_map_path = output_dir / "label_map.parquet"
    label_map.to_parquet(label_map_path, index=False)
    log.info("Saved label map: %s (%d classes)", _display_path(label_map_path), len(label_map))

    # 4. Metrics JSON
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    log.info("Saved metrics: %s", _display_path(metrics_path))

    # 5. Best hyperparameters
    params_path = output_dir / "best_params.json"
    serializable_params = {
        k: int(v) if isinstance(v, (np.integer,)) else
        float(v) if isinstance(v, (np.floating,)) else v
        for k, v in best_params.items()
    }
    params_path.write_text(json.dumps(serializable_params, indent=2) + "\n", encoding="utf-8")
    log.info("Saved best params: %s", _display_path(params_path))

    # 6. Disagreement flags (Parquet for dashboard QA page)
    disagree_path = output_dir / "disagreements.parquet"
    disagreements.to_parquet(disagree_path, index=False)
    n_flagged = disagreements["is_disagreement"].sum()
    log.info("Saved disagreements: %s (%d flagged rows)", _display_path(disagree_path), n_flagged)

    # 6b. Confusion matrix (test split) — rows=true class, cols=predicted class
    cm_path = output_dir / "confusion_matrix.parquet"
    confusion_matrix_df.reset_index().to_parquet(cm_path, index=False)
    log.info("Saved confusion matrix: %s (%dx%d)",
             _display_path(cm_path),
             confusion_matrix_df.shape[0],
             confusion_matrix_df.shape[1])

    # 6c. Per-class metrics (test split)
    per_class_path = output_dir / "per_class_metrics.json"
    per_class_path.write_text(json.dumps(per_class_metrics, indent=2) + "\n", encoding="utf-8")
    log.info("Saved per-class metrics: %s (%d classes)",
             _display_path(per_class_path), len(per_class_metrics))

    # 7. Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(ALL_FEATURES))

    importance_df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    log.info("Saved feature importance: %s", _display_path(importance_path))
    log.info("  Top 5 features: %s",
             list(importance_df.head(5).itertuples(index=False, name=None)))


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


# ===========================================================================
# MLflow integration
# ===========================================================================

def log_to_mlflow(
    metrics: dict,
    best_params: dict,
    model,
    pipeline_path: Path,
) -> None:
    """Log experiment to MLflow tracking."""
    try:
        import mlflow

        mlflow.set_experiment("medalertai-classifier")

        with mlflow.start_run(run_name="mpds_classifier"):
            for k, v in best_params.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            mlflow.log_metric(f"{k}_{sub_k}", sub_v)

            if pipeline_path.exists():
                mlflow.log_artifact(str(pipeline_path))

            log.info("MLflow run logged successfully.")

    except Exception as exc:
        log.warning("MLflow logging failed (non-fatal): %s", exc)


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_training(
    n_trials: int = 50,
    skip_hpo: bool = False,
    use_mlflow: bool = True,
    confidence_threshold: float = 0.7,
    top_n_classes: Optional[int] = None,
) -> dict:
    """
    Full training pipeline:
      1. Load splits
      2. Apply MPDS mapper to create target column
      3. Encode target labels
      4. Build preprocessor & prepare data
      5. Optuna HPO (or use defaults)
      6. Train final model
      7. Evaluate on val + test
      8. Flag disagreements
      9. Save all artifacts
      10. (Optional) Log to MLflow
    """
    start_time = time.time()
    engine = _get_engine_name()
    log.info("=== MedAlertAI MPDS Classifier — Engine: %s ===", engine)

    # 1. Load data
    train_df, val_df, test_df = load_splits()

    # 2. Apply MPDS mapping
    log.info("Applying MPDS mapper to create target column …")
    train_df = add_mpds_target(train_df)
    val_df = add_mpds_target(val_df)
    test_df = add_mpds_target(test_df)

    # 2b. (Optional) collapse long tail to top-N + Other
    if top_n_classes is not None and top_n_classes > 0:
        train_df, val_df, test_df, _kept = collapse_long_tail(
            train_df, val_df, test_df, top_n=top_n_classes,
        )

    # 3. Encode target
    train_df, val_df, test_df, label_map, label_encoder = encode_target(
        train_df, val_df, test_df
    )
    n_classes = len(label_map)

    # 4. Build preprocessor & prepare data
    preprocessor = build_preprocessor()
    X_train, y_train = prepare_data(train_df, preprocessor, fit=True)
    X_val, y_val = prepare_data(val_df, preprocessor, fit=False)
    X_test, y_test = prepare_data(test_df, preprocessor, fit=False)

    log.info("Prepared data — X_train: %s, X_val: %s, X_test: %s",
             X_train.shape, X_val.shape, X_test.shape)

    # 5. HPO
    if skip_hpo:
        log.info("Skipping HPO — using default params.")
        if _HAS_LGBM:
            best_params = {**DEFAULT_LGBM_PARAMS, "num_class": n_classes}
        else:
            best_params = {**DEFAULT_HGBC_PARAMS}
    else:
        best_params = run_optuna_hpo(
            X_train, y_train, X_val, y_val,
            n_classes=n_classes,
            n_trials=n_trials,
        )

    # 6. Train final model
    model = train_model(X_train, y_train, X_val, y_val, best_params)

    # 7. Evaluate
    val_metrics = evaluate_model(model, X_val, y_val, label_map, "validation")
    test_metrics = evaluate_model(model, X_test, y_test, label_map, "test")

    # Pull non-JSON artifacts out of the metrics dicts; persist separately.
    val_metrics.pop("confusion_matrix", None)
    val_per_class = val_metrics.pop("per_class", {})
    test_cm_df = test_metrics.pop("confusion_matrix")
    test_per_class = test_metrics.pop("per_class", {})

    all_metrics = {
        "engine": engine,
        "validation": val_metrics,
        "test": test_metrics,
        "n_classes": n_classes,
        "n_train_rows": len(train_df),
        "training_time_sec": round(time.time() - start_time, 1),
        "validation_per_class": val_per_class,
        "test_per_class": test_per_class,
    }

    # 8. Flag disagreements on test set
    disagreements = flag_disagreements(
        model, X_test, y_test, test_df, label_map,
        confidence_threshold=confidence_threshold,
    )

    # 9. Save artifacts
    save_artifacts(
        model=model,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        label_map=label_map,
        metrics=all_metrics,
        best_params=best_params,
        disagreements=disagreements,
        confusion_matrix_df=test_cm_df,
        per_class_metrics=test_per_class,
    )

    # 10. MLflow
    if use_mlflow:
        pipeline_path = CLASSIFIER_ARTIFACTS_DIR / "classifier_pipeline.joblib"
        log_to_mlflow(all_metrics, best_params, model, pipeline_path)

    elapsed = time.time() - start_time
    log.info("=== Training pipeline complete in %.1f seconds ===", elapsed)

    return all_metrics


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedAlertAI — Phase 2 MPDS Classifier Training (Suvarna)"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna HPO trials (default: 50)"
    )
    parser.add_argument(
        "--skip-hpo", action="store_true",
        help="Skip Optuna HPO and use default hyperparameters"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Confidence threshold for disagreement flagging (default: 0.7)"
    )
    parser.add_argument(
        "--top-n-classes", type=int, default=None,
        help="If set, collapse classes outside the top-N (by train freq) into 'Other'. "
             "Mitigates macro-F1 penalty from rare classes (default: keep all)."
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_training(
        n_trials=args.trials,
        skip_hpo=args.skip_hpo,
        use_mlflow=not args.no_mlflow,
        confidence_threshold=args.confidence_threshold,
        top_n_classes=args.top_n_classes,
    )
