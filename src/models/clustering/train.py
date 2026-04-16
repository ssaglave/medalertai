"""
clustering/train.py — Hotspot Detection & Anomaly Detection.

Owner: Sanika (C3)
Phase: 2

Responsibilities:
  - DBSCAN hotspot detection (eps=0.3, min_samples=5)
  - Isolation Forest anomaly detection (contamination=0.05)
  - Target: Silhouette > 0.4, Recall@20 > 0.7
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config.settings import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("medalertai.clustering")

warnings.filterwarnings("ignore")

CLUSTERING_ARTIFACTS_DIR = MODEL_ARTIFACTS_DIR / "clustering"


def load_clustering_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dispatch data and prepare features for clustering/anomaly detection."""
    path = PROCESSED_DATA_DIR / "fact_dispatch_clean.parquet"
    if not path.exists():
        log.error("Clean data not found: %s", path)
        log.error("Ensure Phase 1 ingestion is complete.")
        sys.exit(1)
        
    df = pd.read_parquet(path)
    
    if "census_block_group" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
        log.error("Missing geographic columns in dispatch data.")
        sys.exit(1)
        
    hotspot_df = df.groupby("census_block_group").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        call_volume=("incident_id", "count")
    ).dropna()
    
    log.info("Clustering data aggregated to %d census block groups.", len(hotspot_df))
    return hotspot_df, df


def train_dbscan(hotspot_df: pd.DataFrame) -> dict:
    """Run DBSCAN to find geographical hotspots among block groups."""
    log.info("Running DBSCAN hotspot detection...")
    
    X_geo = hotspot_df[["latitude", "longitude"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_geo)
    
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    hotspot_df["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    try:
        if n_clusters > 1:
            sil_score = silhouette_score(X_scaled, labels)
        else:
            sil_score = 0.0
    except Exception as e:
        log.warning("Silhouette score failed: %s", e)
        sil_score = 0.0
        
    target_met = bool(sil_score > 0.4)
    log.info("Found %d hotspots (clusters). Silhouette Score: %.3f", n_clusters, sil_score)
    if target_met:
        log.info("  ✅ PASSED Silhouette target (>0.4)")
    else:
        log.warning("  ⚠️  DID NOT MEET Silhouette target (>0.4)")

    return {
        "model": dbscan,
        "scaler": scaler,
        "labels": labels,
        "silhouette_score": float(sil_score),
        "n_clusters": int(n_clusters),
        "target_met": target_met
    }


def train_isolation_forest(df: pd.DataFrame) -> dict:
    """Run Isolation Forest to find anomalous calls based on time/location."""
    log.info("Running Isolation Forest anomaly detection...")
    
    features = ["latitude", "longitude", "hour", "day_of_week", "month"]
    available_features = [f for f in features if f in df.columns]
    
    train_df = df.dropna(subset=available_features).sample(min(100000, len(df)), random_state=42)
    
    X = train_df[available_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_scaled)
    
    scores = iso_forest.score_samples(X_scaled)
    train_df["anomaly_score"] = scores
    
    top_20 = train_df.sort_values("anomaly_score").head(int(0.2 * len(train_df)))
    
    recall_20 = 0.0
    if "priority_code" in train_df.columns:
        high_pri = train_df["priority_code"].astype(str).str.contains("E1|F1")
        total_high = high_pri.sum()
        if total_high > 0:
            captured_high = top_20["priority_code"].astype(str).str.contains("E1|F1").sum()
            recall_20 = captured_high / total_high
            
    target_met = bool(recall_20 > 0.7)
    
    log.info("Isolation Forest Proxy Recall@20: %.3f", recall_20)
    if target_met:
        log.info("  ✅ PASSED Recall@20 target (>0.7)")
    else:
        log.warning("  ⚠️  DID NOT MEET Recall@20 target (>0.7)")
    
    return {
        "model": iso_forest,
        "scaler": scaler,
        "recall_20": float(recall_20),
        "target_met": target_met
    }


def save_artifacts(dbscan_res: dict, iso_res: dict, hotspot_df: pd.DataFrame, output_dir: Path):
    """Serialize clustering artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(dbscan_res["model"], output_dir / "dbscan_model.joblib")
    joblib.dump(dbscan_res["scaler"], output_dir / "dbscan_scaler.joblib")
    
    joblib.dump(iso_res["model"], output_dir / "isolation_forest_model.joblib")
    joblib.dump(iso_res["scaler"], output_dir / "isolation_forest_scaler.joblib")
    
    hotspots_path = output_dir / "hotspots.parquet"
    hotspot_df.to_parquet(hotspots_path, index=True)
    
    metrics = {
        "dbscan_silhouette": dbscan_res["silhouette_score"],
        "dbscan_clusters": dbscan_res["n_clusters"],
        "iso_forest_recall_20": iso_res["recall_20"]
    }
    
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    
    log.info("Saved clustering artifacts to %s", output_dir)


def log_to_mlflow(dbscan_res: dict, iso_res: dict):
    try:
        import mlflow
        mlflow.set_experiment("medalertai-clustering")
        with mlflow.start_run(run_name="hotspots_and_anomalies"):
            mlflow.log_metric("silhouette_score", dbscan_res["silhouette_score"])
            mlflow.log_metric("clusters", dbscan_res["n_clusters"])
            mlflow.log_metric("recall_20", iso_res["recall_20"])
            mlflow.log_param("eps", 0.3)
            mlflow.log_param("min_samples", 5)
            mlflow.log_param("contamination", 0.05)
            log.info("MLflow run logged successfully.")
    except Exception as exc:
        log.warning("MLflow logging failed (non-fatal): %s", exc)


def run_training(use_mlflow: bool = True):
    start = time.time()
    log.info("=== MedAlertAI Clustering Pipeline (Sanika) ===")
    
    hotspot_df, full_df = load_clustering_data()
    
    dbscan_res = train_dbscan(hotspot_df)
    iso_res = train_isolation_forest(full_df)
    
    save_artifacts(dbscan_res, iso_res, hotspot_df, CLUSTERING_ARTIFACTS_DIR)
    
    if use_mlflow:
        log_to_mlflow(dbscan_res, iso_res)
        
    log.info("=== Clustering Pipeline complete in %.1f seconds ===", time.time() - start)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="MedAlertAI — Phase 2 Clustering Training")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_training(use_mlflow=not args.no_mlflow)
