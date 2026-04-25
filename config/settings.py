"""
config/settings.py
Application-wide settings. Uses python-dotenv for environment overrides.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Flask / Dash ──
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", 8050))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
FLASK_THREADED = True

# ── Data Paths ──
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / os.getenv("MODEL_ARTIFACTS_DIR", "models/artifacts")

# ── ChromaDB ──
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

# ── API Keys ──
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HUGGINGFACE_ENDPOINT_URL = os.getenv("HUGGINGFACE_ENDPOINT_URL", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct:novita")

# ── Model Defaults ──
CLASSIFIER_TARGET = "mpds_group"
FORECAST_HORIZON_QUARTERS = 4
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 5
ISOLATION_FOREST_CONTAMINATION = 0.05

# ── RAG Evaluation Targets (Phase 3, C5) ──
RAG_PRECISION_AT_K_TARGET = 0.6          # Precision@5 > 0.6
RAG_FAITHFULNESS_AVG_TARGET = 1.5        # LLM-as-judge mean > 1.5 (0–3 scale)
RAG_LATENCY_P50_TARGET_S = 3.0           # p50 < 3 seconds
RAG_LATENCY_P95_TARGET_S = 8.0           # p95 < 8 seconds
RAG_EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
