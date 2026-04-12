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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Model Defaults ──
CLASSIFIER_TARGET = "mpds_group"
FORECAST_HORIZON_QUARTERS = 4
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 5
ISOLATION_FOREST_CONTAMINATION = 0.05
