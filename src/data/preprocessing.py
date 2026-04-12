"""
preprocessing.py — NEMSIS-aligned data preprocessing.

Owner: Greeshma (C1)
Phase: 1

Responsibilities:
  - Load raw EMS_Data.csv and Fire_Data.csv
  - Normalize columns using config.contracts.COLUMN_MAPPING
  - Coerce dtypes per config.contracts.PARQUET_DTYPES
  - Handle missing values, duplicates, outliers
  - Output: data/processed/fact_dispatch_clean.parquet
"""

import pandas as pd
from config.contracts import COLUMN_MAPPING, PARQUET_DTYPES, DATA_FILES


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate raw EMS and Fire CSVs."""
    # TODO: Implement
    pass


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to NEMSIS v3 equivalents using COLUMN_MAPPING."""
    # TODO: Implement
    pass


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and type coercion."""
    # TODO: Implement
    pass


def save_parquet(df: pd.DataFrame, path: str = None) -> None:
    """Save cleaned DataFrame as Parquet."""
    # TODO: Implement
    pass


if __name__ == "__main__":
    df = load_raw_data()
    df = normalize_columns(df)
    df = clean_data(df)
    save_parquet(df)
    print(f"✅ Clean Parquet saved: {DATA_FILES['dispatch_clean']}")
