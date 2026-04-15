"""
schemas.py — Pydantic NEMSIS-aligned data validation schemas.

Owner: Deekshitha (C5)
Phase: 1

Responsibilities:
  - Define Pydantic models for dispatch records
  - data_completeness_pct scoring per row
  - Validate DataFrames against schema
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np


# ── Valid value enumerations (aligned with config/contracts.py) ──
VALID_SERVICE_TYPES = {"EMS", "Fire"}
VALID_QUARTERS = {"Q1", "Q2", "Q3", "Q4"}
YEAR_MIN = 2000
YEAR_MAX = 2030

# Fields used for completeness scoring — core fields expected to be populated
CORE_FIELDS = [
    "incident_id",
    "service_type",
    "priority_code",
    "priority_description",
    "quarter",
    "year",
    "call_type",
    "city_code",
    "city_name",
]

OPTIONAL_FIELDS = [
    "census_block_group",
    "longitude",
    "latitude",
]

ALL_FIELDS = CORE_FIELDS + OPTIONAL_FIELDS

# ── Quarter integer → string mapping ──
QUARTER_INT_TO_STR = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


# ── Column Normalization (raw → NEMSIS) ──


def rename_to_nemsis(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw / preprocessing column names to NEMSIS-aligned names.

    Uses the shared ``COLUMN_MAPPING`` from ``config/contracts.py`` so that
    DataFrames produced by Greeshma's ``preprocessing.py`` (which keeps raw
    column names like ``call_id_hash``, ``service``, ``description_short``)
    are converted to the canonical NEMSIS names used by ``DispatchRecord``,
    the dashboard pages, and all downstream consumers.

    Columns not in the mapping are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with NEMSIS-aligned column names.
    """
    try:
        from config.contracts import COLUMN_MAPPING
    except ImportError:
        COLUMN_MAPPING = {}

    # Only rename columns that actually exist in the DataFrame
    rename_map = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
    return df.rename(columns=rename_map)


def map_quarter_values(df: pd.DataFrame, col: str = "quarter") -> pd.DataFrame:
    """Convert integer quarter values (1, 2, 3, 4) to strings ("Q1"–"Q4").

    Greeshma's preprocessing outputs ``call_quarter`` as int8 (1–4).
    The NEMSIS schema and dashboard pages expect string format ("Q1"–"Q4").
    This function handles both cases gracefully:
      - If values are already strings like "Q1", they are left as-is.
      - If values are ints (1–4), they are mapped to "Q1"–"Q4".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the quarter column.
    col : str
        Name of the quarter column (default: ``"quarter"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with quarter values as strings.
    """
    if col not in df.columns:
        return df

    df = df.copy()

    # Check if already string format ("Q1", "Q2", etc.)
    sample = df[col].dropna().head(10)
    if len(sample) > 0 and sample.astype(str).str.match(r"^Q[1-4]$").all():
        # Already in "Q1"–"Q4" format
        return df

    # Map int → str, handling NaN and nullable int types
    df[col] = (
        pd.to_numeric(df[col], errors="coerce")
        .map(QUARTER_INT_TO_STR)
        .astype("string")
    )
    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: rename columns to NEMSIS names + fix quarter format.

    Combines ``rename_to_nemsis()`` and ``map_quarter_values()`` into a
    single call. Use this when loading data from Greeshma's
    ``fact_dispatch_clean.parquet`` before passing it to dashboards,
    validation, or completeness scoring.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw / preprocessing column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with NEMSIS-aligned column names and "Q1"–"Q4" quarters.
    """
    df = rename_to_nemsis(df)
    df = map_quarter_values(df)
    return df


class DispatchRecord(BaseModel):
    """Pydantic schema for a single cleaned dispatch record.

    Field names match the NEMSIS v3-aligned column mapping defined in
    ``config/contracts.py::COLUMN_MAPPING``.
    """

    incident_id: str = Field(..., description="Anonymized incident identifier")
    service_type: str = Field(..., description="EMS or Fire")
    priority_code: str = Field(..., description="Priority code (e.g., E4, F1)")
    priority_description: str = Field(..., description="Priority description text")
    quarter: str = Field(..., description="Quarter of the year (Q1–Q4)")
    year: int = Field(..., ge=YEAR_MIN, le=YEAR_MAX, description="Call year")
    call_type: str = Field(..., description="Short call description / MPDS type")
    city_code: str = Field(..., description="City abbreviation")
    city_name: str = Field(..., description="City full name")
    census_block_group: Optional[str] = Field(
        None, description="Census block group FIPS code (geoid)"
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Block group centroid longitude"
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Block group centroid latitude"
    )

    # ── Validators ──

    @field_validator("service_type")
    @classmethod
    def validate_service_type(cls, v: str) -> str:
        """Ensure service_type is either 'EMS' or 'Fire'."""
        if v not in VALID_SERVICE_TYPES:
            raise ValueError(
                f"service_type must be one of {VALID_SERVICE_TYPES}, got '{v}'"
            )
        return v

    @field_validator("quarter")
    @classmethod
    def validate_quarter(cls, v: str) -> str:
        """Ensure quarter is Q1–Q4."""
        if v not in VALID_QUARTERS:
            raise ValueError(f"quarter must be one of {VALID_QUARTERS}, got '{v}'")
        return v

    @field_validator("incident_id", "priority_code", "call_type", "city_code", "city_name")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field must not be empty or whitespace-only")
        return v.strip()


# ── Completeness Scoring ──


def compute_completeness(record: dict) -> float:
    """Compute data completeness percentage for a single record.

    Returns a float between 0.0 and 1.0 representing the fraction of
    ALL_FIELDS that have non-null, non-empty values.

    Parameters
    ----------
    record : dict
        A dictionary representing a single dispatch record.

    Returns
    -------
    float
        Completeness score (0.0–1.0).
    """
    total = len(ALL_FIELDS)
    if total == 0:
        return 1.0

    filled = 0
    for field in ALL_FIELDS:
        value = record.get(field)
        if value is not None:
            # Treat NaN, empty strings, and whitespace-only strings as missing
            if isinstance(value, float) and np.isnan(value):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            filled += 1

    return round(filled / total, 4)


def add_completeness_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``data_completeness_pct`` column to a DataFrame.

    Uses vectorized operations for efficiency on large DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with NEMSIS-aligned column names.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``data_completeness_pct`` column (0.0–1.0).
    """
    available_fields = [f for f in ALL_FIELDS if f in df.columns]
    total = len(available_fields)

    if total == 0:
        df["data_completeness_pct"] = 0.0
        return df

    # Count non-null, non-empty values per row (vectorized)
    filled_counts = pd.Series(0, index=df.index)
    for field in available_fields:
        col = df[field]
        if col.dtype == "object" or str(col.dtype) == "string":
            # For string columns: not null AND not empty/whitespace
            is_filled = col.notna() & col.astype(str).str.strip().ne("")
        else:
            is_filled = col.notna()
        filled_counts += is_filled.astype(int)

    df["data_completeness_pct"] = (filled_counts / total).round(4)
    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validate an entire DataFrame against the DispatchRecord schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with NEMSIS-aligned column names.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``valid_count`` (int): number of rows passing validation
        - ``invalid_count`` (int): number of rows failing validation
        - ``total_count`` (int): total number of rows
        - ``errors`` (list[dict]): list of ``{"row_index": int, "error": str}``
        - ``completeness_scores`` (list[float]): per-row completeness scores
        - ``mean_completeness`` (float): average completeness across all rows
    """
    valid_count = 0
    invalid_count = 0
    errors: List[Dict[str, Any]] = []
    completeness_scores: List[float] = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()

        # Compute completeness for every row (even invalid ones)
        score = compute_completeness(row_dict)
        completeness_scores.append(score)

        # Validate against Pydantic schema
        try:
            DispatchRecord(**row_dict)
            valid_count += 1
        except Exception as e:
            invalid_count += 1
            # Limit stored errors to first 100 to avoid memory issues
            if len(errors) < 100:
                errors.append({"row_index": int(idx), "error": str(e)})

    mean_completeness = (
        round(sum(completeness_scores) / len(completeness_scores), 4)
        if completeness_scores
        else 0.0
    )

    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "total_count": valid_count + invalid_count,
        "errors": errors,
        "completeness_scores": completeness_scores,
        "mean_completeness": mean_completeness,
    }


def validate_parquet_schema(df: pd.DataFrame) -> dict:
    """Check that a DataFrame's columns match the expected Parquet schema.

    Compares against ``config/contracts.py::PARQUET_DTYPES``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``missing_columns`` (list[str]): expected columns not in df
        - ``extra_columns`` (list[str]): columns in df not in the schema
        - ``dtype_mismatches`` (list[dict]): columns where dtype differs
        - ``is_valid`` (bool): True if no missing columns and no mismatches
    """
    try:
        from config.contracts import PARQUET_DTYPES
    except ImportError:
        # Fallback if running outside project context
        PARQUET_DTYPES = {}

    expected_cols = set(PARQUET_DTYPES.keys())
    actual_cols = set(df.columns)

    missing = sorted(expected_cols - actual_cols)
    extra = sorted(actual_cols - expected_cols)

    dtype_mismatches = []
    for col in expected_cols & actual_cols:
        expected_dtype = PARQUET_DTYPES[col]
        actual_dtype = str(df[col].dtype)
        # Loose match: allow e.g. "int16" to match "int64", "string" to match "object"
        if not _dtype_compatible(expected_dtype, actual_dtype):
            dtype_mismatches.append(
                {
                    "column": col,
                    "expected": expected_dtype,
                    "actual": actual_dtype,
                }
            )

    return {
        "missing_columns": missing,
        "extra_columns": extra,
        "dtype_mismatches": dtype_mismatches,
        "is_valid": len(missing) == 0 and len(dtype_mismatches) == 0,
    }


def _dtype_compatible(expected: str, actual: str) -> bool:
    """Check if an actual dtype is compatible with the expected dtype string.

    Allows flexible matching — e.g., 'int16' matches 'int64', 'string' matches
    'object', etc.
    """
    expected = expected.lower()
    actual = actual.lower()

    # Exact match
    if expected == actual:
        return True

    # Integer family
    int_types = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    if expected in int_types and actual in int_types:
        return True

    # Float family
    float_types = {"float16", "float32", "float64"}
    if expected in float_types and actual in float_types:
        return True

    # String family
    string_types = {"string", "object", "str"}
    if expected in string_types and actual in string_types:
        return True

    # Category
    if expected == "category" and actual == "category":
        return True

    return False
