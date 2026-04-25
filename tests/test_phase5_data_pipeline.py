"""
Phase 5 data pipeline validation tests.

Owner: Greeshma (C1)

Scope:
  - NEMSIS schema normalization
  - Pydantic validation on canonical dispatch records
  - Data completeness scoring
  - Clean parquet integration checks when local artifacts exist
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.settings import PROJECT_ROOT
from src.data.schemas import (
    add_completeness_column,
    compute_completeness,
    normalize_dataframe,
    validate_dataframe,
    validate_parquet_schema,
)


@pytest.fixture
def canonical_dispatch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "incident_id": "abc123",
                "service_type": "EMS",
                "priority_code": "E1",
                "priority_description": "Life Threatening",
                "quarter": "Q1",
                "year": 2023,
                "call_type": "FALL",
                "city_code": "PIT",
                "city_name": "Pittsburgh",
                "census_block_group": "420030103001",
                "longitude": -79.99,
                "latitude": 40.44,
            },
            {
                "incident_id": "def456",
                "service_type": "Fire",
                "priority_code": "F3",
                "priority_description": "Routine",
                "quarter": "Q2",
                "year": 2024,
                "call_type": "DWELLING FIRE",
                "city_code": "PIT",
                "city_name": "Pittsburgh",
                "census_block_group": "420030103002",
                "longitude": -79.98,
                "latitude": 40.45,
            },
        ]
    )


@pytest.fixture
def raw_style_dispatch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "call_id_hash": "ghi789",
                "service": "EMS",
                "priority": "E2",
                "priority_desc": "Urgent",
                "call_quarter": 3,
                "call_year": 2022,
                "description_short": "CHEST PAIN",
                "city_code": "MTL",
                "city_name": "Mount Lebanon",
                "geoid": "420030201001",
                "census_block_group_center__x": -80.04,
                "census_block_group_center__y": 40.38,
            }
        ]
    )


@pytest.fixture
def clean_dispatch_parquet() -> Path:
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "fact_dispatch_clean.parquet",
        PROJECT_ROOT.parent / "medalertai" / "data" / "processed" / "fact_dispatch_clean.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip("fact_dispatch_clean.parquet not found locally - skipping parquet integration tests")


class TestSchemaNormalization:
    def test_normalize_dataframe_renames_expected_columns(self, raw_style_dispatch_df: pd.DataFrame):
        normalized = normalize_dataframe(raw_style_dispatch_df)
        expected = {
            "incident_id",
            "service_type",
            "priority_code",
            "priority_description",
            "quarter",
            "year",
            "call_type",
            "city_code",
            "city_name",
            "census_block_group",
            "longitude",
            "latitude",
        }
        assert expected.issubset(set(normalized.columns))

    def test_normalize_dataframe_maps_integer_quarters(self, raw_style_dispatch_df: pd.DataFrame):
        normalized = normalize_dataframe(raw_style_dispatch_df)
        assert normalized.loc[0, "quarter"] == "Q3"


class TestCompletenessScoring:
    def test_compute_completeness_full_record(self, canonical_dispatch_df: pd.DataFrame):
        score = compute_completeness(canonical_dispatch_df.iloc[0].to_dict())
        assert score == 1.0

    def test_compute_completeness_missing_optional_fields(self, canonical_dispatch_df: pd.DataFrame):
        record = canonical_dispatch_df.iloc[0].to_dict()
        record["longitude"] = None
        record["latitude"] = None
        score = compute_completeness(record)
        assert score == round(10 / 12, 4)

    def test_add_completeness_column_adds_expected_range(self, canonical_dispatch_df: pd.DataFrame):
        df = canonical_dispatch_df.copy()
        df.loc[1, "city_name"] = ""
        scored = add_completeness_column(df)
        assert "data_completeness_pct" in scored.columns
        assert scored["data_completeness_pct"].between(0.0, 1.0).all()
        assert scored.loc[0, "data_completeness_pct"] > scored.loc[1, "data_completeness_pct"]


class TestPydanticValidation:
    def test_validate_dataframe_accepts_valid_canonical_records(self, canonical_dispatch_df: pd.DataFrame):
        result = validate_dataframe(canonical_dispatch_df)
        assert result["invalid_count"] == 0
        assert result["valid_count"] == len(canonical_dispatch_df)
        assert result["mean_completeness"] == 1.0

    def test_validate_dataframe_rejects_invalid_quarter(self, canonical_dispatch_df: pd.DataFrame):
        df = canonical_dispatch_df.copy()
        df.loc[0, "quarter"] = "Q5"
        result = validate_dataframe(df)
        assert result["invalid_count"] == 1
        assert "quarter" in result["errors"][0]["error"]


class TestCleanParquetIntegration:
    @pytest.mark.slow
    def test_clean_parquet_schema_matches_contracts(self, clean_dispatch_parquet: Path):
        df = pd.read_parquet(clean_dispatch_parquet)
        result = validate_parquet_schema(df)
        assert result["missing_columns"] == []
        allowed = [
            {
                "column": "service_type",
                "expected": "category",
                "actual": "object",
            }
        ]
        assert result["dtype_mismatches"] in ([], allowed)

    @pytest.mark.slow
    def test_clean_parquet_has_completeness_score_column(self, clean_dispatch_parquet: Path):
        df = pd.read_parquet(clean_dispatch_parquet, columns=["completeness_score"])
        assert "completeness_score" in df.columns
        assert df["completeness_score"].between(0.0, 1.0).all()

    @pytest.mark.slow
    def test_clean_parquet_average_completeness_is_reasonable(self, clean_dispatch_parquet: Path):
        df = pd.read_parquet(clean_dispatch_parquet, columns=["completeness_score"])
        mean_score = float(df["completeness_score"].mean())
        assert mean_score >= 0.70, (
            f"Average completeness_score is {mean_score:.3f}, below the expected 0.70 minimum."
        )
