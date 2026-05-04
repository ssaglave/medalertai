"""
demographic_join.py - Census block-group demographic data join.

Owner: Srileakhana (C4)
Phase: 1

Responsibilities:
  - Fetch/load census block-group demographic data
    (poverty rate, population density, race/ethnicity)
  - Join with dispatch data on census_block_group (geoid)
  - Output: data/processed/dim_cbg_demographics.parquet
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from config.contracts import DATA_FILES
from config.settings import PROCESSED_DATA_DIR, PROJECT_ROOT


DEFAULT_ACS_YEAR = "2022"
DEFAULT_STATE_FIPS = "42"  # Pennsylvania
DEFAULT_COUNTY_FIPS = "003"  # Allegheny County

CENSUS_COLUMNS = {
    "B01003_001E": "population",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_count",
    "B02001_002E": "white_alone_count",
    "B02001_003E": "black_alone_count",
    "B02001_005E": "asian_alone_count",
    "B03003_003E": "hispanic_or_latino_count",
    "B19013_001E": "median_household_income",
}

OUTPUT_COLUMNS = [
    "census_block_group",
    "census_name",
    "population",
    "land_area_sq_km",
    "population_density_per_sq_km",
    "poverty_rate",
    "white_alone_pct",
    "black_alone_pct",
    "asian_alone_pct",
    "hispanic_or_latino_pct",
    "median_household_income",
]


def _resolve_path(path: str | Path) -> Path:
    """Resolve repo-relative paths against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_geoid(value: object) -> str | pd.NA:
    """Normalize a census block-group GEOID to its 12-digit string form."""
    if pd.isna(value):
        return pd.NA

    geoid = str(value).strip()
    if not geoid:
        return pd.NA

    if geoid.endswith(".0"):
        geoid = geoid[:-2]

    digits = "".join(char for char in geoid if char.isdigit())
    if not digits:
        return pd.NA

    return digits.zfill(12)


def normalize_geoid_column(
    df: pd.DataFrame,
    source_col: str,
    target_col: str = "census_block_group",
) -> pd.DataFrame:
    """Return a copy with a normalized census block-group GEOID column."""
    if source_col not in df.columns:
        raise KeyError(f"Missing GEOID column: {source_col}")

    normalized = df.copy()
    normalized[target_col] = normalized[source_col].map(_normalize_geoid).astype("string")
    return normalized


def _read_local_census_file(path: str | Path) -> pd.DataFrame:
    """Read a local Census/demographic file from CSV or Parquet."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Census demographic file not found: {resolved}")

    if resolved.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, dtype={"geoid": "string", "GEOID": "string", "census_block_group": "string"})


def _fetch_acs_block_groups(
    year: str = DEFAULT_ACS_YEAR,
    state_fips: str = DEFAULT_STATE_FIPS,
    county_fips: str = DEFAULT_COUNTY_FIPS,
) -> pd.DataFrame:
    """Fetch ACS 5-year block-group demographics for a county from Census API."""
    variables = ["NAME", *CENSUS_COLUMNS.keys()]
    query = {
        "get": ",".join(variables),
        "for": "block group:*",
        "in": f"state:{state_fips} county:{county_fips} tract:*",
    }

    api_key = os.getenv("CENSUS_API_KEY")
    if api_key:
        query["key"] = api_key

    url = f"https://api.census.gov/data/{year}/acs/acs5?{urlencode(query)}"
    with urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    headers, rows = payload[0], payload[1:]
    df = pd.DataFrame(rows, columns=headers)
    df["census_block_group"] = df["state"] + df["county"] + df["tract"] + df["block group"]
    return df.rename(columns={"NAME": "census_name", **CENSUS_COLUMNS})


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    coerced = df.copy()
    for column in columns:
        if column in coerced.columns:
            coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced


def standardize_census_data(census_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Census data and calculate Phase 1 demographic features."""
    df = census_df.copy()

    if "census_block_group" not in df.columns:
        geoid_column = next((col for col in ("geoid", "GEOID") if col in df.columns), None)
        if geoid_column is None:
            raise KeyError("Census data must include census_block_group, geoid, or GEOID")
        df = normalize_geoid_column(df, geoid_column)
    else:
        df = normalize_geoid_column(df, "census_block_group")

    rename_map = {key: value for key, value in CENSUS_COLUMNS.items() if key in df.columns}
    df = df.rename(columns=rename_map)

    if "NAME" in df.columns and "census_name" not in df.columns:
        df = df.rename(columns={"NAME": "census_name"})
    if "census_name" not in df.columns:
        df["census_name"] = pd.NA

    numeric_columns = [
        "population",
        "poverty_universe",
        "poverty_count",
        "white_alone_count",
        "black_alone_count",
        "asian_alone_count",
        "hispanic_or_latino_count",
        "median_household_income",
        "land_area_sq_km",
        "land_area_sq_m",
        "ALAND",
    ]
    df = _coerce_numeric(df, numeric_columns)

    if "land_area_sq_km" not in df.columns:
        if "land_area_sq_m" in df.columns:
            df["land_area_sq_km"] = df["land_area_sq_m"] / 1_000_000
        elif "ALAND" in df.columns:
            df["land_area_sq_km"] = df["ALAND"] / 1_000_000
        else:
            df["land_area_sq_km"] = pd.NA

    if "population_density_per_sq_km" not in df.columns:
        df["population_density_per_sq_km"] = df["population"] / df["land_area_sq_km"].replace({0: pd.NA})

    df["poverty_rate"] = df["poverty_count"] / df["poverty_universe"].replace({0: pd.NA})
    df["white_alone_pct"] = df["white_alone_count"] / df["population"].replace({0: pd.NA})
    df["black_alone_pct"] = df["black_alone_count"] / df["population"].replace({0: pd.NA})
    df["asian_alone_pct"] = df["asian_alone_count"] / df["population"].replace({0: pd.NA})
    df["hispanic_or_latino_pct"] = df["hispanic_or_latino_count"] / df["population"].replace({0: pd.NA})

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    return df[OUTPUT_COLUMNS].drop_duplicates(subset=["census_block_group"]).reset_index(drop=True)


def load_census_data(
    path: str | Path | None = None,
    *,
    fetch_if_missing: bool = True,
    year: str = DEFAULT_ACS_YEAR,
    state_fips: str = DEFAULT_STATE_FIPS,
    county_fips: str = DEFAULT_COUNTY_FIPS,
) -> pd.DataFrame:
    """Load census block-group demographics from local data or Census API."""
    default_path = PROJECT_ROOT / "data" / "external" / "census_demographics.csv"
    source_path = _resolve_path(path) if path else default_path

    if source_path.exists():
        return standardize_census_data(_read_local_census_file(source_path))

    if not fetch_if_missing:
        raise FileNotFoundError(f"No local census demographics file found at {source_path}")

    return standardize_census_data(
        _fetch_acs_block_groups(year=year, state_fips=state_fips, county_fips=county_fips)
    )


def join_demographics(dispatch_df: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    """Join dispatch data with census demographics on census_block_group."""
    dispatch = dispatch_df.copy()
    if "census_block_group" not in dispatch.columns:
        if "geoid" not in dispatch.columns:
            raise KeyError("Dispatch data must include census_block_group or geoid")
        dispatch = normalize_geoid_column(dispatch, "geoid")
    else:
        dispatch = normalize_geoid_column(dispatch, "census_block_group")

    demographics = standardize_census_data(census_df)
    return dispatch.merge(demographics, on="census_block_group", how="left", validate="many_to_one")


def save_demographics_parquet(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Save demographics dimension table as Parquet and return the output path."""
    output_path = _resolve_path(path or DATA_FILES["demographics"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    standardize_census_data(df).to_parquet(output_path, index=False)
    return output_path


def build_demographics_dimension(
    census_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Build and save the Phase 1 demographics dimension parquet."""
    census_df = load_census_data(census_path)
    return save_demographics_parquet(census_df, output_path or PROCESSED_DATA_DIR / "dim_cbg_demographics.parquet")


if __name__ == "__main__":
    saved_path = build_demographics_dimension()
    print(f"Demographics Parquet saved: {saved_path}")
