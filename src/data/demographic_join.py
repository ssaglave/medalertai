"""
demographic_join.py — Census block-group demographic data join.

Owner: Srileakhana (C4)
Phase: 1

Responsibilities:
  - Fetch/load census block-group demographic data
    (poverty rate, population density, race/ethnicity)
  - Join with dispatch data on census_block_group (geoid)
  - Output: data/processed/dim_cbg_demographics.parquet
"""

import pandas as pd


def load_census_data() -> pd.DataFrame:
    """Load census block-group demographics from external source."""
    # TODO: Implement — can use Census API or pre-downloaded CSV
    pass


def join_demographics(dispatch_df: pd.DataFrame, census_df: pd.DataFrame) -> pd.DataFrame:
    """Join dispatch data with census demographics on census_block_group."""
    # TODO: Implement
    pass


def save_demographics_parquet(df: pd.DataFrame) -> None:
    """Save demographics dimension table as Parquet."""
    # TODO: Implement
    pass
