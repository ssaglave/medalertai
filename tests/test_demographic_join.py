import pandas as pd
import pytest

from src.data.demographic_join import (
    join_demographics,
    standardize_census_data,
)


def test_standardize_census_data_computes_phase_1_metrics():
    census = pd.DataFrame(
        {
            "GEOID": ["420030103001"],
            "NAME": ["Block Group 1, Census Tract 103, Allegheny County, Pennsylvania"],
            "B01003_001E": ["1000"],
            "B17001_001E": ["800"],
            "B17001_002E": ["120"],
            "B02001_002E": ["600"],
            "B02001_003E": ["250"],
            "B02001_005E": ["80"],
            "B03003_003E": ["70"],
            "B19013_001E": ["55000"],
            "ALAND": [2_000_000],
        }
    )

    demographics = standardize_census_data(census)

    row = demographics.iloc[0]
    assert row["census_block_group"] == "420030103001"
    assert row["population"] == 1000
    assert row["land_area_sq_km"] == 2
    assert row["population_density_per_sq_km"] == 500
    assert row["poverty_rate"] == pytest.approx(0.15)
    assert row["white_alone_pct"] == pytest.approx(0.60)
    assert row["black_alone_pct"] == pytest.approx(0.25)
    assert row["asian_alone_pct"] == pytest.approx(0.08)
    assert row["hispanic_or_latino_pct"] == pytest.approx(0.07)
    assert row["median_household_income"] == 55000


def test_join_demographics_normalizes_geoid_and_left_joins():
    dispatch = pd.DataFrame(
        {
            "call_id_hash": ["a", "b"],
            "geoid": [420030103001, 420039999999],
        }
    )
    census = pd.DataFrame(
        {
            "census_block_group": ["420030103001"],
            "population": [1000],
            "poverty_universe": [800],
            "poverty_count": [120],
            "white_alone_count": [600],
            "black_alone_count": [250],
            "asian_alone_count": [80],
            "hispanic_or_latino_count": [70],
            "land_area_sq_km": [2],
        }
    )

    joined = join_demographics(dispatch, census)

    assert len(joined) == 2
    assert joined.loc[0, "census_block_group"] == "420030103001"
    assert joined.loc[0, "poverty_rate"] == pytest.approx(0.15)
    assert pd.isna(joined.loc[1, "poverty_rate"])
