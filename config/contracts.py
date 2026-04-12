"""
config/contracts.py
Shared schema contracts — NEMSIS v3 column names, Parquet dtypes,
and Dash callback input/output contracts.

ALL CONTRIBUTORS: Do not modify without team consensus.

Contributor Mapping:
  C1 — Greeshma   (Data Engineering & Ingestion)
  C2 — Suvarna    (ML — Classification & QA)
  C3 — Sanika     (ML — Forecasting & Clustering)
  C4 — Srileakhana (RAG Pipeline & LLM)
  C5 — Deekshitha (Dashboard & Testing)
"""

# ── Raw CSV Column Names (as-is from WPRDC) ──
RAW_COLUMNS = [
    "_id",
    "call_id_hash",
    "service",
    "priority",
    "priority_desc",
    "call_quarter",
    "call_year",
    "description_short",
    "city_code",
    "city_name",
    "geoid",
    "census_block_group_center__x",  # longitude
    "census_block_group_center__y",  # latitude
]

# ── NEMSIS v3-Aligned Column Mapping (raw → canonical) ──
COLUMN_MAPPING = {
    "call_id_hash": "incident_id",
    "service": "service_type",
    "priority": "priority_code",
    "priority_desc": "priority_description",
    "call_quarter": "quarter",
    "call_year": "year",
    "description_short": "call_type",
    "city_code": "city_code",
    "city_name": "city_name",
    "geoid": "census_block_group",
    "census_block_group_center__x": "longitude",
    "census_block_group_center__y": "latitude",
}

# ── Parquet Schema (dtypes for fact_dispatch_clean.parquet) ──
PARQUET_DTYPES = {
    "incident_id": "string",
    "service_type": "category",
    "priority_code": "string",
    "priority_description": "string",
    "quarter": "string",
    "year": "int16",
    "call_type": "string",
    "city_code": "string",
    "city_name": "string",
    "census_block_group": "string",
    "longitude": "float64",
    "latitude": "float64",
}

# ── Derived Temporal Columns (added during feature engineering) ──
DERIVED_COLUMNS = {
    "month": "int8",
    "hour": "int8",
    "day_of_week": "int8",
    "hour_sin": "float64",
    "hour_cos": "float64",
    "dow_sin": "float64",
    "dow_cos": "float64",
    "month_sin": "float64",
    "month_cos": "float64",
}

# ── MPDS Complaint Code Groups ──
MPDS_GROUPS = [
    "Abdominal Pain", "Allergies", "Animal Bites",
    "Assault", "Back Pain", "Breathing Problems",
    "Burns", "Carbon Monoxide", "Cardiac Arrest",
    "Chest Pain", "Choking", "Diabetic Problems",
    "Drowning", "Electrocution", "Eye Problems",
    "Falls", "Fire", "Headache",
    "Heart Problems", "Hemorrhage", "Overdose",
    "Psychiatric", "Seizures", "Stabbing",
    "Stroke", "Traffic Accident", "Unconscious",
    "Unknown Problem",
]

# ── Dash Callback Contracts ──
# Global filter store schema (shared via dcc.Store)
FILTER_STORE_SCHEMA = {
    "year": "list[int]",           # selected years
    "service_type": "list[str]",   # selected service types  (EMS / Fire)
    "mpds_group": "list[str]",     # selected MPDS groups
    "city_name": "list[str]",      # selected cities
}

# ── Data File Paths (relative to project root) ──
DATA_FILES = {
    "ems_raw": "data/raw/EMS_Data.csv",
    "fire_raw": "data/raw/Fire_Data.csv",
    "dispatch_clean": "data/processed/fact_dispatch_clean.parquet",
    "demographics": "data/processed/dim_cbg_demographics.parquet",
}
