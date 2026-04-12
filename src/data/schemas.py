"""
schemas.py — Pydantic NEMSIS-aligned data validation schemas.

Owner: Deekshitha (C5)
Phase: 1

Responsibilities:
  - Define Pydantic models for dispatch records
  - data_completeness_pct scoring per row
  - Validate DataFrames against schema
"""

from pydantic import BaseModel, Field
from typing import Optional


class DispatchRecord(BaseModel):
    """Pydantic schema for a single cleaned dispatch record."""
    incident_id: str
    service_type: str
    priority_code: str
    priority_description: str
    quarter: str
    year: int
    call_type: str
    city_code: str
    city_name: str
    census_block_group: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None


def compute_completeness(record: dict) -> float:
    """Compute data completeness percentage for a single record (0.0–1.0)."""
    # TODO: Implement — count non-null fields / total fields
    pass


def validate_dataframe(df) -> dict:
    """Validate an entire DataFrame against DispatchRecord schema.
    Returns dict with 'valid_count', 'invalid_count', 'errors'.
    """
    # TODO: Implement
    pass
