"""
mpds_mapper.py — Map raw call_type (description_short) to MPDS complaint codes.

Owner: Suvarna (C2)
Phase: 1

Responsibilities:
  - Build mapping from raw description_short values to standardized MPDS groups
  - Target: >80% coverage of all call types
  - Unit tests for mapping accuracy

Approach:
  We use a two-tier matching strategy:
    1. EXACT_MAPPING — direct string-to-MPDS-group lookup for known call types.
    2. KEYWORD_RULES — ordered list of (keywords, MPDS group) pairs for substring
       matching, which catches variations (e.g., all TRAFFIC-* variants → Traffic Accident).

  Call types that don't match either tier are assigned "Unknown Problem".

  The mapper works on the raw `description_short` column as well as the NEMSIS-
  normalized `call_type` column (same values, different column name).
"""

import logging
from typing import Optional

import pandas as pd

from config.contracts import MPDS_GROUPS

logger = logging.getLogger(__name__)

# ── Default label for unmapped call types ──
UNMAPPED_LABEL = "Unknown Problem"

# ── Tier 1: Exact match mapping (upper-cased key → MPDS group) ──
# Covers ~85% of all rows by targeting the high-frequency call types.
EXACT_MAPPING: dict[str, str] = {
    # --- EMS Medical ---
    "SICK": "Sick Person",  # MPDS Protocol 26: Sick Person (non-specific)
    "FALL": "Falls",
    "BREATHING": "Breathing Problems",
    "UNCONSCIOUS": "Unconscious",
    "CHEST PAIN": "Chest Pain",
    "UNKNOWN": "Unknown Problem",
    "ABDOMINAL PAIN": "Abdominal Pain",
    "PSYCH": "Psychiatric",
    "CONVULSION": "Seizures",
    "CONVULSIONS": "Seizures",
    "OVERDOSE": "Overdose",
    "DIABETIC": "Diabetic Problems",
    "HEART": "Heart Problems",
    "STROKE": "Stroke",
    "HEMORRHAGE": "Hemorrhage",
    "ASSAULT": "Assault",
    "TRAUMA": "Falls",  # Trauma maps to Falls (injury mechanism)
    "BACK PAIN": "Back Pain",
    "ALLERGIES": "Allergies",
    "ALLERGIC REACTION": "Allergies",
    "HEADACHE": "Headache",
    "CHOKING": "Choking",
    "ANIMAL BITES": "Animal Bites",
    "EYE INJURY": "Eye Problems",
    "BURNS": "Burns",
    "DROWNING": "Drowning",
    "ELECTROCUTION OR LIGHTNING": "Electrocution",
    "EXTREME TEMPERATURE": "Unknown Problem",
    "PREGNANCY": "Abdominal Pain",  # Pregnancy complications → Abdominal Pain
    "POSSIBLE OR OBVIOUS DEATH": "Cardiac Arrest",
    "GUNSHOT, STABBING, OR OTHER WOUND": "Stabbing",

    # --- Fire/Hazmat ---
    "DWELLING FIRE": "Fire",
    "COMMERCIAL OR APARTMENT BLDG FIRE": "Fire",
    "POSS DWELLING FIRE": "Fire",
    "POSS COMMERCIAL OR APARTMENT BLDG FIRE": "Fire",
    "VEHICLE FIRE": "Fire",
    "BRUSH/GRASS/MULCH FIRE": "Fire",
    "FIRE UNCATEGORIZED": "Fire",
    "UNKNOWN TYPE FIRE": "Fire",
    "TRANSFORMER FIRE / EXPLOSION": "Fire",
    "WIRES FIRE/ARCING/UNK DANGER": "Fire",
    "GARAGE/AUXILIARY (NON COM) BLDG FIRE": "Fire",
    "POSS GARAGE/AUXILIARY (NON COM)BLDG FIRE": "Fire",
    "DUMPSTER FIRE": "Fire",
    "ILLEGAL FIRE": "Fire",
    "RAIL CAR / TRAIN FIRE": "Fire",
    "TUNNEL FIRE": "Fire",
    "EXTINGUISHED FIRE OUTSIDE": "Fire",
    "AIRCRAFT FIRE": "Fire",

    # --- Carbon Monoxide / Hazmat ---
    "CO OR HAZMAT ISSUE": "Carbon Monoxide",
    "NATURAL GAS ISSUE": "Carbon Monoxide",

    # --- Traffic / MVA ---
    "TRAFFIC -WITH INJURIES": "Traffic Accident",
    "TRAFFIC - UNKNOWN STATUS": "Traffic Accident",
    "TRAFFIC - 1ST PARTY/NOT DANGEROUS INJ": "Traffic Accident",
    "TRAFFIC - OTHER HAZARDS": "Traffic Accident",
    "TRAFFIC - NOT ALERT": "Traffic Accident",
    "TRAFFIC - ENTRAPMENT": "Traffic Accident",
    "TRAFFIC - SERIOUS HEMORRHAGE": "Traffic Accident",
    "TRAFFIC - HIGH MECHANISM": "Traffic Accident",
    "TRAFFIC - NO INJURIES REPORTED": "Traffic Accident",

    # --- EMS/Fire administrative ---
    "EMS CALL/ASSIST": "EMS Assist",
    "EMS ASSIST": "EMS Assist",
    "EMS CALL RINGDOWN": "EMS Assist",
    "NON EMERGENCY TRANSPORT": "Transfer/Transport",
    "MUTUAL AID": "EMS Assist",
    "MUTUAL AID REQUEST": "EMS Assist",
    "RQST ASST  EMS": "EMS Assist",
    "RQST ASST EMS": "EMS Assist",
    "RQST ASST  FIRE": "EMS Assist",
    "RQST ASST FIRE": "EMS Assist",
    "FIRE ALARM COM BLDG": "Fire",
    "FIRE ALARM RES BLDG": "Fire",
    "FIRE ALARM TESTING": "Fire",

    # --- Rescue / Access ---
    "ELEVATOR RESCUE": "Unknown Problem",
    "WATER RESCUE": "Drowning",
    "MASS CASUALTY INCIDENT": "Unknown Problem",

    # --- Administrative / Non-emergency ---
    "DETAIL": "EMS Assist",
    "PUBLIC SERVICE DETAILS": "EMS Assist",
    "CONTAINMENT/CLEAN UP DETAIL": "EMS Assist",
    "LANDING ZONE": "EMS Assist",
    "PHONE CALL": "Unknown Problem",
    "TRAINING": "Unknown Problem",
    "OUT AT": "Unknown Problem",
    "OUT OF SERVICE": "Unknown Problem",
    "ROAD CLOSED/OPENED": "Unknown Problem",
    "LOCKED OUT": "Unknown Problem",
    "LOCKED INSIDE": "Unknown Problem",
    "LOCKOUT/IN": "Unknown Problem",
    "STAND BY": "EMS Assist",
    "WIRE DOWN": "Fire",
    "WATER CONDITION INSIDE": "Unknown Problem",
    "FLOODING": "Unknown Problem",
    "SMOKE OUTSIDE - SEEN OR SMELLED": "Fire",
    "VEHICLE LEAKING GASOLINE": "Carbon Monoxide",
    "BOAT NON EMERGENCY": "Unknown Problem",
    "BOAT EMERGENCY": "Unknown Problem",
    "AIRPORT INSPECTION": "Unknown Problem",
    "AIRPORT ALERT 1": "Unknown Problem",
    "AIRPORT ALERT 2": "Unknown Problem",
    "AIRPORT ALERT 3": "Unknown Problem",
    "AIRPORT DISABLED AIRCRAFT": "Unknown Problem",
    "AIRPORT FAA TIME RESPONSE": "Unknown Problem",
    "AIRPORT AGC CRASH PHONE TEST": "Unknown Problem",
    "BAGGAGE SYSTEM EMERGENCY": "Unknown Problem",
    "TRAFFIC CONTROL / FIRE POLICE REQUEST": "Traffic Accident",
    "FIRE CALL RINGDOWN": "EMS Assist",
    "CALL OFF EMS": "Unknown Problem",
    "FUEL": "Carbon Monoxide",
    "BOMB THREAT": "Unknown Problem",
    "BOMB/POSS BOMB FOUND": "Unknown Problem",
    "HYDRANT NOTIFICATION / INSPECTIONS": "Unknown Problem",
    "NOTIFICATION": "Unknown Problem",
    "UTILITY COMPLAINT": "Unknown Problem",
    "VEHICLE MAINTENANCE": "Unknown Problem",
    "WELLPAD INCIDENT": "Carbon Monoxide",

    # --- Removed / redacted ---
    "Removed": "Unknown Problem",
}

# ── Tier 2: Keyword-based substring rules (checked in order) ──
# Each tuple: (list of keywords ALL of which must appear, MPDS group)
# Order matters — first match wins.
KEYWORD_RULES: list[tuple[list[str], str]] = [
    # Traffic variants (many compound names)
    (["TRAFFIC"], "Traffic Accident"),

    # Inaccessible incidents (rescue scenarios)
    (["INACCESS", "ENTRAP"], "Unknown Problem"),
    (["INACCESS", "CONFINED"], "Unknown Problem"),
    (["INACCESS", "TRENCH"], "Unknown Problem"),
    (["INACCESS", "MUDSLIDE"], "Unknown Problem"),
    (["INACCESS", "AVALANCHE"], "Unknown Problem"),
    (["INACCESS"], "Unknown Problem"),

    # Fire-related keywords
    (["FIRE"], "Fire"),
    (["ARCING"], "Fire"),

    # EMS-related keywords
    (["CONVULS"], "Seizures"),
    (["HEMORRHAG"], "Hemorrhage"),
    (["BLEED"], "Hemorrhage"),
]


def _normalize_call_type(call_type: str) -> str:
    """Normalize a call type string: strip whitespace and uppercase."""
    if not isinstance(call_type, str):
        return ""
    return call_type.strip().upper()


# ── Build the cached mapping at module load time ──
_CACHED_MAPPING: Optional[dict[str, str]] = None


def build_mpds_mapping() -> dict[str, str]:
    """Create a mapping dictionary from raw call_type → MPDS group.

    Uses the two-tier strategy:
      1. Exact match lookup (covers high-frequency call types)
      2. Keyword substring rules (catches long-tail variants)

    Returns:
        dict mapping uppercased call_type strings → MPDS group names.
        This is the Tier 1 exact mapping. Tier 2 keyword rules are
        applied dynamically in `map_call_to_mpds()`.
    """
    # Return the exact mapping dict (uppercased keys)
    mapping = {}
    for raw_key, group in EXACT_MAPPING.items():
        normalized = _normalize_call_type(raw_key)
        if normalized:
            mapping[normalized] = group
    return mapping


def map_call_to_mpds(call_type: str) -> str:
    """Map a single call_type string to its MPDS group.

    Strategy:
      1. Try exact match in EXACT_MAPPING
      2. Try keyword substring rules in KEYWORD_RULES
      3. Fall back to "Unknown Problem"

    Args:
        call_type: Raw description_short or normalized call_type value.

    Returns:
        MPDS complaint group name (one of config.contracts.MPDS_GROUPS,
        or "Unknown Problem" for unmapped types).
    """
    global _CACHED_MAPPING
    if _CACHED_MAPPING is None:
        _CACHED_MAPPING = build_mpds_mapping()

    normalized = _normalize_call_type(call_type)
    if not normalized:
        return UNMAPPED_LABEL

    # Tier 1: Exact match
    if normalized in _CACHED_MAPPING:
        return _CACHED_MAPPING[normalized]

    # Tier 2: Keyword substring matching
    for keywords, group in KEYWORD_RULES:
        if all(kw in normalized for kw in keywords):
            return group

    # No match found
    return UNMAPPED_LABEL


def map_dataframe(
    df: pd.DataFrame,
    source_col: str = "description_short",
    target_col: str = "mpds_group",
) -> pd.DataFrame:
    """Apply MPDS mapping to an entire DataFrame.

    Args:
        df: Input DataFrame with a call_type column.
        source_col: Name of the column containing raw call types.
            Defaults to "description_short" (raw CSV column name).
            Also accepts "call_type" (NEMSIS-normalized name).
        target_col: Name of the new column to create with MPDS groups.

    Returns:
        DataFrame with the new `target_col` column added.
    """
    df = df.copy()
    df[target_col] = df[source_col].apply(map_call_to_mpds)
    return df


def compute_coverage(
    df: pd.DataFrame,
    source_col: str = "description_short",
    target_col: str = "mpds_group",
) -> float:
    """Calculate the percentage of rows successfully mapped to MPDS codes.

    A row is considered "mapped" if its MPDS group is NOT "Unknown Problem"
    and is one of the recognized MPDS_GROUPS from contracts.

    Args:
        df: DataFrame — must already have the target_col, OR have the
            source_col so we can map it first.
        source_col: Column containing raw call types.
        target_col: Column containing (or to contain) MPDS groups.

    Returns:
        Float between 0.0 and 1.0 representing the fraction of mapped rows.
    """
    if target_col not in df.columns:
        df = map_dataframe(df, source_col=source_col, target_col=target_col)

    total = len(df)
    if total == 0:
        return 0.0

    mapped = df[target_col] != UNMAPPED_LABEL
    coverage = mapped.sum() / total

    logger.info(
        "MPDS coverage: %.2f%% (%d / %d rows mapped)",
        coverage * 100,
        mapped.sum(),
        total,
    )
    return coverage


def get_mapping_report(df: pd.DataFrame, source_col: str = "description_short") -> pd.DataFrame:
    """Generate a detailed report showing each raw call type, its MPDS mapping,
    row count, and percentage of total.

    Useful for auditing coverage and identifying unmapped call types.

    Args:
        df: Input DataFrame with the source column.
        source_col: Column containing raw call type descriptions.

    Returns:
        DataFrame with columns: call_type, mpds_group, count, pct, is_mapped
        sorted by count descending.
    """
    mapped_df = map_dataframe(df, source_col=source_col)
    total = len(mapped_df)

    report = (
        mapped_df
        .groupby([source_col, "mpds_group"])
        .size()
        .reset_index(name="count")
        .rename(columns={source_col: "call_type"})
    )
    report["pct"] = (report["count"] / total * 100).round(2)
    report["is_mapped"] = report["mpds_group"] != UNMAPPED_LABEL
    report = report.sort_values("count", ascending=False).reset_index(drop=True)

    return report
