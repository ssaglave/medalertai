"""
tests/test_rag_mpds_nemsis_ingest.py

End-to-end tests for the MPDS / NEMSIS additions to the RAG ingestion
pipeline (src/rag/ingest.py).

Owner: Suvarna (C2)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag import ingest
from src.rag.ingest import DEFAULT_SOURCES, build_chunks, load_documents


MPDS_PROTOCOL_BODY = """# MPDS Protocol Reference

## Protocol 6 — Breathing Problems

Determinant levels: Alpha (low), Bravo, Charlie, Delta, Echo (highest).
Dispatchers ask key questions about airway patency, breathing effort, and skin
color before assigning a determinant code. The Echo level corresponds to
ineffective breathing or full respiratory arrest and triggers the highest
priority response with full ALS resources.

## Protocol 9 — Cardiac or Respiratory Arrest

Echo level dispatch with simultaneous CPR pre-arrival instructions. Cardiac
arrest scenarios always escalate to a full advanced life support response
including paramedic units and supervisor notification.
"""

MPDS_DISPATCH_BODY = """# MPDS Dispatch Mapping

The Pittsburgh raw call type "MEDIC EMERGENCY" maps to MPDS Protocol 32
(Unknown Problem) by default until a more specific code is determined.
"FIRE STILL ALARM" maps to MPDS Protocol 69 (Structure Fire) when there is a
visible fire, or to Protocol 53 (Outside Fire) for non-structure incidents.
This document explains the rationale used by the project mapper.
"""

NEMSIS_DICTIONARY_BODY = """# NEMSIS v3 Data Dictionary

## eResponse.05 — Type of Service Requested

A categorical element describing the type of EMS service requested at the time
of dispatch. Common values include 911 Response (Scene), Interfacility
Transport, and Mutual Aid. Each value has a defined NEMSIS code that maps to
the project's normalized service_type column.

## eDispatch.01 — Dispatch Reason

The chief complaint or reason recorded at dispatch, normalized to MPDS
protocol categories where possible. This element is the primary join key
between dispatch records and the MPDS protocol reference.
"""

NEMSIS_REFERENCE_BODY = """# NEMSIS v3 Reference Overview

The National EMS Information System (NEMSIS) defines a national standard for
EMS data collection and exchange. Version 3 of the standard introduces an XML
schema with strongly typed elements, value sets, and usage guidance. This
overview document summarizes how the MedAlertAI project aligns Pittsburgh EMS
and Fire dispatch records with the NEMSIS v3 canonical schema for downstream
modeling and reporting.
"""

NEW_SOURCE_IDS = (
    "mpds_protocol_reference",
    "mpds_dispatch_mapping",
    "nemsis_v3_data_dictionary",
    "nemsis_v3_reference",
)

BODIES = {
    "mpds_protocol_reference": MPDS_PROTOCOL_BODY,
    "mpds_dispatch_mapping": MPDS_DISPATCH_BODY,
    "nemsis_v3_data_dictionary": NEMSIS_DICTIONARY_BODY,
    "nemsis_v3_reference": NEMSIS_REFERENCE_BODY,
}


@pytest.fixture
def staged_sources(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "source_documents"
    source_dir.mkdir()
    for source_id, body in BODIES.items():
        (source_dir / f"{source_id}.md").write_text(body, encoding="utf-8")
    registry_path = source_dir / "source_registry.json"
    registry_path.write_text(json.dumps(DEFAULT_SOURCES, indent=2), encoding="utf-8")
    return source_dir, registry_path


def test_default_sources_include_new_entries() -> None:
    source_ids = {entry["source_id"] for entry in DEFAULT_SOURCES}
    for new_id in NEW_SOURCE_IDS:
        assert new_id in source_ids


def test_default_sources_have_expected_categories() -> None:
    by_id = {entry["source_id"]: entry for entry in DEFAULT_SOURCES}
    assert by_id["mpds_protocol_reference"]["category"] == "mpds_protocols"
    assert by_id["mpds_dispatch_mapping"]["category"] == "mpds_protocols"
    assert by_id["nemsis_v3_data_dictionary"]["category"] == "nemsis_standard"
    assert by_id["nemsis_v3_reference"]["category"] == "nemsis_standard"


def test_committed_registry_matches_defaults() -> None:
    registry_path = ingest.SOURCE_LIST_PATH
    on_disk = json.loads(registry_path.read_text(encoding="utf-8"))
    assert {entry["source_id"] for entry in on_disk} == {
        entry["source_id"] for entry in DEFAULT_SOURCES
    }


def test_load_documents_finds_new_source_files(staged_sources: tuple[Path, Path]) -> None:
    source_dir, registry_path = staged_sources
    docs = load_documents(input_dir=source_dir, registry_path=registry_path)
    found_ids = {doc.source_id for doc in docs}
    for new_id in NEW_SOURCE_IDS:
        assert new_id in found_ids


def test_build_chunks_produces_chunks_with_correct_metadata(
    staged_sources: tuple[Path, Path],
) -> None:
    source_dir, registry_path = staged_sources
    docs = load_documents(input_dir=source_dir, registry_path=registry_path)
    chunks = build_chunks(docs)
    assert chunks, "expected at least one chunk to be produced"

    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk["source_id"], []).append(chunk)

    for new_id in NEW_SOURCE_IDS:
        assert new_id in by_source, f"missing chunks for {new_id}"
        sample = by_source[new_id][0]
        assert sample["text"].strip()
        assert sample["metadata"]["category"] in {"mpds_protocols", "nemsis_standard"}
        assert sample["metadata"]["file_name"].endswith(".md")


def test_build_chunks_filters_garbled_chunk(staged_sources: tuple[Path, Path]) -> None:
    source_dir, registry_path = staged_sources
    garbled = "Th\ufffd " * 200
    (source_dir / "mpds_protocol_reference.md").write_text(
        MPDS_PROTOCOL_BODY + "\n\n" + garbled, encoding="utf-8"
    )
    docs = load_documents(input_dir=source_dir, registry_path=registry_path)
    chunks = build_chunks(docs)
    for chunk in chunks:
        assert "\ufffd" not in chunk["text"], "garbled chunk leaked through validate_chunk"
