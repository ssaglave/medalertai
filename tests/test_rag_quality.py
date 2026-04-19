"""
tests/test_rag_quality.py

Tests for src/rag/quality.py — OCR garble detection, chunk validation,
and quality reporting.

Owner: Suvarna (C2)
"""

import pytest

from src.rag.quality import (
    detect_garbled_ocr,
    generate_quality_report,
    validate_chunk,
)


CLEAN_PARAGRAPH = (
    "MPDS Protocol 6 covers breathing problems. Determinant levels range from "
    "Alpha (low priority) through Echo (cardiac or respiratory arrest). "
    "Dispatchers ask key questions about airway patency, breathing effort, "
    "and skin color before assigning a determinant code."
)

CLEAN_NEMSIS_PARAGRAPH = (
    "The NEMSIS v3 data dictionary defines core elements such as eResponse.05 "
    "(Type of Service Requested) and eDispatch.01 (Dispatch Reason). Each "
    "element has a defined value set, data type, and usage guidance for "
    "compliant EMS reporting across the United States."
)


@pytest.mark.parametrize("text", [CLEAN_PARAGRAPH, CLEAN_NEMSIS_PARAGRAPH])
def test_clean_text_is_not_garbled(text: str) -> None:
    result = detect_garbled_ocr(text)
    assert result["is_garbled"] is False
    assert result["has_garble_marker"] is False
    assert result["non_ascii_ratio"] <= 0.10


@pytest.mark.parametrize(
    "text",
    [
        "Th\ufffd qu\ufffdck br\ufffdwn f\ufffdx jumps over the lazy dog.",
        "ÿþABCDEFGH some text after the BOM-style garble marker here.",
        "þÿPage header with replacement bytes preceding the actual content body.",
    ],
)
def test_garble_markers_are_detected(text: str) -> None:
    result = detect_garbled_ocr(text)
    assert result["is_garbled"] is True
    assert result["has_garble_marker"] is True


def test_vowel_less_gibberish_is_flagged() -> None:
    text = "qwrtp lkjhgf zxcvbn mnbvc qwrt plkj zxcv bnm pqrs xyzz wxyz mnbv"
    result = detect_garbled_ocr(text)
    assert result["is_garbled"] is True
    assert result["vowel_less_token_ratio"] > 0.30


def test_long_non_ascii_run_is_flagged() -> None:
    text = "Header line followed by garble bytes ÆØÅÞßæøåþ then more body text content."
    result = detect_garbled_ocr(text)
    assert result["is_garbled"] is True
    assert result["has_long_special_run"] is True


def test_markdown_table_separator_is_not_flagged() -> None:
    text = (
        "| Level | Code | Name | Description | Typical Response |\n"
        "|-------|------|------|-------------|------------------|\n"
        "| 1 | E | Echo | Immediately life-threatening | Full ALS dispatched |\n"
        "| 2 | D | Delta | Time-critical | ALS dispatched |\n"
    )
    result = detect_garbled_ocr(text)
    assert result["is_garbled"] is False
    assert result["has_long_special_run"] is False


def test_validate_chunk_passes_clean_paragraph() -> None:
    result = validate_chunk(CLEAN_PARAGRAPH)
    assert result["passes"] is True
    assert result["reasons"] == []
    assert result["word_count"] >= 8


def test_validate_chunk_rejects_short_text() -> None:
    result = validate_chunk("Too short.")
    assert result["passes"] is False
    assert "too_short" in result["reasons"]


def test_validate_chunk_rejects_empty() -> None:
    result = validate_chunk("")
    assert result["passes"] is False
    assert "empty" in result["reasons"]


def test_validate_chunk_rejects_garbled_chunk() -> None:
    text = (
        "MPDS Protocol 6 covers \ufffd\ufffd\ufffd breathing problems "
        "with determinants \ufffd\ufffd Alpha through Echo and key questions."
    )
    result = validate_chunk(text)
    assert result["passes"] is False
    assert "garbled_ocr" in result["reasons"]


def test_validate_chunk_rejects_too_few_words() -> None:
    text = "x " * 80
    result = validate_chunk(text)
    assert result["passes"] is False
    assert "too_few_words" in result["reasons"]


def test_generate_quality_report_summarises_chunks() -> None:
    chunks = [
        {"chunk_id": "a1", "source_id": "mpds_protocol_reference", "text": CLEAN_PARAGRAPH},
        {"chunk_id": "a2", "source_id": "mpds_protocol_reference", "text": CLEAN_PARAGRAPH},
        {"chunk_id": "b1", "source_id": "nemsis_v3_data_dictionary", "text": CLEAN_NEMSIS_PARAGRAPH},
        {"chunk_id": "b2", "source_id": "nemsis_v3_data_dictionary", "text": "Th\ufffd garbled body of text with replacement chars \ufffd repeated."},
    ]
    report = generate_quality_report(chunks)
    assert report["total"] == 4
    assert report["passed"] == 3
    assert report["failed"] == 1
    assert report["per_source"]["mpds_protocol_reference"] == {"total": 2, "passed": 2, "failed": 0}
    assert report["per_source"]["nemsis_v3_data_dictionary"]["failed"] == 1
    assert report["rejected_samples"][0]["source_id"] == "nemsis_v3_data_dictionary"
    assert "garbled_ocr" in report["rejected_samples"][0]["reasons"]
