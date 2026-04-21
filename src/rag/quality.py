"""
src/rag/quality.py

Phase 3 - Chunk quality validation for the RAG ingestion pipeline.
Owner: Suvarna (C2)

Extends the basic checks in `src/rag/ingest.py::_quality_score()` with
OCR-aware garble detection and a richer chunk validator. Used by the
ingestion pipeline as the gatekeeper before chunks are written to
`chunks.jsonl` and embedded into the vector store.
"""

from __future__ import annotations

import re
from typing import Iterable

MIN_CHUNK_CHARS = 120
MIN_ALPHA_RATIO = 0.50
MIN_READABLE_RATIO = 0.95
MIN_ALPHABETIC_WORDS = 8
MAX_NON_ASCII_RATIO = 0.10
MAX_VOWEL_LESS_TOKEN_RATIO = 0.30

GARBLE_MARKERS = ("\ufffd", "ÿþ", "þÿ")
SPECIAL_RUN_RE = re.compile(r"[^\x00-\x7f]{4,}")
WORD_RE = re.compile(r"[A-Za-z]{2,}")
VOWEL_RE = re.compile(r"[AEIOUaeiou]")


def detect_garbled_ocr(text: str) -> dict:
    if not text:
        return {
            "is_garbled": True,
            "non_ascii_ratio": 0.0,
            "has_garble_marker": False,
            "vowel_less_token_ratio": 0.0,
            "has_long_special_run": False,
        }

    non_ascii = sum(1 for char in text if ord(char) > 127)
    non_ascii_ratio = non_ascii / max(len(text), 1)

    has_marker = any(marker in text for marker in GARBLE_MARKERS)
    has_long_special_run = bool(SPECIAL_RUN_RE.search(text))

    tokens = WORD_RE.findall(text)
    if tokens:
        vowel_less = sum(1 for token in tokens if not VOWEL_RE.search(token))
        vowel_less_ratio = vowel_less / len(tokens)
    else:
        vowel_less_ratio = 1.0

    is_garbled = (
        has_marker
        or non_ascii_ratio > MAX_NON_ASCII_RATIO
        or vowel_less_ratio > MAX_VOWEL_LESS_TOKEN_RATIO
        or has_long_special_run
    )

    return {
        "is_garbled": is_garbled,
        "non_ascii_ratio": round(non_ascii_ratio, 3),
        "has_garble_marker": has_marker,
        "vowel_less_token_ratio": round(vowel_less_ratio, 3),
        "has_long_special_run": has_long_special_run,
    }


def validate_chunk(text: str) -> dict:
    if not text:
        return {
            "passes": False,
            "reasons": ["empty"],
            "chars": 0,
            "alpha_ratio": 0.0,
            "readable_ratio": 0.0,
            "word_count": 0,
            "garble": detect_garbled_ocr(""),
        }

    chars = len(text)
    alpha = sum(char.isalpha() for char in text)
    readable = sum(char.isprintable() or char.isspace() for char in text)
    alpha_ratio = alpha / max(chars, 1)
    readable_ratio = readable / max(chars, 1)
    word_count = len(WORD_RE.findall(text))
    garble = detect_garbled_ocr(text)

    reasons: list[str] = []
    if chars < MIN_CHUNK_CHARS:
        reasons.append("too_short")
    if alpha_ratio < MIN_ALPHA_RATIO:
        reasons.append("low_alpha_ratio")
    if readable_ratio < MIN_READABLE_RATIO:
        reasons.append("low_readable_ratio")
    if word_count < MIN_ALPHABETIC_WORDS:
        reasons.append("too_few_words")
    if garble["is_garbled"]:
        reasons.append("garbled_ocr")

    return {
        "passes": not reasons,
        "reasons": reasons,
        "chars": chars,
        "alpha_ratio": round(alpha_ratio, 3),
        "readable_ratio": round(readable_ratio, 3),
        "word_count": word_count,
        "garble": garble,
    }


def generate_quality_report(chunks: Iterable[dict]) -> dict:
    per_source: dict[str, dict] = {}
    rejected_samples: list[dict] = []
    total = 0
    passed = 0

    for chunk in chunks:
        total += 1
        source_id = chunk.get("source_id", "unknown")
        text = chunk.get("text", "")
        result = validate_chunk(text)
        bucket = per_source.setdefault(source_id, {"total": 0, "passed": 0, "failed": 0})
        bucket["total"] += 1
        if result["passes"]:
            passed += 1
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
            if len(rejected_samples) < 10:
                rejected_samples.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "source_id": source_id,
                        "reasons": result["reasons"],
                        "preview": text[:160],
                    }
                )

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "per_source": per_source,
        "rejected_samples": rejected_samples,
    }
