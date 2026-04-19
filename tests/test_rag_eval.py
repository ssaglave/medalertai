"""
tests/test_rag_eval.py

Phase 3 — RAG evaluation test suite.
Owner: Deekshitha (C5)

Tests for:
  - Precision@K evaluation with mocked retriever
  - LLM-as-judge faithfulness scoring with mocked LLM
  - Latency benchmarking with mocked RAG chain
  - Full evaluation suite integration
  - Report serialization
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.rag.eval import (
    EVAL_QUERIES,
    FAITHFULNESS_AVG_TARGET,
    LATENCY_P50_TARGET_S,
    LATENCY_P95_TARGET_S,
    PRECISION_AT_K_TARGET,
    EvalReport,
    FaithfulnessResult,
    LatencyMeasurement,
    PrecisionResult,
    benchmark_latency,
    compute_latency_percentiles,
    evaluate_faithfulness,
    evaluate_precision_at_k,
    run_full_evaluation,
    save_report,
    _parse_faithfulness_response,
)


# ═══════════════════════════════════════════════════════════════════════
# Mock / Fake Components
# ═══════════════════════════════════════════════════════════════════════

def _make_doc(source_id: str, title: str = "", chunk_index: int = 1, text: str = "mock text"):
    """Helper: create a fake LangChain Document."""
    return SimpleNamespace(
        page_content=text,
        metadata={
            "source_id": source_id,
            "title": title or source_id,
            "chunk_index": chunk_index,
            "chunk_id": f"{source_id}_c{chunk_index}",
            "url": "",
            "file_name": f"{source_id}.md",
        },
    )


class FakeRetriever:
    """Fake retriever that returns predetermined documents per query."""

    def __init__(self, docs_by_query: dict[str, list] | None = None, default_docs: list | None = None):
        self._docs_by_query = docs_by_query or {}
        self._default_docs = default_docs or []

    def invoke(self, question: str) -> list:
        return self._docs_by_query.get(question, self._default_docs)

    def get_relevant_documents(self, question: str) -> list:
        return self.invoke(question)


class FakeQAChain:
    """Fake QA chain that returns a canned answer + source documents."""

    def __init__(self, answer: str = "Test answer from protocol.", docs: list | None = None, delay_s: float = 0.0):
        self._answer = answer
        self._docs = docs or [
            _make_doc("mpds_protocol_reference", "MPDS Protocol Reference", 1, "Protocol details for test."),
        ]
        self._delay_s = delay_s

    def invoke(self, payload: dict) -> dict:
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return {
            "result": self._answer,
            "source_documents": self._docs,
        }


class FakeJudgeLLM:
    """Fake LLM that returns a preset faithfulness score."""

    def __init__(self, score: float = 2.5, reasoning: str = "The answer is mostly faithful."):
        self._score = score
        self._reasoning = reasoning

    def invoke(self, prompt: str) -> SimpleNamespace:
        response_json = json.dumps({
            "score": self._score,
            "reasoning": self._reasoning,
        })
        return SimpleNamespace(content=response_json)


# ═══════════════════════════════════════════════════════════════════════
# Precision@K Tests
# ═══════════════════════════════════════════════════════════════════════

class TestPrecisionAtK:
    """Tests for the Precision@K evaluation component."""

    def test_perfect_precision_when_all_relevant(self):
        """All retrieved docs come from relevant sources → precision = 1.0."""
        docs = [
            _make_doc("mpds_protocol_reference", chunk_index=i)
            for i in range(5)
        ]
        retriever = FakeRetriever(default_docs=docs)
        queries = [{
            "question": "Test question",
            "relevant_source_ids": {"mpds_protocol_reference"},
            "category": "test",
        }]

        results = evaluate_precision_at_k(retriever=retriever, queries=queries, k=5)

        assert len(results) == 1
        assert results[0].precision_at_k == 1.0
        assert len(results[0].relevant_retrieved) == 5

    def test_zero_precision_when_none_relevant(self):
        """No retrieved docs from relevant sources → precision = 0.0."""
        docs = [_make_doc("irrelevant_source", chunk_index=i) for i in range(5)]
        retriever = FakeRetriever(default_docs=docs)
        queries = [{
            "question": "Test question",
            "relevant_source_ids": {"mpds_protocol_reference"},
            "category": "test",
        }]

        results = evaluate_precision_at_k(retriever=retriever, queries=queries, k=5)

        assert results[0].precision_at_k == 0.0
        assert results[0].relevant_retrieved == []

    def test_partial_precision(self):
        """Some retrieved docs relevant → partial precision."""
        docs = [
            _make_doc("mpds_protocol_reference", chunk_index=1),
            _make_doc("mpds_protocol_reference", chunk_index=2),
            _make_doc("irrelevant_source", chunk_index=1),
            _make_doc("mpds_dispatch_mapping", chunk_index=1),
            _make_doc("irrelevant_source", chunk_index=2),
        ]
        retriever = FakeRetriever(default_docs=docs)
        queries = [{
            "question": "Test question",
            "relevant_source_ids": {"mpds_protocol_reference", "mpds_dispatch_mapping"},
            "category": "test",
        }]

        results = evaluate_precision_at_k(retriever=retriever, queries=queries, k=5)

        assert results[0].precision_at_k == 3.0 / 5.0  # 3 relevant out of 5

    def test_multiple_queries(self):
        """Multiple eval queries produce separate results."""
        docs = [_make_doc("mpds_protocol_reference", chunk_index=i) for i in range(5)]
        retriever = FakeRetriever(default_docs=docs)
        queries = [
            {"question": "Q1", "relevant_source_ids": {"mpds_protocol_reference"}, "category": "a"},
            {"question": "Q2", "relevant_source_ids": {"other_source"}, "category": "b"},
        ]

        results = evaluate_precision_at_k(retriever=retriever, queries=queries, k=5)

        assert len(results) == 2
        assert results[0].precision_at_k == 1.0
        assert results[1].precision_at_k == 0.0

    def test_precision_result_stores_metadata(self):
        """PrecisionResult captures expected_source_ids and retrieved_source_ids."""
        docs = [_make_doc("src_a"), _make_doc("src_b")]
        retriever = FakeRetriever(default_docs=docs)
        queries = [{
            "question": "Test",
            "relevant_source_ids": {"src_a"},
            "category": "test",
        }]

        results = evaluate_precision_at_k(retriever=retriever, queries=queries, k=2)

        assert results[0].expected_source_ids == {"src_a"}
        assert "src_a" in results[0].retrieved_source_ids
        assert "src_b" in results[0].retrieved_source_ids
        assert results[0].k == 2

    def test_empty_queries_returns_empty(self):
        """No queries produces no results."""
        retriever = FakeRetriever()
        results = evaluate_precision_at_k(retriever=retriever, queries=[], k=5)
        assert results == []


# ═══════════════════════════════════════════════════════════════════════
# Faithfulness Scorer Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFaithfulness:
    """Tests for the LLM-as-judge faithfulness scoring."""

    def test_high_faithfulness_score(self):
        """Mock judge returns high score → result captures it."""
        qa_chain = FakeQAChain(answer="Faithful answer grounded in protocol.")
        judge = FakeJudgeLLM(score=3.0, reasoning="Fully grounded.")
        queries = [{
            "question": "Test question",
            "relevant_source_ids": set(),
            "category": "test",
        }]

        results = evaluate_faithfulness(queries=queries, qa_chain=qa_chain, llm=judge)

        assert len(results) == 1
        assert results[0].score == 3.0
        assert results[0].reasoning == "Fully grounded."
        assert results[0].error == ""

    def test_low_faithfulness_score(self):
        """Mock judge returns low score → result reflects it."""
        qa_chain = FakeQAChain(answer="Hallucinated answer.")
        judge = FakeJudgeLLM(score=0.0, reasoning="Completely fabricated.")
        queries = [{
            "question": "Test question",
            "relevant_source_ids": set(),
            "category": "test",
        }]

        results = evaluate_faithfulness(queries=queries, qa_chain=qa_chain, llm=judge)

        assert results[0].score == 0.0

    def test_multiple_faithfulness_queries(self):
        """Multiple queries each get scored independently."""
        qa_chain = FakeQAChain()
        judge = FakeJudgeLLM(score=2.0)
        queries = [
            {"question": "Q1", "relevant_source_ids": set(), "category": "a"},
            {"question": "Q2", "relevant_source_ids": set(), "category": "b"},
        ]

        results = evaluate_faithfulness(queries=queries, qa_chain=qa_chain, llm=judge)

        assert len(results) == 2
        assert all(r.score == 2.0 for r in results)

    def test_faithfulness_captures_context_snippets(self):
        """Faithfulness result should include context snippets from sources."""
        docs = [
            _make_doc("src_a", text="Protocol text about cardiac arrest."),
            _make_doc("src_b", text="MPDS code 17D1 dispatch info."),
        ]
        qa_chain = FakeQAChain(
            answer="Based on the protocol...",
            docs=docs,
        )
        judge = FakeJudgeLLM(score=2.5)
        queries = [{"question": "Q1", "relevant_source_ids": set(), "category": "test"}]

        results = evaluate_faithfulness(queries=queries, qa_chain=qa_chain, llm=judge)

        assert len(results[0].context_snippets) > 0


class TestParseFaithfulnessResponse:
    """Tests for the faithfulness response parser."""

    def test_valid_json_response(self):
        text = '{"score": 2, "reasoning": "Mostly faithful to context."}'
        result = _parse_faithfulness_response(text)
        assert result["score"] == 2.0
        assert result["reasoning"] == "Mostly faithful to context."

    def test_json_embedded_in_text(self):
        text = 'Here is my evaluation: {"score": 3, "reasoning": "Fully grounded."} End.'
        result = _parse_faithfulness_response(text)
        assert result["score"] == 3.0

    def test_score_clamped_to_max(self):
        text = '{"score": 5, "reasoning": "Over the scale."}'
        result = _parse_faithfulness_response(text)
        assert result["score"] == 3.0  # Clamped

    def test_score_clamped_to_min(self):
        text = '{"score": -1, "reasoning": "Below the scale."}'
        result = _parse_faithfulness_response(text)
        assert result["score"] == 0.0  # Clamped

    def test_fallback_parsing(self):
        text = "The score: 2. The answer is mostly faithful."
        result = _parse_faithfulness_response(text)
        assert result["score"] == 2.0

    def test_unparseable_returns_zero(self):
        text = "No valid score here at all."
        result = _parse_faithfulness_response(text)
        assert result["score"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Latency Benchmark Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLatencyBenchmark:
    """Tests for the latency benchmarking component."""

    def test_fast_chain_under_target(self):
        """A fast QA chain should produce low latency measurements."""
        qa_chain = FakeQAChain(delay_s=0.01)
        queries = [{"question": "Fast query", "relevant_source_ids": set(), "category": "test"}]

        measurements = benchmark_latency(queries=queries, qa_chain=qa_chain, n_repeats=3)

        assert len(measurements) == 3
        assert all(m.success for m in measurements)
        assert all(m.latency_s < 1.0 for m in measurements)

    def test_latency_repeats(self):
        """n_repeats multiplies the number of measurements."""
        qa_chain = FakeQAChain()
        queries = [
            {"question": "Q1", "relevant_source_ids": set(), "category": "a"},
            {"question": "Q2", "relevant_source_ids": set(), "category": "b"},
        ]

        measurements = benchmark_latency(queries=queries, qa_chain=qa_chain, n_repeats=2)

        assert len(measurements) == 4  # 2 queries × 2 repeats

    def test_latency_measurement_records_question(self):
        """Each measurement should record which question was asked."""
        qa_chain = FakeQAChain()
        queries = [{"question": "Specific test query", "relevant_source_ids": set(), "category": "test"}]

        measurements = benchmark_latency(queries=queries, qa_chain=qa_chain)

        assert measurements[0].question == "Specific test query"


class TestLatencyPercentiles:
    """Tests for latency percentile computation."""

    def test_simple_percentiles(self):
        """Basic p50/p95 from a small set of measurements."""
        measurements = [
            LatencyMeasurement(question="q", latency_s=1.0, success=True),
            LatencyMeasurement(question="q", latency_s=2.0, success=True),
            LatencyMeasurement(question="q", latency_s=3.0, success=True),
            LatencyMeasurement(question="q", latency_s=4.0, success=True),
            LatencyMeasurement(question="q", latency_s=5.0, success=True),
        ]
        stats = compute_latency_percentiles(measurements)
        assert stats["p50"] == 3.0
        assert stats["p95"] == 5.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_empty_measurements_returns_inf(self):
        """No measurements → infinity."""
        stats = compute_latency_percentiles([])
        assert stats["p50"] == float("inf")
        assert stats["p95"] == float("inf")

    def test_failed_measurements_excluded(self):
        """Only successful measurements count toward percentiles."""
        measurements = [
            LatencyMeasurement(question="q", latency_s=1.0, success=True),
            LatencyMeasurement(question="q", latency_s=100.0, success=False, error="timeout"),
            LatencyMeasurement(question="q", latency_s=2.0, success=True),
        ]
        stats = compute_latency_percentiles(measurements)
        assert stats["n"] == 2
        assert stats["max"] == 2.0  # The 100s failure is excluded

    def test_single_measurement(self):
        """Single measurement → p50 == p95 == that value."""
        measurements = [
            LatencyMeasurement(question="q", latency_s=1.5, success=True),
        ]
        stats = compute_latency_percentiles(measurements)
        assert stats["p50"] == 1.5
        assert stats["p95"] == 1.5


# ═══════════════════════════════════════════════════════════════════════
# Full Evaluation Suite Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFullEvaluation:
    """Tests for the full evaluation suite integration."""

    def test_full_eval_with_all_passing(self):
        """All components pass with ideal mock data."""
        # Perfect retriever
        docs = [_make_doc("mpds_protocol_reference", chunk_index=i) for i in range(5)]
        retriever = FakeRetriever(default_docs=docs)

        # High faithfulness judge
        judge = FakeJudgeLLM(score=3.0, reasoning="Fully faithful.")

        # Fast chain
        qa_chain = FakeQAChain(delay_s=0.01)

        queries = [{
            "question": "Test",
            "relevant_source_ids": {"mpds_protocol_reference"},
            "category": "test",
        }]

        report = run_full_evaluation(
            retriever=retriever,
            qa_chain=qa_chain,
            judge_llm=judge,
            queries=queries,
            k=5,
        )

        assert report.precision_passes is True
        assert report.faithfulness_passes is True
        assert report.latency_passes is True
        assert report.all_pass is True

    def test_full_eval_with_precision_failing(self):
        """Precision fails when no relevant docs retrieved."""
        docs = [_make_doc("irrelevant", chunk_index=i) for i in range(5)]
        retriever = FakeRetriever(default_docs=docs)
        judge = FakeJudgeLLM(score=3.0)
        qa_chain = FakeQAChain(delay_s=0.01)

        queries = [{
            "question": "Test",
            "relevant_source_ids": {"mpds_protocol_reference"},
            "category": "test",
        }]

        report = run_full_evaluation(
            retriever=retriever, qa_chain=qa_chain, judge_llm=judge,
            queries=queries, k=5,
        )

        assert report.precision_passes is False
        assert report.all_pass is False

    def test_skip_components(self):
        """Skipping components leaves their defaults untouched."""
        retriever = FakeRetriever()
        report = run_full_evaluation(
            retriever=retriever,
            queries=[],
            skip_precision=True,
            skip_faithfulness=True,
            skip_latency=True,
        )

        assert report.precision_results == []
        assert report.faithfulness_results == []
        assert report.latency_measurements == []
        assert report.all_pass is False  # No components ran

    def test_skip_faithfulness_and_latency(self):
        """Only precision runs when others are skipped."""
        docs = [_make_doc("mpds_protocol_reference", chunk_index=i) for i in range(5)]
        retriever = FakeRetriever(default_docs=docs)
        queries = [{
            "question": "Test",
            "relevant_source_ids": {"mpds_protocol_reference"},
            "category": "test",
        }]

        report = run_full_evaluation(
            retriever=retriever,
            queries=queries,
            k=5,
            skip_faithfulness=True,
            skip_latency=True,
        )

        assert report.precision_passes is True
        assert report.faithfulness_results == []
        assert report.latency_measurements == []
        assert report.all_pass is True  # Only precision ran and passed


# ═══════════════════════════════════════════════════════════════════════
# Report Serialization Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReportSerialization:
    """Tests for report generation and saving."""

    def test_summary_structure(self):
        """EvalReport.summary() returns expected keys."""
        report = EvalReport()
        summary = report.summary()

        assert "precision_at_k" in summary
        assert "faithfulness" in summary
        assert "latency" in summary
        assert "all_pass" in summary
        assert "mean" in summary["precision_at_k"]
        assert "target" in summary["precision_at_k"]
        assert "passes" in summary["precision_at_k"]

    def test_save_report_writes_json(self, tmp_path):
        """save_report() creates a valid JSON file."""
        report = EvalReport(
            precision_results=[
                PrecisionResult(
                    question="Test Q",
                    category="test",
                    expected_source_ids={"src_a"},
                    retrieved_source_ids=["src_a", "src_b"],
                    relevant_retrieved=["src_a"],
                    precision_at_k=0.5,
                    k=2,
                ),
            ],
            mean_precision_at_k=0.5,
            precision_passes=False,
            faithfulness_results=[
                FaithfulnessResult(
                    question="Test Q",
                    answer="Answer.",
                    context_snippets=["snippet"],
                    score=2.5,
                    reasoning="Mostly faithful.",
                ),
            ],
            mean_faithfulness=2.5,
            faithfulness_passes=True,
            latency_measurements=[
                LatencyMeasurement(question="Test Q", latency_s=1.5, success=True),
            ],
            latency_p50_s=1.5,
            latency_p95_s=1.5,
            latency_passes=True,
        )

        output_path = save_report(report, output_dir=tmp_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["precision_at_k"]["mean"] == 0.5
        assert data["faithfulness"]["mean"] == 2.5
        assert data["latency"]["p50_s"] == 1.5
        assert len(data["precision_details"]) == 1
        assert len(data["faithfulness_details"]) == 1
        assert len(data["latency_details"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# Ground-Truth Query Validation
# ═══════════════════════════════════════════════════════════════════════

class TestEvalQueries:
    """Validate the ground-truth eval query definitions."""

    def test_eval_queries_not_empty(self):
        assert len(EVAL_QUERIES) >= 5

    def test_each_query_has_required_fields(self):
        for entry in EVAL_QUERIES:
            assert "question" in entry
            assert "relevant_source_ids" in entry
            assert isinstance(entry["relevant_source_ids"], set)
            assert len(entry["relevant_source_ids"]) > 0

    def test_eval_query_source_ids_are_strings(self):
        for entry in EVAL_QUERIES:
            for sid in entry["relevant_source_ids"]:
                assert isinstance(sid, str)


# ═══════════════════════════════════════════════════════════════════════
# Targets Validation
# ═══════════════════════════════════════════════════════════════════════

class TestTargets:
    """Validate that evaluation targets match the implementation plan."""

    def test_precision_target(self):
        assert PRECISION_AT_K_TARGET == 0.6

    def test_faithfulness_target(self):
        assert FAITHFULNESS_AVG_TARGET == 1.5

    def test_latency_p50_target(self):
        assert LATENCY_P50_TARGET_S == 3.0

    def test_latency_p95_target(self):
        assert LATENCY_P95_TARGET_S == 8.0
