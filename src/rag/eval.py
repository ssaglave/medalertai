"""
src/rag/eval.py

Phase 3 — RAG evaluation harness.
Owner: Deekshitha (C5)

Responsibilities:
  - Precision@5 test suite: measure how many of the top-5 retrieved chunks
    are actually relevant to a given query (based on ground-truth source_ids).
  - LLM-as-judge faithfulness scorer: use Claude to score whether the RAG
    answer is faithful to the retrieved context (scale 0–3).
  - Latency benchmark: measure query latency and enforce p50 < 3s, p95 < 8s
    targets.

Dependencies:
  - src.rag.chain  (C4 — Srileakhana): query(), get_retriever(), get_llm()
  - src.rag.vectorstore (C3 — Sanika): get_vectorstore()
  - config.settings: ANTHROPIC_API_KEY

Usage from the repo root:
    python -m src.rag.eval                         # full evaluation suite
    python -m src.rag.eval --benchmark-only        # latency benchmark only
    python -m src.rag.eval --precision-only        # precision@5 only
    python -m src.rag.eval --faithfulness-only     # LLM-as-judge only
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from config.settings import ANTHROPIC_API_KEY, PROJECT_ROOT

log = logging.getLogger("medalertai.rag.eval")

# ── Targets ──
PRECISION_AT_K_TARGET = 0.6       # Precision@5 > 0.6
FAITHFULNESS_AVG_TARGET = 1.5     # LLM-as-judge mean faithfulness > 1.5 (0–3 scale)
LATENCY_P50_TARGET_S = 3.0        # p50 latency < 3 seconds
LATENCY_P95_TARGET_S = 8.0        # p95 latency < 8 seconds

EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
DEFAULT_K = 5

# ── Ground-Truth Evaluation Queries ──
# Each entry maps a question to the set of source_ids that should appear
# in the top-k retrieval results for a correct answer.
EVAL_QUERIES: list[dict[str, Any]] = [
    {
        "question": "What is the MPDS dispatch protocol for chest pain?",
        "relevant_source_ids": {"mpds_protocol_reference", "mpds_dispatch_mapping"},
        "category": "mpds_protocols",
    },
    {
        "question": "What does MPDS code 17D1 mean?",
        "relevant_source_ids": {"mpds_protocol_reference", "mpds_dispatch_mapping"},
        "category": "mpds_protocols",
    },
    {
        "question": "Describe the NEMSIS v3 data elements for incident response times.",
        "relevant_source_ids": {"nemsis_v3_data_dictionary", "nemsis_v3_reference"},
        "category": "nemsis_standard",
    },
    {
        "question": "What fields does the WPRDC EMS dataset contain?",
        "relevant_source_ids": {"wprdc_data_dictionaries"},
        "category": "wprdc_glossary",
    },
    {
        "question": "What are the PA BLS protocols for cardiac arrest management?",
        "relevant_source_ids": {"pa_doh_2023_bls_protocols", "pa_doh_ems_regulations"},
        "category": "pa_doh_ems",
    },
    {
        "question": "Explain the NFPA 1221 standard for emergency communications.",
        "relevant_source_ids": {"nfpa_1221_2019_reference"},
        "category": "nfpa_communications",
    },
    {
        "question": "What is the difference between ALS and BLS dispatch protocols in Pennsylvania?",
        "relevant_source_ids": {
            "pa_doh_2023_bls_protocols",
            "pa_doh_2023_als_protocols",
            "pa_doh_ems_regulations",
        },
        "category": "pa_doh_ems",
    },
    {
        "question": "How are MPDS complaint codes mapped from Pittsburgh raw call types?",
        "relevant_source_ids": {"mpds_dispatch_mapping"},
        "category": "mpds_protocols",
    },
    {
        "question": "What is the NEMSIS v3 schema for patient assessment data?",
        "relevant_source_ids": {"nemsis_v3_data_dictionary", "nemsis_v3_reference"},
        "category": "nemsis_standard",
    },
    {
        "question": "What data sources does the WPRDC provide for EMS and fire incidents?",
        "relevant_source_ids": {"wprdc_data_dictionaries"},
        "category": "wprdc_glossary",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Data classes for evaluation results
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PrecisionResult:
    """Result for a single Precision@K evaluation query."""
    question: str
    category: str
    expected_source_ids: set[str]
    retrieved_source_ids: list[str]
    relevant_retrieved: list[str]
    precision_at_k: float
    k: int


@dataclass
class FaithfulnessResult:
    """Result for a single LLM-as-judge faithfulness evaluation."""
    question: str
    answer: str
    context_snippets: list[str]
    score: float          # 0–3 scale
    reasoning: str
    error: str = ""


@dataclass
class LatencyMeasurement:
    """Result for a single latency measurement."""
    question: str
    latency_s: float
    success: bool
    error: str = ""


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    # Precision@K
    precision_results: list[PrecisionResult] = field(default_factory=list)
    mean_precision_at_k: float = 0.0
    precision_target: float = PRECISION_AT_K_TARGET
    precision_passes: bool = False

    # Faithfulness
    faithfulness_results: list[FaithfulnessResult] = field(default_factory=list)
    mean_faithfulness: float = 0.0
    faithfulness_target: float = FAITHFULNESS_AVG_TARGET
    faithfulness_passes: bool = False

    # Latency
    latency_measurements: list[LatencyMeasurement] = field(default_factory=list)
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    latency_p50_target: float = LATENCY_P50_TARGET_S
    latency_p95_target: float = LATENCY_P95_TARGET_S
    latency_passes: bool = False

    # Overall
    all_pass: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "precision_at_k": {
                "mean": round(self.mean_precision_at_k, 4),
                "target": self.precision_target,
                "passes": self.precision_passes,
                "n_queries": len(self.precision_results),
            },
            "faithfulness": {
                "mean": round(self.mean_faithfulness, 4),
                "target": self.faithfulness_target,
                "passes": self.faithfulness_passes,
                "n_queries": len(self.faithfulness_results),
            },
            "latency": {
                "p50_s": round(self.latency_p50_s, 3),
                "p95_s": round(self.latency_p95_s, 3),
                "p50_target": self.latency_p50_target,
                "p95_target": self.latency_p95_target,
                "passes": self.latency_passes,
                "n_measurements": len(self.latency_measurements),
            },
            "all_pass": self.all_pass,
        }


# ═══════════════════════════════════════════════════════════════════════
# Precision@K Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_precision_at_k(
    retriever: Any | None = None,
    queries: list[dict[str, Any]] | None = None,
    k: int = DEFAULT_K,
) -> list[PrecisionResult]:
    """
    Evaluate Precision@K for each ground-truth query.

    For each query, retrieve the top-k chunks and check how many of them
    come from a relevant source_id according to our ground-truth set.

    Precision@K = |relevant ∩ retrieved| / K
    """
    if retriever is None:
        from src.rag.chain import get_retriever
        retriever = get_retriever(k=k)

    eval_queries = queries if queries is not None else EVAL_QUERIES
    results: list[PrecisionResult] = []

    for entry in eval_queries:
        question = entry["question"]
        expected = set(entry["relevant_source_ids"])
        category = entry.get("category", "")

        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older LangChain versions
            docs = retriever.get_relevant_documents(question)

        retrieved_ids = []
        for doc in docs[:k]:
            meta = getattr(doc, "metadata", {}) or {}
            sid = str(meta.get("source_id", "") or meta.get("source", ""))
            retrieved_ids.append(sid)

        relevant = [sid for sid in retrieved_ids if sid in expected]
        precision = len(relevant) / k if k > 0 else 0.0

        results.append(PrecisionResult(
            question=question,
            category=category,
            expected_source_ids=expected,
            retrieved_source_ids=retrieved_ids,
            relevant_retrieved=relevant,
            precision_at_k=precision,
            k=k,
        ))
        log.info(
            "Precision@%d for '%s': %.2f (%d/%d relevant)",
            k, question[:60], precision, len(relevant), k,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# LLM-as-Judge Faithfulness Scorer
# ═══════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Your task is to evaluate whether the ANSWER is faithful to the CONTEXT provided. 
Faithfulness means the answer only contains information that can be directly supported by the context. 
The answer should not hallucinate facts, protocol details, or data that are not present in the context.

Score on a 0–3 scale:
  0 = Completely unfaithful — the answer contradicts the context or is entirely fabricated.
  1 = Mostly unfaithful — the answer has some connection to the context but includes significant hallucinated content.
  2 = Mostly faithful — the answer is largely grounded in the context with only minor unsupported claims.
  3 = Fully faithful — every claim in the answer is directly supported by the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Respond in JSON format only:
{{"score": <0-3>, "reasoning": "<brief explanation>"}}
"""


def evaluate_faithfulness(
    queries: list[dict[str, Any]] | None = None,
    qa_chain: Any | None = None,
    llm: Any | None = None,
    k: int = DEFAULT_K,
) -> list[FaithfulnessResult]:
    """
    Use an LLM-as-judge to evaluate whether RAG answers are faithful to
    the retrieved context.

    For each query:
    1. Run the RAG chain to get an answer + source documents.
    2. Ask the LLM judge to score faithfulness on a 0–3 scale.
    """
    eval_queries = queries if queries is not None else EVAL_QUERIES
    results: list[FaithfulnessResult] = []

    # Build the RAG chain if not provided
    if qa_chain is None:
        from src.rag.chain import build_qa_chain
        qa_chain = build_qa_chain(k=k)

    # Build the judge LLM
    judge_llm = llm
    if judge_llm is None:
        judge_llm = _get_judge_llm()

    for entry in eval_queries:
        question = entry["question"]
        try:
            result = _run_faithfulness_single(question, qa_chain, judge_llm)
            results.append(result)
            log.info(
                "Faithfulness for '%s': %.1f — %s",
                question[:60], result.score, result.reasoning[:80],
            )
        except Exception as exc:
            log.warning("Faithfulness eval failed for '%s': %s", question[:60], exc)
            results.append(FaithfulnessResult(
                question=question,
                answer="",
                context_snippets=[],
                score=0.0,
                reasoning="",
                error=str(exc),
            ))

    return results


def _run_faithfulness_single(
    question: str,
    qa_chain: Any,
    judge_llm: Any,
) -> FaithfulnessResult:
    """Run a single faithfulness evaluation."""
    from src.rag.chain import query as rag_query

    # Get RAG answer
    rag_result = rag_query(question, qa_chain=qa_chain)
    answer = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])
    snippets = [s.get("snippet", "") for s in sources if s.get("snippet")]

    context_text = "\n\n---\n\n".join(snippets) if snippets else "(no context retrieved)"

    # Ask the judge
    prompt = FAITHFULNESS_PROMPT.format(
        context=context_text,
        question=question,
        answer=answer,
    )

    try:
        judge_response = judge_llm.invoke(prompt)
        # Handle different response types
        if hasattr(judge_response, "content"):
            response_text = judge_response.content
        else:
            response_text = str(judge_response)

        parsed = _parse_faithfulness_response(response_text)
        return FaithfulnessResult(
            question=question,
            answer=answer,
            context_snippets=snippets,
            score=parsed["score"],
            reasoning=parsed["reasoning"],
        )
    except Exception as exc:
        return FaithfulnessResult(
            question=question,
            answer=answer,
            context_snippets=snippets,
            score=0.0,
            reasoning="",
            error=f"Judge LLM failed: {exc}",
        )


def _parse_faithfulness_response(text: str) -> dict[str, Any]:
    """Parse the LLM judge's JSON response."""
    import re

    # Try to extract JSON from the response
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0))
            score = max(0.0, min(3.0, score))  # Clamp to [0, 3]
            return {
                "score": score,
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Fallback: try to find a numeric score in the text
    score_match = re.search(r'score["\s:]+(\d)', text)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "reasoning": text.strip(),
        }

    return {"score": 0.0, "reasoning": f"Could not parse judge response: {text[:200]}"}


def _get_judge_llm() -> Any:
    """Get the LLM used as a faithfulness judge."""
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for LLM-as-judge faithfulness evaluation. "
            "Set it in .env or as an environment variable."
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError(
            "langchain-anthropic is required for faithfulness evaluation."
        ) from exc

    model = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5")
    return ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=0.0,
        timeout=30,
        max_retries=2,
    )


# ═══════════════════════════════════════════════════════════════════════
# Latency Benchmark
# ═══════════════════════════════════════════════════════════════════════

def benchmark_latency(
    queries: list[dict[str, Any]] | None = None,
    qa_chain: Any | None = None,
    k: int = DEFAULT_K,
    n_repeats: int = 1,
) -> list[LatencyMeasurement]:
    """
    Measure end-to-end query latency for each evaluation query.

    Runs each query n_repeats times and records the latency.
    Targets: p50 < 3s, p95 < 8s.
    """
    eval_queries = queries if queries is not None else EVAL_QUERIES

    if qa_chain is None:
        from src.rag.chain import build_qa_chain
        qa_chain = build_qa_chain(k=k)

    measurements: list[LatencyMeasurement] = []

    for entry in eval_queries:
        question = entry["question"]
        for repeat in range(n_repeats):
            start = time.perf_counter()
            try:
                from src.rag.chain import query as rag_query
                rag_query(question, qa_chain=qa_chain)
                elapsed = time.perf_counter() - start
                measurements.append(LatencyMeasurement(
                    question=question,
                    latency_s=elapsed,
                    success=True,
                ))
                log.info(
                    "Latency for '%s' (repeat %d): %.3fs",
                    question[:60], repeat + 1, elapsed,
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start
                measurements.append(LatencyMeasurement(
                    question=question,
                    latency_s=elapsed,
                    success=False,
                    error=str(exc),
                ))
                log.warning(
                    "Latency measurement failed for '%s': %s (%.3fs)",
                    question[:60], exc, elapsed,
                )

    return measurements


def compute_latency_percentiles(
    measurements: list[LatencyMeasurement],
) -> dict[str, float]:
    """Compute p50 and p95 from successful latency measurements."""
    successful = [m.latency_s for m in measurements if m.success]
    if not successful:
        return {"p50": float("inf"), "p95": float("inf"), "mean": float("inf")}

    successful.sort()
    n = len(successful)
    p50_idx = int(n * 0.50)
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "p50": successful[p50_idx],
        "p95": successful[p95_idx],
        "mean": statistics.mean(successful),
        "min": successful[0],
        "max": successful[-1],
        "n": n,
    }


# ═══════════════════════════════════════════════════════════════════════
# Full Evaluation Suite
# ═══════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    retriever: Any | None = None,
    qa_chain: Any | None = None,
    judge_llm: Any | None = None,
    queries: list[dict[str, Any]] | None = None,
    k: int = DEFAULT_K,
    n_latency_repeats: int = 1,
    skip_precision: bool = False,
    skip_faithfulness: bool = False,
    skip_latency: bool = False,
) -> EvalReport:
    """
    Run the complete RAG evaluation suite and return an EvalReport.

    Components:
      1. Precision@K — retrieval relevance
      2. Faithfulness — LLM-as-judge grounding check
      3. Latency — end-to-end timing benchmark
    """
    report = EvalReport()

    # ── 1. Precision@K ──
    if not skip_precision:
        log.info("═══ Running Precision@%d evaluation ═══", k)
        precision_results = evaluate_precision_at_k(
            retriever=retriever, queries=queries, k=k,
        )
        report.precision_results = precision_results
        if precision_results:
            report.mean_precision_at_k = statistics.mean(
                r.precision_at_k for r in precision_results
            )
        report.precision_passes = report.mean_precision_at_k >= PRECISION_AT_K_TARGET
        log.info(
            "Precision@%d: mean=%.4f, target=%.2f, passes=%s",
            k, report.mean_precision_at_k, PRECISION_AT_K_TARGET, report.precision_passes,
        )

    # ── 2. Faithfulness ──
    if not skip_faithfulness:
        log.info("═══ Running Faithfulness evaluation ═══")
        faith_results = evaluate_faithfulness(
            queries=queries, qa_chain=qa_chain, llm=judge_llm, k=k,
        )
        report.faithfulness_results = faith_results
        scored = [r for r in faith_results if not r.error]
        if scored:
            report.mean_faithfulness = statistics.mean(r.score for r in scored)
        report.faithfulness_passes = report.mean_faithfulness >= FAITHFULNESS_AVG_TARGET
        log.info(
            "Faithfulness: mean=%.4f, target=%.2f, passes=%s",
            report.mean_faithfulness, FAITHFULNESS_AVG_TARGET, report.faithfulness_passes,
        )

    # ── 3. Latency ──
    if not skip_latency:
        log.info("═══ Running Latency benchmark ═══")
        latency_measurements = benchmark_latency(
            queries=queries, qa_chain=qa_chain, k=k, n_repeats=n_latency_repeats,
        )
        report.latency_measurements = latency_measurements
        percentiles = compute_latency_percentiles(latency_measurements)
        report.latency_p50_s = percentiles["p50"]
        report.latency_p95_s = percentiles["p95"]
        report.latency_passes = (
            report.latency_p50_s <= LATENCY_P50_TARGET_S
            and report.latency_p95_s <= LATENCY_P95_TARGET_S
        )
        log.info(
            "Latency: p50=%.3fs (target <%.1fs), p95=%.3fs (target <%.1fs), passes=%s",
            report.latency_p50_s, LATENCY_P50_TARGET_S,
            report.latency_p95_s, LATENCY_P95_TARGET_S,
            report.latency_passes,
        )

    # ── Overall ──
    components_run = []
    if not skip_precision:
        components_run.append(report.precision_passes)
    if not skip_faithfulness:
        components_run.append(report.faithfulness_passes)
    if not skip_latency:
        components_run.append(report.latency_passes)

    report.all_pass = all(components_run) if components_run else False

    return report


def save_report(report: EvalReport, output_dir: Path = EVAL_RESULTS_DIR) -> Path:
    """Save the evaluation report to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"rag_eval_{timestamp}.json"

    # Convert to JSON-serializable dict
    data = report.summary()
    data["precision_details"] = []
    for pr in report.precision_results:
        data["precision_details"].append({
            "question": pr.question,
            "category": pr.category,
            "expected_source_ids": sorted(pr.expected_source_ids),
            "retrieved_source_ids": pr.retrieved_source_ids,
            "relevant_retrieved": pr.relevant_retrieved,
            "precision_at_k": round(pr.precision_at_k, 4),
            "k": pr.k,
        })

    data["faithfulness_details"] = []
    for fr in report.faithfulness_results:
        data["faithfulness_details"].append({
            "question": fr.question,
            "score": fr.score,
            "reasoning": fr.reasoning,
            "error": fr.error,
        })

    data["latency_details"] = []
    for lm in report.latency_measurements:
        data["latency_details"].append({
            "question": lm.question,
            "latency_s": round(lm.latency_s, 4),
            "success": lm.success,
            "error": lm.error,
        })

    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    log.info("Saved evaluation report to %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MedAlertAI Phase 3 RAG evaluation suite (C5 — Deekshitha)",
    )
    parser.add_argument("--precision-only", action="store_true", help="Run Precision@5 only.")
    parser.add_argument("--faithfulness-only", action="store_true", help="Run faithfulness only.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run latency benchmark only.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K for retrieval (default: 5).")
    parser.add_argument("--repeats", type=int, default=1, help="Latency measurement repeats per query.")
    parser.add_argument("--output-dir", type=Path, default=EVAL_RESULTS_DIR, help="Output directory.")
    parser.add_argument("--no-save", action="store_true", help="Do not save report to disk.")
    args = parser.parse_args(argv)

    skip_precision = args.faithfulness_only or args.benchmark_only
    skip_faithfulness = args.precision_only or args.benchmark_only
    skip_latency = args.precision_only or args.faithfulness_only

    report = run_full_evaluation(
        k=args.k,
        n_latency_repeats=args.repeats,
        skip_precision=skip_precision,
        skip_faithfulness=skip_faithfulness,
        skip_latency=skip_latency,
    )

    # Print summary
    summary = report.summary()
    print("\n" + "=" * 70)
    print("MedAlertAI RAG Evaluation Report")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)
    print(f"OVERALL: {'✅ ALL PASS' if report.all_pass else '❌ SOME TARGETS NOT MET'}")
    print("=" * 70 + "\n")

    if not args.no_save:
        save_report(report, args.output_dir)

    return 0 if report.all_pass else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
