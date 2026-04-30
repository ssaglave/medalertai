"""
src/rag/chain.py

Phase 3 - LangChain RetrievalQA chain with Hugging Face inference.
Owner: Srileakhana (C4)

Responsibilities covered here:
  - Build a protocol-only RetrievalQA chain over the ChromaDB retriever.
  - Use a Hugging Face inference endpoint by default when a token is configured.
  - Return citation-ready source metadata for dashboard callbacks and tests.

Usage from the repo root after running ingestion/vectorstore setup:
    python -c "from src.rag.chain import query; print(query('MPDS 17D1'))"
"""

from __future__ import annotations

import logging
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

# Strips any leftover "[source: ...; chunk: N]" markers the LLM may emit
# despite the prompt instruction not to.
_CITATION_MARKER_RE = re.compile(r"\s*\[source:[^\]]*\]\s*", re.IGNORECASE)

from config.settings import HUGGINGFACE_API_TOKEN, HUGGINGFACE_ENDPOINT_URL, HUGGINGFACE_MODEL, PROCESSED_DATA_DIR


log = logging.getLogger("medalertai.rag.chain")

DEFAULT_MODEL = os.getenv("HUGGINGFACE_MODEL", HUGGINGFACE_MODEL)
DEFAULT_HF_ENDPOINT_URL = os.getenv("HUGGINGFACE_ENDPOINT_URL", HUGGINGFACE_ENDPOINT_URL)
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CHUNKS_PATH = PROCESSED_DATA_DIR / "rag" / "chunks.jsonl"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_KEYWORD_STOPWORDS = {
    "about",
    "against",
    "answer",
    "could",
    "does",
    "from",
    "have",
    "into",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}
FALLBACK_ANSWER = (
    "I could not find an answer to that in the indexed knowledge base. "
    "This assistant only uses the documents currently ingested into the RAG "
    "corpus, which covers:\n\n"
    "- Medical Priority Dispatch System (MPDS) protocol concepts and structure\n"
    "- Pittsburgh EMS/Fire dispatch call types and their mapping to MPDS codes\n"
    "- NEMSIS v3 data dictionary and reference fields\n"
    "- Pennsylvania DOH EMS protocols (BLS, ALS, IALS) published on pa.gov\n"
    "- Pennsylvania Office of the State Fire Commissioner standards, training, "
    "and regulations published on pa.gov\n\n"
    "Try rephrasing your question around those topics. For anything outside "
    "this scope (individual MPDS dispatch codes not yet ingested, "
    "non-Pennsylvania state protocols, etc.), consult the official protocol "
    "document or a qualified clinical/dispatch supervisor."
)

SYSTEM_PROMPT = """You are MedAlertAI's EMS and fire dispatch protocol assistant.

Use only the retrieved protocol, NEMSIS, MPDS mapping, NFPA, and WPRDC context below.
Do not use outside medical knowledge, do not invent protocol details, and do not
provide medical direction beyond what the retrieved context supports.
Keep the answer short: one to three sentences when possible.

If the context does not contain the answer, say exactly:
"{fallback_answer}"

Do not include chunk markers or text like "[source: ...; chunk: N]" in your answer;
the dashboard appends retrieved source titles separately.

Retrieved context:
{context}

Question: {question}

Answer:"""


class RagChainError(RuntimeError):
    """Raised when the RAG chain cannot be created or queried."""


try:
    from langchain_core.language_models.llms import LLM
except ImportError as exc:
    raise RuntimeError("LangChain core is required for the RAG chain.") from exc


class HuggingFaceInferenceLLM(LLM):
    """Small adapter that exposes Hugging Face text generation to LangChain RetrievalQA."""

    def __init__(
        self,
        model: str,
        api_token: str,
        endpoint_url: str = "",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RagChainError(
                "huggingface-hub is required for Hugging Face inference."
            ) from exc

        client_kwargs = {"token": api_token, "timeout": 60}
        if endpoint_url:
            client_kwargs["base_url"] = endpoint_url
        else:
            client_kwargs["model"] = model

        super().__init__()
        self._client = InferenceClient(**client_kwargs)
        self._model = model
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        stop = kwargs.get("stop")
        return self._call(prompt, stop=stop)

    def _call(self, prompt: str, stop: list[str] | None = None, **_: Any) -> str:
        response = self._client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            stop=stop,
        )
        try:
            return response.choices[0].message.content.strip()
        except (AttributeError, IndexError, KeyError, TypeError):
            return str(response).strip()


@dataclass(frozen=True)
class SourceCitation:
    """Citation metadata returned with every RAG answer."""

    source_id: str
    title: str
    chunk_index: int | None = None
    chunk_id: str = ""
    url: str = ""
    file_name: str = ""
    snippet: str = ""

    def label(self) -> str:
        chunk = f"; chunk: {self.chunk_index}" if self.chunk_index is not None else ""
        source = self.title or self.source_id or "unknown source"
        return f"[source: {source}{chunk}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "url": self.url,
            "file_name": self.file_name,
            "snippet": self.snippet,
            "citation": self.label(),
        }


def build_prompt():
    """Create the LangChain prompt used by the RetrievalQA chain."""
    try:
        from langchain_core.prompts import PromptTemplate
    except ImportError as exc:
        raise RagChainError("LangChain is required to build the RAG prompt.") from exc

    return PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
        partial_variables={"fallback_answer": FALLBACK_ANSWER},
    )


def get_llm(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    """Return the Hugging Face inference model used by the RAG chain."""
    api_token = HUGGINGFACE_API_TOKEN or os.getenv("HUGGINGFACE_API_TOKEN", "")
    if not api_token:
        raise RagChainError("HUGGINGFACE_API_TOKEN is required to query the Hugging Face endpoint.")
    return HuggingFaceInferenceLLM(
        model=model,
        api_token=api_token,
        endpoint_url=DEFAULT_HF_ENDPOINT_URL,
        temperature=temperature,
        max_new_tokens=512,
    )


def get_retriever(k: int = DEFAULT_TOP_K, vectorstore: Any | None = None):
    """Return a Chroma-backed retriever for top-k protocol chunks."""
    if k < 1:
        raise ValueError("k must be at least 1")

    if vectorstore is None:
        from src.rag.vectorstore import get_vectorstore

        store = get_vectorstore()
    else:
        store = vectorstore
    try:
        return store.as_retriever(search_kwargs={"k": k})
    except AttributeError as exc:
        raise RagChainError("Vector store does not expose as_retriever().") from exc


def build_qa_chain(
    llm: Any | None = None,
    retriever: Any | None = None,
    k: int = DEFAULT_TOP_K,
    return_source_documents: bool = True,
):
    """Build a LangChain RetrievalQA chain over the configured retriever."""
    try:
        from langchain.chains import RetrievalQA
    except ImportError:
        try:
            from langchain_classic.chains import RetrievalQA
        except ImportError as exc:
            raise RagChainError("LangChain is required to build RetrievalQA.") from exc

    chain_llm = llm or get_llm()
    chain_retriever = retriever or get_retriever(k=k)

    return RetrievalQA.from_chain_type(
        llm=chain_llm,
        chain_type="stuff",
        retriever=chain_retriever,
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": build_prompt()},
    )


def query(
    question: str,
    k: int = DEFAULT_TOP_K,
    qa_chain: Any | None = None,
) -> dict[str, Any]:
    """
    Query the RAG assistant and return answer text plus normalized citations.

    A prebuilt qa_chain can be injected for tests or Dash callbacks.
    """
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")

    try:
        chain = qa_chain or build_qa_chain(k=k)
    except Exception as exc:
        if qa_chain is not None:
            raise
        log.warning("Falling back to local chunk search because RAG chain could not start: %s", exc)
        return keyword_fallback_query(cleaned_question, k=k, error=exc)

    try:
        raw_result = chain.invoke({"query": cleaned_question})
    except AttributeError:
        raw_result = chain({"query": cleaned_question})
    except Exception as exc:
        if qa_chain is not None:
            log.exception("RAG query failed")
            raise RagChainError(f"RAG query failed: {exc}") from exc
        log.warning("Falling back to local chunk search because RAG query failed: %s", exc)
        return keyword_fallback_query(cleaned_question, k=k, error=exc)

    answer = _extract_answer(raw_result)
    answer = _CITATION_MARKER_RE.sub(" ", answer).strip()
    documents = raw_result.get("source_documents", []) if isinstance(raw_result, dict) else []
    citations = citations_from_documents(documents)

    if not answer:
        answer = FALLBACK_ANSWER
    answer = _append_source_summary(answer, citations)

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": [citation.as_dict() for citation in citations],
        "raw": raw_result,
    }


def keyword_fallback_query(
    question: str,
    k: int = DEFAULT_TOP_K,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Return a source-backed extractive answer when the vector/LLM path is unavailable."""
    rows = _rank_chunks_by_keyword(question, limit=k)
    if not rows:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "raw": {"fallback": "keyword", "error": str(error or "")},
        }

    citation_objects = [
        SourceCitation(
            source_id=row.get("source_id", ""),
            title=row.get("title", ""),
            chunk_index=_optional_int(row.get("chunk_index")),
            chunk_id=row.get("chunk_id", ""),
            url=str(row.get("metadata", {}).get("url", "")),
            file_name=str(row.get("metadata", {}).get("file_name", "")),
            snippet=_snippet(row.get("text", "")),
        )
        for row in rows
    ]
    citations = [citation.as_dict() for citation in citation_objects]
    bullets = "\n".join(f"- {_snippet(row.get('text', ''), max_chars=220)}" for row in rows[:2])
    answer = f"Based on the retrieved documents:\n\n{bullets}"
    answer = _append_source_summary(answer, citation_objects)
    return {
        "question": question,
        "answer": answer,
        "sources": citations,
        "raw": {"fallback": "keyword", "error": str(error or ""), "rows": rows},
    }


def _rank_chunks_by_keyword(question: str, limit: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    terms = [token for token in _TOKEN_RE.findall(question.lower()) if token not in _KEYWORD_STOPWORDS and len(token) > 2]
    if not terms or not CHUNKS_PATH.exists():
        return []

    scored: list[tuple[int, int, dict[str, Any]]] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            haystack = " ".join(
                [
                    str(row.get("title", "")),
                    str(row.get("source_id", "")),
                    str(row.get("text", "")),
                ]
            ).lower()
            score = sum(haystack.count(term) for term in terms)
            if all(term in haystack for term in terms[: min(3, len(terms))]):
                score += 5
            if score:
                scored.append((score, -row_number, row))

    scored.sort(reverse=True)
    return [row for _, _, row in scored[: max(limit, 1)]]


def citations_from_documents(documents: Iterable[Any]) -> list[SourceCitation]:
    """Normalize LangChain Document metadata into dashboard-friendly citations."""
    citations: list[SourceCitation] = []
    seen: set[tuple[str, str, int | None]] = set()

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source_id = str(metadata.get("source_id", "") or metadata.get("source", ""))
        title = str(metadata.get("title", "") or source_id)
        chunk_index = _optional_int(metadata.get("chunk_index"))
        key = (source_id, title, chunk_index)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            SourceCitation(
                source_id=source_id,
                title=title,
                chunk_index=chunk_index,
                chunk_id=str(metadata.get("chunk_id", "")),
                url=str(metadata.get("url", "")),
                file_name=str(metadata.get("file_name", "")),
                snippet=_snippet(getattr(document, "page_content", "")),
            )
        )

    return citations


def format_response(result: dict[str, Any]) -> str:
    """Format a query() result as a readable text block for scripts/CLI use."""
    answer = result.get("answer", FALLBACK_ANSWER)
    sources = result.get("sources", [])
    if not sources:
        return answer

    source_lines = [f"- {source['citation']}" for source in sources]
    return f"{answer}\n\nSources:\n" + "\n".join(source_lines)


def _extract_answer(raw_result: Any) -> str:
    if isinstance(raw_result, dict):
        return str(raw_result.get("result") or raw_result.get("answer") or "").strip()
    return str(raw_result or "").strip()


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snippet(text: str, max_chars: int = 260) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _append_source_summary(answer: str, citations: list[SourceCitation]) -> str:
    titles: list[str] = []
    seen: set[str] = set()
    for citation in citations:
        title = (citation.title or citation.source_id or "").strip()
        if not title or title in seen:
            continue
        seen.add(title)
        titles.append(title)
        if len(titles) == 3:
            break

    if not titles:
        return answer
    return f"{answer}\n\nSources: " + "; ".join(titles)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the MedAlertAI RAG assistant a question.")
    parser.add_argument("question", help="Protocol or dispatch question to ask.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    print(format_response(query(args.question, k=args.k)))
