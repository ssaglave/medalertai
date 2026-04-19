"""
src/rag/chain.py

Phase 3 - LangChain RetrievalQA chain with Claude.
Owner: Srileakhana (C4)

Responsibilities covered here:
  - Build a protocol-only RetrievalQA chain over the ChromaDB retriever.
  - Use Claude Haiku by default when an Anthropic API key is configured.
  - Return citation-ready source metadata for dashboard callbacks and tests.

Usage from the repo root after running ingestion/vectorstore setup:
    python -c "from src.rag.chain import query; print(query('MPDS 17D1'))"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable

from config.settings import ANTHROPIC_API_KEY


log = logging.getLogger("medalertai.rag.chain")

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
FALLBACK_ANSWER = (
    "I do not have enough protocol context in the retrieved sources to answer "
    "that safely. Please consult the official EMS protocol or a qualified "
    "clinical/dispatch supervisor."
)

SYSTEM_PROMPT = """You are MedAlertAI's EMS and fire dispatch protocol assistant.

Use only the retrieved protocol, NEMSIS, MPDS mapping, NFPA, and WPRDC context below.
Do not use outside medical knowledge, do not invent protocol details, and do not
provide medical direction beyond what the retrieved context supports.

If the context does not contain the answer, say exactly:
"{fallback_answer}"

When you answer, include concise citations in this format:
[source: <title or source_id>; chunk: <chunk_index>]

Retrieved context:
{context}

Question: {question}

Answer:"""


class RagChainError(RuntimeError):
    """Raised when the RAG chain cannot be created or queried."""


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
    """Return the Claude chat model used by the RAG chain."""
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RagChainError("ANTHROPIC_API_KEY is required to query Claude.")

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RagChainError(
            "langchain-anthropic is required for Claude RetrievalQA. "
            "Install requirements.txt before running the RAG assistant."
        ) from exc

    return ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=temperature,
        timeout=30,
        max_retries=2,
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

    chain = qa_chain or build_qa_chain(k=k)
    try:
        raw_result = chain.invoke({"query": cleaned_question})
    except AttributeError:
        raw_result = chain({"query": cleaned_question})
    except Exception as exc:
        log.exception("RAG query failed")
        raise RagChainError(f"RAG query failed: {exc}") from exc

    answer = _extract_answer(raw_result)
    documents = raw_result.get("source_documents", []) if isinstance(raw_result, dict) else []
    citations = citations_from_documents(documents)

    if not answer:
        answer = FALLBACK_ANSWER

    return {
        "question": cleaned_question,
        "answer": answer,
        "sources": [citation.as_dict() for citation in citations],
        "raw": raw_result,
    }


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the MedAlertAI RAG assistant a question.")
    parser.add_argument("question", help="Protocol or dispatch question to ask.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    print(format_response(query(args.question, k=args.k)))
