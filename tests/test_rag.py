"""
test_rag.py - RAG chain tests.

Owners: Srileakhana (C4), Deekshitha (C5)
Phase: 5
"""

from types import SimpleNamespace

import pytest

from src.rag import chain


class FakeRetriever:
    pass


class FakeVectorStore:
    def __init__(self):
        self.search_kwargs = None

    def as_retriever(self, search_kwargs):
        self.search_kwargs = search_kwargs
        return FakeRetriever()


class FakeQAChain:
    def invoke(self, payload):
        assert payload == {"query": "What does MPDS 17D1 mean?"}
        return {
            "result": (
                "MPDS 17D1 should be handled according to the retrieved "
                "dispatch mapping. [source: MPDS Dispatch Mapping; chunk: 3]"
            ),
            "source_documents": [
                SimpleNamespace(
                    page_content="17D1 reference text with dispatch mapping details.",
                    metadata={
                        "source_id": "mpds_dispatch_mapping",
                        "title": "MPDS Dispatch Mapping",
                        "chunk_index": 3,
                        "chunk_id": "abc123",
                        "url": "https://example.test/mpds",
                        "file_name": "mpds_dispatch_mapping.md",
                    },
                )
            ],
        }


def test_prompt_contains_protocol_only_fallback():
    prompt = chain.SYSTEM_PROMPT

    assert "Use only the retrieved" in prompt
    assert "Do not use outside medical knowledge" in prompt
    assert "{fallback_answer}" in prompt
    assert "[source: <title or source_id>; chunk: <chunk_index>]" in prompt


def test_get_retriever_passes_top_k_to_vectorstore():
    vectorstore = FakeVectorStore()

    retriever = chain.get_retriever(k=5, vectorstore=vectorstore)

    assert isinstance(retriever, FakeRetriever)
    assert vectorstore.search_kwargs == {"k": 5}


def test_get_retriever_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        chain.get_retriever(k=0, vectorstore=FakeVectorStore())


def test_query_returns_answer_and_normalized_sources():
    result = chain.query("  What does MPDS 17D1 mean?  ", qa_chain=FakeQAChain())

    assert result["question"] == "What does MPDS 17D1 mean?"
    assert "MPDS 17D1" in result["answer"]
    assert result["sources"] == [
        {
            "source_id": "mpds_dispatch_mapping",
            "title": "MPDS Dispatch Mapping",
            "chunk_index": 3,
            "chunk_id": "abc123",
            "url": "https://example.test/mpds",
            "file_name": "mpds_dispatch_mapping.md",
            "snippet": "17D1 reference text with dispatch mapping details.",
            "citation": "[source: MPDS Dispatch Mapping; chunk: 3]",
        }
    ]


def test_query_rejects_blank_question():
    with pytest.raises(ValueError, match="question must not be empty"):
        chain.query("   ", qa_chain=FakeQAChain())


def test_citations_are_deduplicated_and_snippets_are_trimmed():
    long_text = " ".join(["protocol"] * 80)
    documents = [
        SimpleNamespace(
            page_content=long_text,
            metadata={"source_id": "pa_doh", "title": "PA DOH Protocols", "chunk_index": "2"},
        ),
        SimpleNamespace(
            page_content="duplicate",
            metadata={"source_id": "pa_doh", "title": "PA DOH Protocols", "chunk_index": 2},
        ),
    ]

    citations = chain.citations_from_documents(documents)

    assert len(citations) == 1
    assert citations[0].chunk_index == 2
    assert citations[0].label() == "[source: PA DOH Protocols; chunk: 2]"
    assert citations[0].snippet.endswith("...")
    assert len(citations[0].snippet) <= 260


def test_format_response_adds_sources_section():
    formatted = chain.format_response(
        {
            "answer": "Use the retrieved protocol guidance.",
            "sources": [{"citation": "[source: PA DOH Protocols; chunk: 2]"}],
        }
    )

    assert "Use the retrieved protocol guidance." in formatted
    assert "Sources:" in formatted
    assert "- [source: PA DOH Protocols; chunk: 2]" in formatted
