"""
src/rag/vectorstore.py

Phase 3 - Vector store setup and embedding generation.
Owner: Sanika (C3)

Responsibilities covered here:
  - Read JSONL chunks from data/processed/rag/chunks.jsonl
  - Initialize HuggingFaceEmbeddings with all-MiniLM-L6-v2 (384-dim)
  - Create/Update a ChromaDB persistent vector store
  - Verify store integrity

Usage from the repo root:
    python -m src.rag.vectorstore
"""

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config.settings import PROCESSED_DATA_DIR, CHROMA_PERSIST_DIR

log = logging.getLogger("medalertai.rag.vectorstore")

CHUNKS_PATH = PROCESSED_DATA_DIR / "rag" / "chunks.jsonl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "medalertai_rag"


def load_chunks_from_jsonl(path: Path = CHUNKS_PATH) -> list[Document]:
    """Loads chunks from the JSONL file and converts them to LangChain Documents."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at {path}. Run ingest.py first.")

    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # data structure matches what was created in src/rag/ingest.py
            page_content = data.get("text", "")
            metadata = data.get("metadata", {})
            
            # include essential chunk IDs in metadata
            metadata["chunk_id"] = data.get("chunk_id", "")
            metadata["source_id"] = data.get("source_id", "")
            metadata["chunk_index"] = data.get("chunk_index", 0)
            metadata["title"] = data.get("title", "")
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
    log.info("Loaded %d documents from %s", len(documents), path)
    return documents


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Returns the HuggingFace embeddings model (384-dim)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(documents: list[Document], persist_directory: Path = CHROMA_PERSIST_DIR) -> Chroma:
    """Initializes and populates the ChromaDB persistent vector store."""
    embeddings = get_embeddings_model()
    
    log.info("Initializing ChromaDB at %s", persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    # We use from_documents to build/append to the vector store
    # Providing persist_directory makes it persistent.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory)
    )
    
    # In newer Chroma/Langchain versions, from_documents persists automatically if not using in-memory
    # vectorstore.persist() might be deprecated but persist_directory handles it
    log.info("Finished embedding and indexing %d documents into ChromaDB.", len(documents))
    return vectorstore


def get_vectorstore(persist_directory: Path = CHROMA_PERSIST_DIR) -> Chroma:
    """Loads the existing persistent ChromaDB vector store."""
    if not persist_directory.exists():
        log.warning("ChromaDB directory does not exist yet at %s", persist_directory)

    embeddings = get_embeddings_model()
    # In LangChain, instantiating Chroma with persist_directory loads it if it exists.
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory)
    )


def verify_store_integrity(persist_directory: Path = CHROMA_PERSIST_DIR) -> bool:
    """Verifies that the vector store can be loaded and responds to queries."""
    if not persist_directory.exists():
        log.error("ChromaDB directory does not exist: %s", persist_directory)
        return False

    vectorstore = get_vectorstore(persist_directory)
    
    count = vectorstore._collection.count()
    log.info("Vector store holds %d items.", count)
    
    if count == 0:
        log.error("Vector store is empty!")
        return False
        
    # Quick connectivity/sanity test
    query = "What is the dispatch code for chest pain?"
    results = vectorstore.similarity_search(query, k=1)
    
    if results:
        log.info("Sanity check query successful. Top match title: '%s'", results[0].metadata.get("title", "Unknown"))
        return True
    else:
        log.warning("Sanity check query returned no results.")
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    
    try:
        docs = load_chunks_from_jsonl()
    except FileNotFoundError as e:
        log.error(e)
        return 1

    if not docs:
        log.error("No documents loaded. Please ensure ingest.py produced valid chunks.")
        return 1

    _ = build_vectorstore(docs)
    
    ok = verify_store_integrity()
    if ok:
        log.info("Vector store setup is complete and verified!")
        return 0
    else:
        log.error("Vector store integrity check failed.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
