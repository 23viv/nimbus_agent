"""
Nimbus Support Agent — RAG Pipeline (ChromaDB Vector Store)
Handles document ingestion, embedding, persistent storage, and semantic retrieval.
Uses ChromaDB with SentenceTransformer embeddings saved locally in vector_store/.
"""

import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langfuse import observe

# ── Configuration ─────────────────────────────────────────────────────────────
_DOCS_DIR        = Path(__file__).parent.parent / "docs"
_VECTOR_DIR      = Path(__file__).parent.parent / "vector_store"
_COLLECTION_NAME = "nimbus_docs"
_CHUNK_SIZE      = 300   # target words per chunk
_CHUNK_OVERLAP   = 50    # word overlap between chunks
_TOP_K           = 4     # number of chunks to retrieve

# Embedding model: SentenceTransformers default ONNX embedding function
_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

# Persistent ChromaDB client
_client = chromadb.PersistentClient(path=str(_VECTOR_DIR))


def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    words     = text.split()
    chunks    = []
    start     = 0
    chunk_idx = 0

    while start < len(words):
        end        = min(start + _CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id":       f"{source}__chunk_{chunk_idx:04d}",
            "text":     chunk_text,
            "metadata": {"source": source, "chunk_index": chunk_idx},
        })
        chunk_idx += 1
        if end == len(words):
            break
        start = end - _CHUNK_OVERLAP

    return chunks


def _get_collection():
    """Get or create the ChromaDB collection with the default embedding function."""
    return _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


# ── Public API ────────────────────────────────────────────────────────────────
@observe(name="rag.ingest_documents")
def ingest_documents(force_rebuild: bool = False) -> int:
    """
    Read all .txt files from docs/, chunk and embed them into ChromaDB.
    Persistent store is saved in vector_store/.
    Returns the number of chunks indexed.
    """
    if force_rebuild:
        try:
            _client.delete_collection(name=_COLLECTION_NAME)
        except Exception:
            pass

    collection = _get_collection()
    existing_count = collection.count()

    if existing_count > 0 and not force_rebuild:
        return existing_count

    doc_files = list(_DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt documents found in {_DOCS_DIR}")

    all_chunks = []
    for doc_path in doc_files:
        source = doc_path.stem
        text   = doc_path.read_text(encoding="utf-8")
        all_chunks.extend(_chunk_text(text, source))

    if not all_chunks:
        return 0

    ids = [c["id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection.count()


@observe(name="rag.retrieve")
def retrieve(query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query using ChromaDB semantic vector search.
    Returns a list of dicts: {text, source, score}.
    """
    collection = _get_collection()
    if collection.count() == 0:
        ingest_documents()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to similarity score (higher is better)
        similarity_score = max(0.0, round(1.0 - dist, 4))
        retrieved.append({
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "score":  similarity_score,
        })

    return retrieved


@observe(name="rag.format_context")
def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        source_label = chunk["source"].replace("_", " ").title()
        parts.append(f"[Source: {source_label}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)

