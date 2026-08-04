"""
Nimbus Support Agent — RAG Pipeline
Handles document ingestion (chunking + embedding) and retrieval.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────
_DOCS_DIR = Path(__file__).parent.parent / "docs"
_VECTOR_STORE_DIR = Path(__file__).parent.parent / "vector_store"
_COLLECTION_NAME = "nimbus_knowledge"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE = 300       # target words per chunk
_CHUNK_OVERLAP = 50     # word overlap between chunks
_TOP_K = 4              # number of chunks to retrieve
# ChromaDB returns L2 distances; lower = more similar.
# Distance > 1.2 means the chunk is likely off-topic.
_DISTANCE_THRESHOLD = 1.2


# ── Embedding model (lazy-loaded singleton) ───────────────────────────────────
_embed_model: SentenceTransformer | None = None

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(_EMBED_MODEL)
    return _embed_model


# ── ChromaDB client (lazy-loaded singleton) ───────────────────────────────────
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None

def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=str(_VECTOR_STORE_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )
    return _collection


# ── Text chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + _CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id": f"{source}__chunk_{chunk_idx:04d}",
            "text": chunk_text,
            "source": source,
        })
        chunk_idx += 1
        if end == len(words):
            break
        start = end - _CHUNK_OVERLAP

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────
def ingest_documents(force_rebuild: bool = False) -> int:
    """
    Read all .txt files from docs/, chunk, embed, and store in ChromaDB.
    Skips ingestion if the collection already has documents (unless force_rebuild=True).
    Returns the number of chunks stored.
    """
    collection = _get_collection()

    if collection.count() > 0 and not force_rebuild:
        return collection.count()

    if force_rebuild and collection.count() > 0:
        collection.delete(where={"source": {"$ne": "__nonexistent__"}})

    doc_files = list(_DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt documents found in {_DOCS_DIR}")

    all_chunks: list[dict] = []
    for doc_path in doc_files:
        source = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        all_chunks.extend(_chunk_text(text, source))

    model = _get_embed_model()
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    return len(all_chunks)


def retrieve(query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query.
    Returns a list of dicts: {text, source, distance}.
    Only returns chunks with distance <= _DISTANCE_THRESHOLD.
    Returns an empty list if nothing relevant is found.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    model = _get_embed_model()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist <= _DISTANCE_THRESHOLD:
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            })

    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        source_label = chunk["source"].replace("_", " ").title()
        parts.append(f"[Source: {source_label}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
