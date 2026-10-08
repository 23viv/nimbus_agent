"""
Nimbus Support Agent — RAG Pipeline (MongoDB Atlas Full-Text Search)
Handles document ingestion and semantic-keyword retrieval using MongoDB Atlas
Search (Lucene-based). No local models, no embedding API calls needed.

Zero memory footprint on the server — fits well within Render's 512 MB free tier.

Environment variables required:
  MONGODB_URI       — MongoDB Atlas connection string
  MONGODB_DB_NAME   — database name (default: nimbus_db)

Atlas Search index required:
  Collection : nimbus_db.knowledge_chunks
  Index name : text_search_index
  Type       : Search (not Vector Search)
  Analyzer   : lucene.english
  Field      : text
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langfuse import observe

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
_DOCS_DIR         = Path(__file__).parent.parent / "docs"
_COLLECTION_NAME  = "knowledge_chunks"
_CHUNK_SIZE       = 300   # target words per chunk
_CHUNK_OVERLAP    = 50    # word overlap between chunks
_TOP_K            = 4     # number of chunks to retrieve

# ── MongoDB connection ─────────────────────────────────────────────────────────
_MONGODB_URI     = os.getenv("MONGODB_URI")
_MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "nimbus_db")

_mongo_client   = None
_collection_ref = None


def _get_collection():
    """Return (and lazily initialise) the MongoDB collection handle."""
    global _mongo_client, _collection_ref
    if _collection_ref is not None:
        return _collection_ref

    if not _MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set — cannot use MongoDB Atlas Search.")

    from pymongo import MongoClient
    _mongo_client = MongoClient(_MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = _mongo_client[_MONGODB_DB_NAME]
    _collection_ref = db[_COLLECTION_NAME]
    return _collection_ref


# ── Text chunking ──────────────────────────────────────────────────────────────
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
            "chunk_id":    f"{source}__chunk_{chunk_idx:04d}",
            "text":        chunk_text,
            "source":      source,
            "chunk_index": chunk_idx,
        })
        chunk_idx += 1
        if end == len(words):
            break
        start = end - _CHUNK_OVERLAP

    return chunks


# ── Public API ─────────────────────────────────────────────────────────────────
@observe(name="rag.ingest_documents")
def ingest_documents(force_rebuild: bool = False) -> int:
    """
    Read all .txt files from docs/, chunk them, and upsert into MongoDB.
    No embeddings are computed — just stores raw text for full-text search.
    Safe to call multiple times — skips chunks already indexed unless force_rebuild=True.
    Returns the total number of chunks in the collection after ingestion.
    """
    collection = _get_collection()

    if force_rebuild:
        collection.delete_many({})
        print("[rag] Cleared existing knowledge_chunks collection.")

    existing_ids: set[str] = set(
        doc["chunk_id"] for doc in collection.find({}, {"chunk_id": 1})
    )

    doc_files = list(_DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt documents found in {_DOCS_DIR}")

    new_chunks: list[dict] = []
    for doc_path in doc_files:
        source = doc_path.stem
        text   = doc_path.read_text(encoding="utf-8")
        for chunk in _chunk_text(text, source):
            if chunk["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)

    if new_chunks:
        collection.insert_many(new_chunks)
        print(f"[rag] Inserted {len(new_chunks)} new chunks into MongoDB.")

    return collection.count_documents({})


@observe(name="rag.retrieve")
def retrieve(query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query using MongoDB Atlas
    Full-Text Search. Falls back to a simple regex scan if the Atlas Search
    index is not yet available.

    Returns a list of dicts: {text, source, score}.
    """
    collection = _get_collection()

    # Auto-ingest if empty
    if collection.count_documents({}) == 0:
        print("[rag] Collection is empty — running ingest_documents().")
        ingest_documents()

    if collection.count_documents({}) == 0:
        return []

    # ── Try Atlas Full-Text Search first ──────────────────────────────────────
    try:
        pipeline = [
            {
                "$search": {
                    "index": "text_search_index",
                    "text": {
                        "query": query,
                        "path":  "text",
                    },
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "searchScore"}
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "_id":    0,
                    "text":   1,
                    "source": 1,
                    "score":  1,
                }
            },
        ]
        results = list(collection.aggregate(pipeline))
        if results:
            return [
                {
                    "text":   doc["text"],
                    "source": doc["source"],
                    "score":  round(doc.get("score", 1.0), 4),
                }
                for doc in results
            ]
    except Exception as e:
        print(f"[rag] Atlas Search unavailable ({e}), falling back to regex scan.")

    # ── Fallback: simple keyword scan (no index required) ─────────────────────
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    if not keywords:
        return []

    regex = "|".join(re.escape(k) for k in keywords)
    cursor = collection.find(
        {"text": {"$regex": regex, "$options": "i"}},
        {"_id": 0, "text": 1, "source": 1},
    ).limit(top_k)

    return [
        {"text": doc["text"], "source": doc["source"], "score": 0.5}
        for doc in cursor
    ]


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
