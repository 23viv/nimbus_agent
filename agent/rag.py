"""
Nimbus Support Agent — RAG Pipeline
Handles document ingestion (chunking) and retrieval.
Uses pure BM25 keyword search — no vector database or ML embeddings required.
"""

import math
import re
from collections import Counter
from pathlib import Path

from langsmith import traceable

# ── Configuration ─────────────────────────────────────────────────────────────
_DOCS_DIR        = Path(__file__).parent.parent / "docs"
_CHUNK_SIZE      = 300   # target words per chunk
_CHUNK_OVERLAP   = 50    # word overlap between chunks
_TOP_K           = 4     # number of chunks to retrieve
_SCORE_THRESHOLD = 0.01  # minimum BM25 score to include a result

# BM25 hyperparameters (standard defaults)
_BM25_K1 = 1.5
_BM25_B  = 0.75


# ── In-memory store ───────────────────────────────────────────────────────────
# Each entry: {"id": str, "text": str, "source": str, "tokens": list[str]}
_chunks: list[dict] = []


# ── Text helpers ──────────────────────────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "this", "that", "are", "was",
    "be", "by", "from", "as", "we", "you", "your", "our", "can",
    "will", "has", "have", "had", "not", "if", "so", "do", "up",
    "its", "my", "i", "no", "any", "all", "also", "than", "then",
}

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


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
            "id":     f"{source}__chunk_{chunk_idx:04d}",
            "text":   chunk_text,
            "source": source,
            "tokens": _tokenize(chunk_text),
        })
        chunk_idx += 1
        if end == len(words):
            break
        start = end - _CHUNK_OVERLAP

    return chunks


# ── BM25 scoring ──────────────────────────────────────────────────────────────
def _bm25_scores(query_tokens: list[str]) -> list[float]:
    """Return a BM25 score for each chunk in _chunks."""
    if not _chunks:
        return []

    N      = len(_chunks)
    avgdl  = sum(len(c["tokens"]) for c in _chunks) / N

    # Document frequency per term
    df: dict[str, int] = Counter()
    for chunk in _chunks:
        for term in set(chunk["tokens"]):
            df[term] += 1

    scores = []
    for chunk in _chunks:
        tf_map = Counter(chunk["tokens"])
        dl     = len(chunk["tokens"])
        score  = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf  = tf_map[term]
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = (tf * (_BM25_K1 + 1)) / (
                tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
            )
            score += idf * tf_norm
        scores.append(score)

    return scores


# ── Public API ────────────────────────────────────────────────────────────────
@traceable(name="rag.ingest_documents")
def ingest_documents(force_rebuild: bool = False) -> int:
    """
    Read all .txt files from docs/, chunk and index them in memory.
    Skips ingestion if already loaded (unless force_rebuild=True).
    Returns the number of chunks indexed.
    """
    global _chunks

    if _chunks and not force_rebuild:
        return len(_chunks)

    doc_files = list(_DOCS_DIR.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No .txt documents found in {_DOCS_DIR}")

    _chunks = []
    for doc_path in doc_files:
        source = doc_path.stem
        text   = doc_path.read_text(encoding="utf-8")
        _chunks.extend(_chunk_text(text, source))

    return len(_chunks)


@traceable(name="rag.retrieve")
def retrieve(query: str, top_k: int = _TOP_K) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query using BM25.
    Returns a list of dicts: {text, source, score}.
    Only returns chunks with score >= _SCORE_THRESHOLD.
    Returns an empty list if nothing relevant is found.
    """
    if not _chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores  = _bm25_scores(query_tokens)
    ranked  = sorted(zip(scores, _chunks), key=lambda x: x[0], reverse=True)
    results = []

    for score, chunk in ranked[:top_k]:
        if score >= _SCORE_THRESHOLD:
            results.append({
                "text":   chunk["text"],
                "source": chunk["source"],
                "score":  round(score, 4),
            })

    return results


@traceable(name="rag.format_context")
def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return ""
    parts = []
    for chunk in chunks:
        source_label = chunk["source"].replace("_", " ").title()
        parts.append(f"[Source: {source_label}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
