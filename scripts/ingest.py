"""
scripts/ingest.py
One-shot script to chunk and embed Nimbus company documents into ChromaDB.
Run this once before starting the agent, or whenever docs/ changes.

Usage:
    python scripts/ingest.py           # build if not already built
    python scripts/ingest.py --rebuild # force rebuild from scratch
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.rag import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="Ingest Nimbus docs into vector store.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force delete existing embeddings and rebuild from scratch.",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Nimbus Knowledge Base Ingestion")
    print("=" * 50)

    try:
        count = ingest_documents(force_rebuild=args.rebuild)
        print(f"\n✅ Done! {count} chunks stored in vector store.")
        print("   Location: vector_store/")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure docs/ contains .txt files.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
