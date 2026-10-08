"""
One-time ingestion script: reads docs/*.txt, chunks them, and upserts
plain text into MongoDB Atlas (nimbus_db.knowledge_chunks).

No embeddings or API calls needed — uses MongoDB Atlas Full-Text Search.

Run this ONCE locally (or on first deploy):
    python scripts/ingest_to_mongo.py
    python scripts/ingest_to_mongo.py --force   # to rebuild from scratch

After ingestion, create an Atlas Search index:
  Go to: Atlas → Browse Collections → nimbus_db → knowledge_chunks
       → Search Indexes → Create Search Index → Atlas Search (not Vector)

  Use this JSON definition:
  {
    "mappings": {
      "dynamic": false,
      "fields": {
        "text": {
          "type": "string",
          "analyzer": "lucene.english"
        },
        "source": { "type": "string" }
      }
    }
  }

  Index name: text_search_index
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.rag import ingest_documents

if __name__ == "__main__":
    print("=== Nimbus Knowledge Base Ingestion ===")
    print("Chunking docs and uploading plain text to MongoDB Atlas...")
    force = "--force" in sys.argv
    if force:
        print("--force flag detected: clearing and rebuilding from scratch.")
    total = ingest_documents(force_rebuild=force)
    print(f"\nDone! {total} chunks now stored in MongoDB Atlas (knowledge_chunks).")
    print("\nNext step — create the Atlas Search index:")
    print("  Atlas → nimbus_db → knowledge_chunks → Search Indexes")
    print("  → Create Search Index → Atlas Search (not Vector Search)")
    print("  → Index name: text_search_index")
    print("  → Field: text (type: string, analyzer: lucene.english)")
