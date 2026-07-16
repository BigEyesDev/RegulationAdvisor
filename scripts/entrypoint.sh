#!/bin/bash
# Container entrypoint for docker-compose (ChromaDB mode).
# Ingests regulation documents into ChromaDB only if the collection is empty,
# then starts the FastAPI server.
set -euo pipefail

needs_ingest() {
    python - <<'PYEOF'
import sys, chromadb
try:
    col = chromadb.HttpClient(host="chromadb", port=8000).get_or_create_collection("regulations")
    sys.exit(0 if col.count() > 0 else 1)
except Exception as e:
    print(f"ChromaDB check failed: {e}", flush=True)
    sys.exit(1)
PYEOF
}

if ! needs_ingest; then
    echo "Vector store is empty — running ingestion (one-time setup)..."
    python scripts/ingest.py
    echo "Ingestion complete."
fi

exec uvicorn regulation_advisor.api.app:app --host 0.0.0.0 --port 8000
