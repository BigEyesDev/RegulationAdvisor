"""
Entry point for the Gradio application.

Run with:
    uv run python src/regulation_advisor/ui/app_runner.py

What this script does (in order):
1. Load the pre-built FAISS index from data/index/
2. Build a Retriever (embedder + store)
3. Build the Gradio UI (passing the retriever in)
4. Launch on http://localhost:7860
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.retrieval.store import FAISSVectorStore
from regulation_advisor.ui.gradio_app import build_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_INDEX_DIR = Path("data/index")


def _load_retriever() -> Retriever:
    """Load the persisted FAISS index and wrap it in a Retriever."""
    if not (_INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {_INDEX_DIR}.\n"
            "Run the ingestion script first:\n"
            "    uv run python scripts/ingest.py"
        )

    logger.info("Loading FAISS index from %s …", _INDEX_DIR)
    store = FAISSVectorStore()
    store.load(_INDEX_DIR)

    logger.info("Loading embedding model …")
    embedder = SentenceTransformerEmbedder()

    return Retriever(store=store, embedder=embedder)


if __name__ == "__main__":
    retriever = _load_retriever()
    demo = build_ui(retriever)
    logger.info("Launching Gradio UI at http://localhost:7860")
    demo.launch()
