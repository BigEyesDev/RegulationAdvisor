"""
Entry point for the Gradio application.

Local run:
    uv run python src/regulation_advisor/ui/app_runner.py

HuggingFace Spaces:
    HF executes this file directly with `python app_runner.py`.
    The app_file in README.md front-matter points here.
    API keys are read from HF Space Secrets (set via the Space Settings UI).

Startup sequence:
    1. Auto-run ingestion if data/index/ is missing (HF cold start or fresh clone)
    2. Load FAISS index + embedding model into memory
    3. Build Gradio UI
    4. Launch web server
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable whether run via `uv run` or plain `python`
_SRC = Path(__file__).parent.parent.parent  # .../src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Resolve paths relative to this file so they work regardless of cwd
_ROOT = Path(__file__).parent.parent.parent.parent  # project root
_DATA_DIR = _ROOT / "data"
_INDEX_DIR = _ROOT / "data" / "index"


def _ensure_index() -> None:
    """
    Build the FAISS index if it does not exist yet.
    This runs automatically on first HF Spaces cold start, or on a fresh local clone.
    On subsequent startups the index is already on disk — this is a no-op.
    """
    if (_INDEX_DIR / "index.faiss").exists():
        logger.info("Index already built at %s — skipping ingestion.", _INDEX_DIR)
        return

    logger.info("Index not found — running ingestion (this takes ~20 s on first run)…")
    from regulation_advisor.ingestion.pipeline import run_ingestion

    run_ingestion(data_dir=_DATA_DIR, index_dir=_INDEX_DIR)
    logger.info("Ingestion complete.")


def _load_retriever():
    """Load the persisted FAISS index and wrap it in a Retriever."""
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.retriever import Retriever
    from regulation_advisor.retrieval.store import FAISSVectorStore

    logger.info("Loading FAISS index from %s …", _INDEX_DIR)
    store = FAISSVectorStore()
    store.load(_INDEX_DIR)

    logger.info("Loading embedding model …")
    embedder = SentenceTransformerEmbedder()

    return Retriever(store=store, embedder=embedder)


if __name__ == "__main__":
    _ensure_index()
    retriever = _load_retriever()

    from regulation_advisor.ui.gradio_app import build_ui

    demo = build_ui(retriever)

    # HF Spaces injects PORT env var; fall back to 7860 locally.
    port = int(os.environ.get("PORT", 7860))
    # server_name="0.0.0.0" makes the server reachable from outside the container.
    logger.info("Launching Gradio UI on 0.0.0.0:%d", port)
    demo.launch(server_name="0.0.0.0", server_port=port)
