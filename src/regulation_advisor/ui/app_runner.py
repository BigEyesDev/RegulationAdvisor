"""
Application entry point — RegulationAdvisor v0.2

Local run:
    uv run python src/regulation_advisor/ui/app_runner.py

HuggingFace Spaces:
    HF executes this file directly with `python app_runner.py`.
    The app_file in README.md front-matter points here.
    API keys are read from HF Space Secrets (set via the Space Settings UI).

Startup sequence (v0.2):
    1. Auto-run ingestion if data/index/ is missing (HF cold start or fresh clone)
    2. Load FAISS index + embedding model into a Retriever
    3. Wire the retriever into the agent tools module (dependency injection)
    4. Build the LangGraph agent graph
    5. Build the Gradio UI around the agent
    6. Launch the web server
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_SRC = Path(__file__).parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent
_DATA_DIR = _ROOT / "data"
_INDEX_DIR = _ROOT / "data" / "index"


def _ensure_index() -> None:
    """Build the FAISS index on first run; no-op if it already exists."""
    if (_INDEX_DIR / "index.faiss").exists():
        logger.info("Index already built — skipping ingestion.")
        return
    logger.info("Index not found — running ingestion (~20 s on first run)…")
    from regulation_advisor.ingestion.pipeline import run_ingestion
    run_ingestion(data_dir=_DATA_DIR, index_dir=_INDEX_DIR)
    logger.info("Ingestion complete.")


def _load_retriever():
    """Load the vector store (FAISS or ChromaDB depending on config) and return a Retriever."""
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.retriever import Retriever
    from regulation_advisor.retrieval.store import build_vector_store
    from regulation_advisor.config import settings

    logger.info("Loading vector store (backend=%s)…", settings.vector_store_backend)
    store = build_vector_store()
    store.load(_INDEX_DIR)

    logger.info("Loading embedding model …")
    embedder = SentenceTransformerEmbedder()

    return Retriever(store=store, embedder=embedder)


if __name__ == "__main__":
    from regulation_advisor.agent.graph import build_agent_graph
    from regulation_advisor.agent.tools import set_retriever
    from regulation_advisor.api.routes import set_agent
    from regulation_advisor.ui.gradio_app import build_ui

    _ensure_index()
    retriever = _load_retriever()

    set_retriever(retriever)
    agent = build_agent_graph()
    set_agent(agent)  # makes agent available to build_ui()'s lazy _get_agent()
    demo = build_ui()

    port = int(os.environ.get("PORT", 7860))
    logger.info("Launching on 0.0.0.0:%d", port)
    demo.launch(server_name="0.0.0.0", server_port=port)
