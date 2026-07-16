"""
FastAPI application — RegulationAdvisor v0.4.

Run:   uvicorn regulation_advisor.api.app:app --reload --port 8000
Docs:  http://localhost:8000/docs
UI:    http://localhost:8000/
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from regulation_advisor.api import routes
from regulation_advisor.ui.gradio_app import build_ui

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index + agent once at startup. Tear down cleanly on shutdown."""
    from regulation_advisor.agent.graph import build_agent_graph
    from regulation_advisor.agent.tools import set_retriever
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.retriever import Retriever
    from regulation_advisor.retrieval.store import build_vector_store
    from regulation_advisor.config import settings

    logger.info("Loading vector store (%s)…", settings.vector_store_backend)
    store = build_vector_store()
    store.load(_ROOT / settings.index_dir)

    embedder = SentenceTransformerEmbedder()
    retriever = Retriever(store=store, embedder=embedder)
    set_retriever(retriever)

    agent = build_agent_graph()
    routes.set_agent(agent)
    logger.info("Agent ready")
    yield
    logger.info("Shutdown complete")


_fastapi_app = FastAPI(
    title="RegulationAdvisor API",
    version="0.5.0",
    lifespan=lifespan,
)
_fastapi_app.include_router(routes.router)

# Gradio reads routes._agent lazily at request time (set during lifespan above).
# This mount line runs at module import, before lifespan fires — that is fine
# because Gradio only calls respond() when a user sends a message.
demo = build_ui()
app = gr.mount_gradio_app(_fastapi_app, demo, path="/")
