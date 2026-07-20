"""Retriever — wraps the vector store with query encoding."""
from __future__ import annotations

import logging

from regulation_advisor.models import RetrievalResult
from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore, embedder: SentenceTransformerEmbedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(self, query: str, k: int = 5) -> RetrievalResult:
        embedding = self._embedder.encode([query])[0]
        chunks = self._store.search(embedding, k=k)
        logger.info("Retrieved %d chunks for query: %.50s...", len(chunks), query)
        return RetrievalResult(chunks=chunks, query=query)
