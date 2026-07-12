"""Ingestion pipeline — run via: python scripts/ingest.py"""
from __future__ import annotations

import logging
from pathlib import Path

from regulation_advisor.config import settings
from regulation_advisor.ingestion.chunkers import ArticleAwareChunker, Chunker, RecursiveCharacterChunker
from regulation_advisor.ingestion.loaders import DocumentLoaderFactory

logger = logging.getLogger(__name__)


def _build_default_chunker() -> Chunker:
    """Build the chunker specified in settings.chunker_strategy."""
    if settings.chunker_strategy == "recursive":
        logger.info(
            "Using RecursiveCharacterChunker (size=%d, overlap=%d)",
            settings.chunk_size,
            settings.chunk_overlap,
        )
        return RecursiveCharacterChunker(size=settings.chunk_size, overlap=settings.chunk_overlap)
    logger.info("Using ArticleAwareChunker (strategy=%s)", settings.chunker_strategy)
    return ArticleAwareChunker()


def run_ingestion(data_dir: Path, index_dir: Path, chunker: Chunker | None = None) -> None:
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.store import build_vector_store

    chunker = chunker or _build_default_chunker()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = build_vector_store()
    all_chunks = []

    for path in sorted(data_dir.iterdir()):
        if not DocumentLoaderFactory.supports(path):
            continue
        texts = DocumentLoaderFactory.create(path).load(path)
        all_chunks.extend(chunker.chunk("\n".join(texts), source=path.name))

    if not all_chunks:
        logger.warning("No chunks produced from %s", data_dir)
        return

    embeddings = embedder.encode([c.content for c in all_chunks])
    store.add(all_chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save(index_dir)
    logger.info("Saved %d chunks to %s", len(all_chunks), index_dir)
