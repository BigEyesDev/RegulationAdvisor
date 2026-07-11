"""
Vector store — Repository pattern.
Week 1-3: FAISSVectorStore (in-memory, fast)
Week 4+:  ChromaDBVectorStore (persistent, production-ready)

Swap by changing VECTOR_STORE_BACKEND in .env — no other code changes needed.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from regulation_advisor.models import RegulationChunk

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None: ...
    def search(self, query_embedding: np.ndarray, k: int) -> list[RegulationChunk]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


class FAISSVectorStore:
    """In-memory. Fast for development. Does not survive restarts."""

    def __init__(self, dimension: int = 384) -> None:
        import faiss
        self._index = faiss.IndexFlatL2(dimension)
        self._chunks: list[RegulationChunk] = []

    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None:
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)
        logger.info("Added %d chunks to FAISS index", len(chunks))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RegulationChunk]:
        _, indices = self._index.search(query_embedding.reshape(1, -1).astype(np.float32), k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    def save(self, path: Path) -> None:
        import faiss
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self, path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)


class ChromaDBVectorStore:
    """Persistent. Survives restarts. Use from Week 4 onward."""

    def __init__(self, host: str = "localhost", port: int = 8001) -> None:
        import chromadb
        self._client = chromadb.HttpClient(host=host, port=port)
        self._col = self._client.get_or_create_collection("regulations",
                                                           metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[RegulationChunk], embeddings: np.ndarray) -> None:
        self._col.add(
            ids=[f"{c.source_document}_{c.article_number}_{i}" for i, c in enumerate(chunks)],
            embeddings=embeddings.tolist(),
            documents=[c.content for c in chunks],
            metadatas=[{"article": c.article_number, "source": c.source_document} for c in chunks],
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RegulationChunk]:
        results = self._col.query(query_embeddings=[query_embedding.tolist()], n_results=k)
        return [
            RegulationChunk(content=doc, article_number=m["article"],
                            article_title="", source_document=m["source"])
            for doc, m in zip(results["documents"][0], results["metadatas"][0])
        ]

    def save(self, path: Path) -> None:
        pass  # ChromaDB persists automatically

    def load(self, path: Path) -> None:
        pass  # ChromaDB loads automatically


def build_vector_store() -> VectorStore:
    """Factory — reads VECTOR_STORE_BACKEND from config."""
    from regulation_advisor.config import settings
    if settings.vector_store_backend == "chromadb":
        return ChromaDBVectorStore(host=settings.chroma_host, port=settings.chroma_port)
    return FAISSVectorStore()
