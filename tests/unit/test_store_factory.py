"""
Unit tests for the vector store factory (build_vector_store).

Tests that the factory returns the right class based on config, without
requiring ChromaDB or a real index to be present.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from regulation_advisor.retrieval.store import FAISSVectorStore, build_vector_store
from regulation_advisor.config import Settings


def test_factory_returns_faiss_by_default():
    assert Settings.model_fields["vector_store_backend"].default == "faiss"
    store = build_vector_store()
    assert isinstance(store, FAISSVectorStore)


def test_factory_returns_chromadb_when_configured():
    from regulation_advisor.retrieval.store import ChromaDBVectorStore

    with patch("chromadb.HttpClient") as mock_client:
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_col

        with patch("regulation_advisor.config.settings") as mock_settings:
            mock_settings.vector_store_backend = "chromadb"
            mock_settings.chroma_host = "localhost"
            mock_settings.chroma_port = 8001
            store = build_vector_store()

    assert isinstance(store, ChromaDBVectorStore)


def test_faiss_save_and_load_roundtrip(tmp_path: Path):
    """FAISS index survives a save/load cycle with the same vectors."""
    store = FAISSVectorStore(dimension=4)

    from regulation_advisor.models import RegulationChunk
    chunk = RegulationChunk(
        content="Article 5 prohibits social scoring",
        article_number="5",
        article_title="Prohibited practices",
        source_document="eu_ai_act.pdf",
    )
    embeddings = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    store.add([chunk], embeddings)
    store.save(tmp_path)

    store2 = FAISSVectorStore(dimension=4)
    store2.load(tmp_path)
    results = store2.search(embeddings[0], k=1)

    assert len(results) == 1
    assert results[0].article_number == "5"


def test_index_dir_has_config_default():
    assert Settings.model_fields["index_dir"].default == "data/index"
