import pytest
from pathlib import Path

INDEX_DIR = Path("data/index")
_skip_if_no_index = pytest.mark.skipif(
    not INDEX_DIR.exists() or not (INDEX_DIR / "index.faiss").exists(),
    reason="FAISS index not built yet — run: uv run python scripts/ingest.py",
)


@_skip_if_no_index
def test_article_5_retrieved_for_prohibited_query():
    """Article 5 (Prohibited AI practices) must appear in top-5 results."""
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.store import FAISSVectorStore
    from regulation_advisor.retrieval.retriever import Retriever

    store = FAISSVectorStore()
    store.load(INDEX_DIR)
    result = Retriever(store, SentenceTransformerEmbedder()).search(
        "What AI practices are completely prohibited?"
    )
    assert "5" in [c.article_number for c in result.chunks]


@_skip_if_no_index
def test_faiss_save_load_round_trip():
    """Load the persisted index, run a query, verify results are non-empty."""
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.store import FAISSVectorStore

    store = FAISSVectorStore()
    store.load(INDEX_DIR)

    embedder = SentenceTransformerEmbedder()
    query_vec = embedder.encode(["What is a high-risk AI system?"])[0]
    chunks = store.search(query_vec, k=5)

    assert len(chunks) > 0, "Search on loaded index returned no results"
    assert all(c.content for c in chunks), "Some chunks have empty content"
    assert all(c.source_document for c in chunks), "Some chunks have empty source"
