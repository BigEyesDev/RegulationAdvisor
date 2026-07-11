import pytest
from pathlib import Path

@pytest.mark.skipif(not Path("data/eu_ai_act.pdf").exists(),
                    reason="PDF not downloaded yet")
def test_article_5_retrieved_for_prohibited_query():
    from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
    from regulation_advisor.retrieval.store import FAISSVectorStore
    from regulation_advisor.retrieval.retriever import Retriever
    store = FAISSVectorStore()
    store.load(Path("data/index"))
    result = Retriever(store, SentenceTransformerEmbedder()).search(
        "What AI practices are completely prohibited?"
    )
    assert "5" in [c.article_number for c in result.chunks]
