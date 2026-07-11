from regulation_advisor.ingestion.chunkers import ArticleAwareChunker, RecursiveCharacterChunker

SAMPLE = """
Article 5
Prohibited AI practices

1. The following shall be prohibited: social scoring by public authorities.

Article 6
Classification rules for high-risk AI systems

1. Irrespective of whether an AI system is placed on the market...
"""

def test_article_aware_chunker_finds_articles():
    chunks = ArticleAwareChunker().chunk(SAMPLE, source="test")
    assert "5" in [c.article_number for c in chunks]
    assert "6" in [c.article_number for c in chunks]

def test_article_aware_chunker_sets_source():
    chunks = ArticleAwareChunker().chunk(SAMPLE, source="eu_ai_act.pdf")
    assert all(c.source_document == "eu_ai_act.pdf" for c in chunks)

def test_recursive_chunker_produces_output():
    chunks = RecursiveCharacterChunker(size=200, overlap=20).chunk(SAMPLE, source="test")
    assert len(chunks) > 0

def test_article_aware_chunker_title_extracted():
    """Every chunk produced from well-formed article text must have a non-empty title."""
    chunks = ArticleAwareChunker().chunk(SAMPLE, source="test")
    assert len(chunks) > 0, "Expected at least one chunk"
    for chunk in chunks:
        assert chunk.article_title != "", (
            f"Article {chunk.article_number} has an empty title"
        )
