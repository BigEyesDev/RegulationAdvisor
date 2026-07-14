"""Tests for the guardrail wiring in gradio_app — no LLM, no streaming."""
import re
from regulation_advisor.models import RegulationChunk


def _parse_articles_from_tool_text(tool_text: str) -> list[RegulationChunk]:
    """Same logic as _context_chunks_from_state, extracted for unit testing."""
    article_numbers = set(re.findall(r"Article\s+(\d+[a-z]?)", tool_text, re.IGNORECASE))
    return [
        RegulationChunk(content="", article_number=a, article_title="", source_document="")
        for a in article_numbers
    ]


def test_article_numbers_parsed_from_tool_text():
    tool_text = "[Article 5 — eu_ai_act.pdf]\nProhibited practices..."
    chunks = _parse_articles_from_tool_text(tool_text)
    assert any(c.article_number == "5" for c in chunks)


def test_multiple_articles_parsed():
    tool_text = "Article 5 and Article 99 are relevant here."
    chunks = _parse_articles_from_tool_text(tool_text)
    numbers = {c.article_number for c in chunks}
    assert numbers == {"5", "99"}


def test_no_articles_returns_empty():
    chunks = _parse_articles_from_tool_text("No regulation text here.")
    assert chunks == []


def test_guardrail_passes_when_cited_article_in_context():
    from regulation_advisor.evaluation.guardrails import build_guardrail_chain
    chain = build_guardrail_chain()
    chunks = _parse_articles_from_tool_text("Article 5 — prohibited practices...")
    result = chain.check("Under Article 5, social scoring is prohibited.", chunks, confidence=1.0)
    assert result.passed


def test_guardrail_blocks_when_cited_article_not_in_context():
    from regulation_advisor.evaluation.guardrails import build_guardrail_chain
    chain = build_guardrail_chain()
    chunks = _parse_articles_from_tool_text("Article 5 — prohibited practices...")
    # LLM cites Article 99 but it was never retrieved
    result = chain.check("Under Article 99, the fine is EUR 35M.", chunks, confidence=1.0)
    assert not result.passed
