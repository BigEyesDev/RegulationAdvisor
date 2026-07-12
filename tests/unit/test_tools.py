"""Unit tests for regulation_advisor.agent.tools.

Run:
    uv run pytest tests/unit/test_tools.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_chunk(article_number: str, source: str, content: str):
    """Build a minimal RegulationChunk-like object for mocking."""
    chunk = MagicMock()
    chunk.article_number = article_number
    chunk.source_document = source
    chunk.content = content
    return chunk


def _make_retriever_result(*chunks):
    result = MagicMock()
    result.chunks = list(chunks)
    return result


# ---------------------------------------------------------------------------
# Test 1: search_regulations with a working retriever returns article text
# ---------------------------------------------------------------------------


def test_search_regulations_returns_article():
    """
    search_regulations should call retriever.search() and return a formatted
    string that contains the article number and content.
    """
    from regulation_advisor.agent import tools as tools_module

    fake_chunk = _make_chunk("5", "eu_ai_act.pdf", "Prohibited practices include…")
    fake_retriever = MagicMock()
    fake_retriever.search.return_value = _make_retriever_result(fake_chunk)

    # Inject mock retriever
    original = tools_module._retriever
    tools_module._retriever = fake_retriever
    try:
        result = tools_module.search_regulations.invoke({"query": "prohibited AI"})
    finally:
        tools_module._retriever = original

    assert "Article 5" in result
    assert "Prohibited practices" in result
    fake_retriever.search.assert_called_once_with("prohibited AI", k=5)


# ---------------------------------------------------------------------------
# Test 2: query_structured_data returns matching rows from real CSVs
# ---------------------------------------------------------------------------


def test_query_structured_data_returns_rows():
    """
    query_structured_data should read the real CSV files from data/ and return
    rows that match the keyword 'prohibited'.
    The risk_classification.csv has an 'Unacceptable' tier row mentioning prohibited.
    """
    result = (
        __import__(
            "regulation_advisor.agent.tools",
            fromlist=["query_structured_data"],
        )
        .query_structured_data.invoke({"question": "prohibited"})
    )

    assert result != "No matching structured data found."
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Test 3: search_regulations without a retriever returns an error string (no crash)
# ---------------------------------------------------------------------------


def test_search_regulations_without_retriever_returns_error_string():
    """
    When no retriever has been set, search_regulations must NOT raise an
    exception — it should return a human-readable error string instead.
    This protects the agent from crashing on cold starts.
    """
    from regulation_advisor.agent import tools as tools_module

    original = tools_module._retriever
    tools_module._retriever = None
    try:
        result = tools_module.search_regulations.invoke({"query": "anything"})
    finally:
        tools_module._retriever = original

    assert isinstance(result, str)
    assert "Error" in result or "retriever" in result.lower()
