"""
Agent tools — RAG search and CSV query.

Active tools (registered in graph.py TOOLS list):
  - search_regulations      semantic search over FAISS index (EU AI Act + GDPR PDFs)
  - query_structured_data   keyword search over CSV files (timelines, penalties, risk tiers)

Inactive tools (defined here but NOT in TOOLS — not exposed to the LLM):
  - search_web              Tavily internet search — excluded by design. This agent
                            answers only from authoritative regulation texts. Internet
                            sources are not authoritative for legal compliance answers.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Set via set_retriever() before running the agent
_retriever = None


def set_retriever(r: object) -> None:
    global _retriever
    _retriever = r


@tool
def search_regulations(query: str) -> str:
    """Search EU AI Act and GDPR documents for relevant articles and provisions."""
    if _retriever is None:
        return "Error: retriever not initialised. Call set_retriever() first."
    result = _retriever.search(query, k=5)
    return "\n\n---\n\n".join(
        f"[Article {c.article_number} — {c.source_document}]\n{c.content}"
        for c in result.chunks
    )


@tool
def query_structured_data(question: str) -> str:
    """Query structured regulation data: timelines, penalties, risk classifications."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    results: list[str] = []
    keywords = question.lower().split()

    for csv_file in ["ai_act_timeline.csv", "risk_classification.csv", "penalty_structure.csv"]:
        path = data_dir / csv_file
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
                if any(kw in row_text.lower() for kw in keywords):
                    results.append(row_text)

    return "\n".join(results) if results else "No matching structured data found."


@tool
def search_web(query: str) -> str:
    """Search the web for recent EU AI Act enforcement news and official guidance."""
    from tavily import TavilyClient
    from regulation_advisor.config import settings
    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(query, max_results=3)
    return "\n\n".join(r["content"] for r in results.get("results", []))
