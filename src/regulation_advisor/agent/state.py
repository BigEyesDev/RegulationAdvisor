"""LangGraph state definition. All agent state lives here."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from regulation_advisor.models import RegulationChunk

CRITICAL_KEYWORDS = ["prohibited", "banned", "Article 5", "35,000,000", "7%", "illegal"]


class RegAdvisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_chunks: list[RegulationChunk]
    tools_used: list[str]
    confidence_score: float
    is_critical_finding: bool
