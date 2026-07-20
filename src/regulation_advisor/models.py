"""
Shared Pydantic data models used across the codebase.
Add new models here — do not define data models inside individual modules.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RegulationChunk(BaseModel):
    """A single chunk of regulation text with metadata."""

    content: str
    article_number: str
    article_title: str
    source_document: str
    page_number: int | None = None


class RegulationFinding(BaseModel):
    """Classified output from the RegClassifier."""

    article: str
    risk_tier: Literal["Unacceptable", "High", "Limited", "Minimal"]
    obligation_type: Literal[
        "PROHIBITED", "TRANSPARENCY", "CONFORMITY", "REGISTRATION", "GENERAL_PURPOSE"
    ]
    urgency: Literal["IMMEDIATE", "2025", "2026", "2027"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class RetrievalResult(BaseModel):
    """Output of a retrieval operation."""

    chunks: list[RegulationChunk]
    query: str
    scores: list[float] = Field(default_factory=list)
