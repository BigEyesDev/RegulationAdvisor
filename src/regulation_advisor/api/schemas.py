"""FastAPI request/response models — built in Week 4."""
from __future__ import annotations
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence_score: float
    warnings: list[str]


class MetricsResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    last_evaluated_at: str
