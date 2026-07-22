"""FastAPI request/response models for RegulationAdvisor v0.4."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=100)
    # BYOK — set api_key to use your own provider account for this request
    # only. Never persisted: read once to build a throwaway LLM client, then
    # discarded when the request finishes. See api/routes.py chat_sync/chat_stream.
    api_key: str | None = Field(default=None, max_length=200, repr=False)
    provider: Literal["openrouter", "groq", "google", "openai", "anthropic"] | None = None
    model: str | None = Field(default=None, max_length=200)


class SourceReference(BaseModel):
    article_number: str
    source_document: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence_score: float = Field(ge=0.0, le=1.0)
    warnings: list[str]
    session_id: str
    risk_tier: str | None = None
    classifier_confidence: float | None = None


class MetricsResponse(BaseModel):
    version: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    total_qa_pairs: int
    acceptable: bool
    evaluated_at: str


class EvaluateResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_backend: str
