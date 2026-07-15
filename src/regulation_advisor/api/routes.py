"""
API routes for RegulationAdvisor.

_agent is set once at startup by api/app.py lifespan.
Routes read it at request time — never at import time.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from regulation_advisor.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SourceReference,
)
from regulation_advisor.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Set by api/app.py lifespan via set_agent()
_agent = None


def set_agent(agent: object) -> None:
    global _agent
    _agent = agent


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.4.0",
        vector_store_backend=settings.vector_store_backend,
    )


@router.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream the LLM response token-by-token as Server-Sent Events."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    async def generate():
        config = {"configurable": {"thread_id": request.session_id}}
        async for event in _agent.astream_events(
            {"messages": [("human", request.message)]}, config=config, version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """Non-streaming: returns the complete answer. Used by eval harness and tests."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": request.session_id}}
    result = await _agent.ainvoke(
        {"messages": [("human", request.message)]}, config=config
    )

    answer = result["messages"][-1].content
    retrieved = result.get("retrieved_chunks", [])
    sources = [
        SourceReference(article_number=c.article_number, source_document=c.source_document)
        for c in retrieved
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence_score=result.get("confidence_score", 1.0),
        warnings=[],
        session_id=request.session_id,
    )
