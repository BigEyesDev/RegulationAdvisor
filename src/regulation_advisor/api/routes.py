"""
API routes for RegulationAdvisor.

_agent is set once at startup by api/app.py lifespan.
Routes read it at request time — never at import time.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from regulation_advisor.api import metrics_store
from regulation_advisor.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluateResponse,
    HealthResponse,
    MetricsResponse,
    SourceReference,
)
from regulation_advisor.classifier.reg_classifier import RegClassifier
from regulation_advisor.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Set by api/app.py lifespan via set_agent()
_agent = None

# Lazily loads the fine-tuned checkpoint on first use, or falls back to the
# prompted LLM if CLASSIFIER_CHECKPOINT is unset — see config.py.
_classifier = RegClassifier(checkpoint_path=settings.classifier_checkpoint or None)


def set_agent(agent: object) -> None:
    global _agent
    _agent = agent


def set_classifier(classifier: object) -> None:
    """Test/startup injection point — mirrors set_agent()."""
    global _classifier
    _classifier = classifier


_BYOK_REQUIRED_DETAIL = (
    "This deployment has no default API key configured — add your own "
    "provider API key to use it."
)


def _agent_for_request(request: ChatRequest) -> object:
    """
    Return the agent to use for one request.

    A supplied ``api_key`` builds a brand-new agent scoped to this call only —
    nothing is cached, stored, or attached to ``request.session_id``, so the
    key exists only for the lifetime of this function call and the request
    it serves. With no ``api_key``, the shared default agent is used — but
    only if the deployment actually has a default key configured. A
    deployment run with empty LLM keys in its secrets is BYOK-only by
    design (no visitor's usage is ever billed to the deployer), so a
    keyless request there is rejected with a clear 400 instead of silently
    calling out with an empty key.
    """
    if request.api_key:
        from regulation_advisor.agent.graph import build_agent_graph

        return build_agent_graph(
            provider=request.provider, model=request.model, api_key=request.api_key
        )
    if not settings.has_default_llm_key:
        raise HTTPException(status_code=400, detail=_BYOK_REQUIRED_DETAIL)
    return _agent


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.6.12",
        vector_store_backend=settings.vector_store_backend,
    )


@router.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream the LLM response token-by-token as Server-Sent Events."""
    agent = _agent_for_request(request)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    async def generate():
        config = {"configurable": {"thread_id": request.session_id}}
        try:
            async for event in agent.astream_events(
                {"messages": [("human", request.message)]}, config=config, version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as exc:
            logger.warning("Chat stream failed: %s", type(exc).__name__)
            error = "The configured LLM provider rejected the request."
            yield f"data: {json.dumps({'type': 'error', 'content': error})}\n\n"
            return
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest) -> ChatResponse:
    """Non-streaming: returns the complete answer. Used by eval harness and tests."""
    agent = _agent_for_request(request)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    config = {"configurable": {"thread_id": request.session_id}}
    try:
        result = await agent.ainvoke(
            {"messages": [("human", request.message)]}, config=config
        )
    except Exception as exc:
        logger.warning("Chat request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="The configured LLM provider rejected the request."
        ) from None

    answer = result["messages"][-1].content
    retrieved = result.get("retrieved_chunks", [])
    sources = [
        SourceReference(article_number=c.article_number, source_document=c.source_document)
        for c in retrieved
    ]

    risk_tier, classifier_confidence = None, None
    try:
        finding = _classifier.classify(answer)
        risk_tier, classifier_confidence = finding.risk_tier, finding.confidence
    except Exception:
        logger.exception("RegClassifier failed — returning answer without a risk tier")

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence_score=result.get("confidence_score", 1.0),
        warnings=[],
        session_id=request.session_id,
        risk_tier=risk_tier,
        classifier_confidence=classifier_confidence,
    )


# ── Evaluation endpoints ──────────────────────────────────────────────────────

_evaluation_running = False


@router.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Return the most recent RAGAS evaluation scores from disk."""
    result = metrics_store.load()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No evaluation scores yet. POST /api/evaluate to run one.",
        )
    return result


@router.post("/api/evaluate", response_model=EvaluateResponse)
async def trigger_evaluation(background_tasks: BackgroundTasks) -> EvaluateResponse:
    """Start a RAGAS evaluation in the background. Returns immediately."""
    global _evaluation_running
    if _evaluation_running:
        return EvaluateResponse(
            status="already_running", message="Evaluation already in progress."
        )
    background_tasks.add_task(_run_evaluation)
    return EvaluateResponse(
        status="started", message="Evaluation started. Poll /api/metrics for results."
    )


async def _run_evaluation() -> None:
    global _evaluation_running
    _evaluation_running = True
    try:
        from pathlib import Path

        from regulation_advisor.evaluation.harness import EvaluationHarness

        harness = EvaluationHarness(Path("evals/qa_pairs.json"))

        def pipeline_fn(question: str) -> tuple[str, list[str]]:
            import asyncio
            config = {"configurable": {"thread_id": f"eval_{hash(question)}"}}
            result = asyncio.get_event_loop().run_until_complete(
                _agent.ainvoke({"messages": [("human", question)]}, config=config)
            )
            answer = result["messages"][-1].content
            contexts = [c.content for c in result.get("retrieved_chunks", [])]
            return answer, contexts

        scores = harness.run(pipeline_fn)
        metrics_store.save(
            faithfulness=scores.faithfulness,
            answer_relevancy=scores.answer_relevancy,
            context_precision=scores.context_precision,
            context_recall=scores.context_recall,
            total_qa_pairs=len(harness._qa_pairs),
        )
        logger.info("Evaluation complete: %s", scores.summary())
    except Exception:
        logger.exception("Evaluation failed")
    finally:
        _evaluation_running = False
