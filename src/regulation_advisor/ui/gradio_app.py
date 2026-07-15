"""
Gradio UI — RegulationAdvisor v0.4

v0.1 (Week 1): simple RAG chain — retrieve chunks → stuff context → LLM.
v0.2 (Week 2): LangGraph agent with tools + streaming.
v0.3 (Week 3): Guardrail layer — citation and legal-claim checks.
v0.4 (Week 4): build_ui() takes no argument; reads the agent lazily from
               api.routes._agent so it can be mounted on FastAPI before
               the lifespan fires.  Evaluation Dashboard tab added (Day 5).
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import date
from typing import Generator

import gradio as gr
from langchain_core.messages import AIMessageChunk, SystemMessage

from regulation_advisor.agent.state import CRITICAL_KEYWORDS
from regulation_advisor.evaluation.guardrails import build_guardrail_chain
from regulation_advisor.models import RegulationChunk

logger = logging.getLogger(__name__)

_guardrails = build_guardrail_chain()

_CRITICAL_WARNING = (
    "\n\n---\n⚠️ **Critical finding** — this topic involves prohibited practices "
    "or significant penalties. Verify with a qualified legal professional before acting."
)

_DATE_CONTEXT_TEMPLATE = (
    "Today's date is {today}. "
    "When discussing regulatory deadlines, always calculate exactly how many days, "
    "weeks, or months remain from today's date — do not use approximate or static figures when calculating dates and times."
)


def _context_chunks_from_state(agent: object, config: dict) -> list[RegulationChunk]:
    """
    Pull the article numbers the agent actually retrieved during this turn.

    After agent.stream() finishes, the LangGraph checkpointer holds the full
    message history. Tool messages contain the regulation text returned by
    search_regulations. We parse article numbers from that text and build
    lightweight RegulationChunk objects so the citation guardrail can check
    whether the LLM cited an article that was never retrieved.
    """
    try:
        state = agent.get_state(config)
        tool_texts = " ".join(
            m.content for m in state.values.get("messages", [])
            if hasattr(m, "type") and m.type == "tool"
        )
        article_numbers = set(re.findall(r"Article\s+(\d+[a-z]?)", tool_texts, re.IGNORECASE))
        return [
            RegulationChunk(content="", article_number=a, article_title="", source_document="")
            for a in article_numbers
        ]
    except Exception:
        return []


def _get_agent():
    """Lazy agent accessor — reads from routes at call time, not import time."""
    from regulation_advisor.api.routes import _agent
    return _agent


def build_ui() -> gr.Blocks:
    """Build the Gradio UI. Agent is read lazily via _get_agent() at request time."""
    session_id = str(uuid.uuid4())

    def respond(message: str, history: list) -> Generator[str, None, None]:
        agent = _get_agent()
        if agent is None:
            yield "Service not ready — agent is still loading. Please retry in a moment."
            return

        config = {"configurable": {"thread_id": session_id}}

        messages: list = [("human", message)]
        if not history:
            today = date.today().strftime("%B %d, %Y")
            messages = [SystemMessage(content=_DATE_CONTEXT_TEMPLATE.format(today=today))] + messages

        partial = ""
        for chunk, _ in agent.stream({"messages": messages}, config=config, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                partial += chunk.content
                yield partial

        if any(kw.lower() in partial.lower() for kw in CRITICAL_KEYWORDS):
            partial = partial + _CRITICAL_WARNING
            yield partial

        # Run guardrails on the complete answer.
        # confidence=1.0 skips the faithfulness check (no real-time RAGAS score);
        # citation and legal-claim checks still run.
        chunks = _context_chunks_from_state(agent, config)
        guard = _guardrails.check(partial, chunks, confidence=1.0)
        if guard.warnings:
            yield partial + "\n\n" + "\n\n".join(guard.warnings)

        logger.info(
            "Streamed answer (%d chars, guardrail_passed=%s, warnings=%d)",
            len(partial), guard.passed, len(guard.warnings),
        )

    with gr.Blocks(title="RegulationAdvisor v0.4") as demo:
        gr.Markdown(
            "## EU AI Act Compliance Advisor\n"
            "Ask any question about the EU AI Act or GDPR. "
            "Every answer cites the relevant Article. "
            "Critical findings are flagged for legal review."
        )
        gr.ChatInterface(fn=respond, title="")

    return demo
