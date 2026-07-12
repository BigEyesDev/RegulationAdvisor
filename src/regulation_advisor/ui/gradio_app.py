"""
Gradio UI — RegulationAdvisor v0.2

v0.1 (Week 1): simple RAG chain — retrieve chunks → stuff context → LLM.
v0.2 (Week 2): LangGraph agent — the agent decides when to call tools and
               how many times, and surfaces a warning on critical findings.
               Streaming: tokens are yielded as they arrive via agent.stream().
"""
from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Generator

import gradio as gr
from langchain_core.messages import AIMessageChunk, SystemMessage

from regulation_advisor.agent.state import CRITICAL_KEYWORDS

logger = logging.getLogger(__name__)

_CRITICAL_WARNING = (
    "\n\n---\n⚠️ **Critical finding** — this topic involves prohibited practices "
    "or significant penalties. Verify with a qualified legal professional before acting."
)

_DATE_CONTEXT_TEMPLATE = (
    "Today's date is {today}. "
    "When discussing regulatory deadlines, always calculate exactly how many days, "
    "weeks, or months remain from today's date — do not use approximate or static figures when calculating dates and times."
)


def build_ui(agent) -> gr.Blocks:
    """
    Build the Gradio ChatInterface around a compiled LangGraph agent.

    respond() is a generator — it yields partial answer strings as tokens
    arrive from the LLM, giving the user a live typing effect. Gradio's
    ChatInterface handles generator functions natively.

    A fresh session_id (UUID) is created per server start so each deployment
    gets its own isolated conversation thread in MemorySaver.

    Args:
        agent: A compiled LangGraph graph (returned by build_agent_graph()).

    Returns:
        A gr.Blocks object ready for demo.launch().
    """
    session_id = str(uuid.uuid4())

    def respond(message: str, history: list) -> Generator[str, None, None]:
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
            yield partial + _CRITICAL_WARNING

        logger.info("Streamed answer (%d chars, critical=%s)", len(partial), bool(partial) and any(
            kw.lower() in partial.lower() for kw in CRITICAL_KEYWORDS
        ))

    with gr.Blocks(title="RegulationAdvisor v0.2") as demo:
        gr.Markdown(
            "## EU AI Act Compliance Advisor\n"
            "Ask any question about the EU AI Act or GDPR. "
            "Every answer cites the relevant Article. "
            "Critical findings are flagged for legal review."
        )
        gr.ChatInterface(fn=respond, title="")

    return demo
