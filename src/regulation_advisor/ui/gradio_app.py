"""
Gradio UI — RegulationAdvisor v0.2

v0.1 (Week 1): simple RAG chain — retrieve chunks → stuff context → LLM.
v0.2 (Week 2): LangGraph agent — the agent decides when to call tools and
               how many times, and surfaces a warning on critical findings.
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)

_CRITICAL_WARNING = (
    "\n\n---\n⚠️ **Critical finding** — this topic involves prohibited practices "
    "or significant penalties. Verify with a qualified legal professional before acting."
)


def build_ui(agent) -> gr.Blocks:
    """
    Build the Gradio ChatInterface around a compiled LangGraph agent.

    Args:
        agent: A compiled LangGraph graph (returned by build_agent_graph()).

    Returns:
        A gr.Blocks object ready for demo.launch().
    """

    def respond(message: str, history: list) -> str:
        config = {"configurable": {"thread_id": "gradio-session"}}
        result = agent.invoke({"messages": [("human", message)]}, config=config)
        answer = result["messages"][-1].content
        if result.get("is_critical_finding"):
            answer += _CRITICAL_WARNING
        logger.info("Answered query (%d chars, critical=%s)", len(answer), result.get("is_critical_finding"))
        return answer

    with gr.Blocks(title="RegulationAdvisor v0.2") as demo:
        gr.Markdown(
            "## EU AI Act Compliance Advisor\n"
            "Ask any question about the EU AI Act or GDPR. "
            "Every answer cites the relevant Article. "
            "Critical findings are flagged for legal review."
        )
        gr.ChatInterface(fn=respond, title="")

    return demo
