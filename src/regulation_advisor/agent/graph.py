"""LangGraph agent graph."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from regulation_advisor.agent.state import CRITICAL_KEYWORDS, RegAdvisorState
from regulation_advisor.agent.tools import query_structured_data, search_regulations, search_web
from regulation_advisor.llm import build_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_prompt.txt").read_text()

# Tool priority is communicated via each tool's docstring (what the LLM reads):
#   1. search_regulations      — always try first (authoritative regulation text)
#   2. query_structured_data   — timelines, penalties, risk tiers from CSVs
#   3. search_web              — last resort only for recent news not in the PDFs
TOOLS = [search_regulations, query_structured_data, search_web]


def build_agent_graph(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """
    Build and compile the LangGraph StateGraph.

    Called once at startup for the shared default agent. ``provider``/``model``/
    ``api_key`` let a caller build a second, independent agent for a single
    BYOK request (see api/routes.py) — no index loading happens here, just an
    LLM client and a graph compile, so this is cheap enough to call per request.
    The returned graph's checkpointer is private to this instance; nothing
    about a BYOK call is retained once the caller drops the reference.
    """
    llm = build_llm(provider=provider, model=model, api_key=api_key)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: RegAdvisorState) -> dict:
        non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + non_system)
        is_critical = any(kw.lower() in response.content.lower() for kw in CRITICAL_KEYWORDS)
        return {"messages": [response], "is_critical_finding": is_critical}

    def should_continue(state: RegAdvisorState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        if state.get("is_critical_finding"):
            return "human_review"
        return END

    graph = StateGraph(RegAdvisorState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("human_review", lambda s: s)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
