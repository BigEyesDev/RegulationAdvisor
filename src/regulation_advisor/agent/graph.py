"""LangGraph agent graph."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from regulation_advisor.agent.state import RegAdvisorState, CRITICAL_KEYWORDS
from regulation_advisor.agent.tools import search_regulations, query_structured_data, search_web
from regulation_advisor.llm import build_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system_prompt.txt").read_text()

# Tool priority is communicated via each tool's docstring (what the LLM reads):
#   1. search_regulations      — always try first (authoritative regulation text)
#   2. query_structured_data   — timelines, penalties, risk tiers from CSVs
#   3. search_web              — last resort only for recent news not in the PDFs
TOOLS = [search_regulations, query_structured_data, search_web]


def build_agent_graph():
    """Build and compile the LangGraph StateGraph. Called once at startup."""
    llm = build_llm()
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
