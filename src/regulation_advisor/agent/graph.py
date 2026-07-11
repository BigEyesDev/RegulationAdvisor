"""LangGraph agent — built in Week 2."""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from regulation_advisor.agent.state import RegAdvisorState, CRITICAL_KEYWORDS
from regulation_advisor.agent.tools import search_regulations, query_structured_data, search_web

logger = logging.getLogger(__name__)

TOOLS = [search_regulations, query_structured_data, search_web]


def build_agent_graph():
    """Build and compile the LangGraph StateGraph. Called once at startup."""
    from langchain_groq import ChatGroq
    from regulation_advisor.config import settings

    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: RegAdvisorState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
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
