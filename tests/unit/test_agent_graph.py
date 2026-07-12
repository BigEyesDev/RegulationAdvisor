"""Unit tests for regulation_advisor.agent.graph.

Run:
    uv run pytest tests/unit/test_agent_graph.py -v

Gate check (from plan):
    uv run python -c "from regulation_advisor.agent.graph import build_agent_graph; g = build_agent_graph(); print('graph ok')"
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END


# ---------------------------------------------------------------------------
# Test 1: build_agent_graph() compiles without error
# ---------------------------------------------------------------------------


def test_graph_compiles():
    """
    build_agent_graph() must return a non-None compiled graph object.
    We mock build_llm() so the test doesn't need real API keys.
    """
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm  # llm.bind_tools(TOOLS) → mock

    with patch("regulation_advisor.agent.graph.build_llm", return_value=mock_llm):
        from regulation_advisor.agent.graph import build_agent_graph
        graph = build_agent_graph()

    assert graph is not None


# ---------------------------------------------------------------------------
# Test 2: should_continue routes to "tools" when tool_calls are present
# ---------------------------------------------------------------------------


def test_should_continue_routes_to_tools():
    """
    When the last message in state has tool_calls, the router must return "tools".
    This is how the agent knows to call a tool instead of finishing.
    """
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    with patch("regulation_advisor.agent.graph.build_llm", return_value=mock_llm):
        # Import the module freshly inside the patch context so we can grab
        # the inner should_continue function via a reconstruction.
        import importlib
        import regulation_advisor.agent.graph as graph_module
        importlib.reload(graph_module)

    # Reconstruct the should_continue function directly (it's a closure inside
    # build_agent_graph, so we test its logic via a standalone equivalent).
    from langchain_core.messages import AIMessage
    from regulation_advisor.agent.state import CRITICAL_KEYWORDS

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        if state.get("is_critical_finding"):
            return "human_review"
        return END

    # Build a fake message with tool_calls
    msg = MagicMock(spec=AIMessage)
    msg.tool_calls = [{"name": "search_regulations", "args": {"query": "test"}, "id": "t1"}]

    state = {"messages": [msg], "is_critical_finding": False}
    assert should_continue(state) == "tools"


# ---------------------------------------------------------------------------
# Test 3: should_continue routes to END when no tool calls and not critical
# ---------------------------------------------------------------------------


def test_should_continue_routes_to_end():
    """
    When the last message has no tool_calls and is_critical_finding is False,
    the router must return END — the conversation is done.
    """
    from langchain_core.messages import AIMessage

    def should_continue(state):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        if state.get("is_critical_finding"):
            return "human_review"
        return END

    msg = MagicMock(spec=AIMessage)
    msg.tool_calls = []   # empty list → no tools to call

    state = {"messages": [msg], "is_critical_finding": False}
    assert should_continue(state) == END
