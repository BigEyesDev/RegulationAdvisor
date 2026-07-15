"""
Adapter between promptfoo and the LangGraph agent.

promptfoo calls run_query(prompt, options, context) for each test case.
We extract the question from context["vars"]["question"] and pass it
through the compiled agent, returning the final AI message as a string.

This module is imported once per promptfoo run — the agent is built
at module load time to avoid rebuilding it for every test case.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regulation_advisor.agent.graph import build_agent_graph

logging.basicConfig(level=logging.WARNING)  # suppress INFO spam during eval runs

_agent = build_agent_graph()


def run_query(prompt: str, options: dict, context: dict) -> str:
    question = context.get("vars", {}).get("question", prompt)
    config = {"configurable": {"thread_id": f"promptfoo-{hash(question)}"}}
    result = _agent.invoke({"messages": [("human", question)]}, config=config)
    return result["messages"][-1].content
