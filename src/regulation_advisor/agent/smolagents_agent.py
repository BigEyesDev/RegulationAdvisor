"""
smolagents comparison agent.

Implements the same 3-tool agent as the LangGraph version so we can run
identical benchmark queries against both and compare framework behaviour.

The LangChain tools from tools.py are reused directly via LangChainTool —
no code duplication of tool logic.

Usage:
    from regulation_advisor.agent.smolagents_agent import build_smolagents_agent
    agent = build_smolagents_agent()
    result = agent.run("What are the prohibited AI practices under Article 5?")
"""
from __future__ import annotations

import logging

from smolagents import LangChainTool, LiteLLMModel, ToolCallingAgent

from regulation_advisor.agent.tools import query_structured_data, search_regulations, search_web
from regulation_advisor.config import settings

logger = logging.getLogger(__name__)

_LITELLM_PREFIXES = {
    "openrouter": "openrouter",
    "google": "gemini",
    "groq": "groq",
}


def _litellm_model_id() -> str:
    """Map LLM_PROVIDER / LLM_MODEL from settings to a LiteLLM model identifier."""
    prefix = _LITELLM_PREFIXES.get(settings.llm_provider, settings.llm_provider)
    return f"{prefix}/{settings.llm_model}"


def build_smolagents_agent() -> ToolCallingAgent:
    """
    Build a smolagents ToolCallingAgent using the same 3 tools as the LangGraph agent.

    Returns:
        A ToolCallingAgent ready to accept .run("your question") calls.
    """
    model = LiteLLMModel(model_id=_litellm_model_id())
    tools = [
        LangChainTool(search_regulations),
        LangChainTool(query_structured_data),
        LangChainTool(search_web),
    ]
    logger.info("Building smolagents agent: model=%s tools=%d", model.model_id, len(tools))
    return ToolCallingAgent(tools=tools, model=model, max_steps=5)
