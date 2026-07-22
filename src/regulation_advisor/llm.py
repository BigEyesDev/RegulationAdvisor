"""
Shared LLM factory — used by both the Gradio chain and the LangGraph agent.

To switch models edit two lines in .env:
    LLM_PROVIDER=openrouter   (or groq / google / openai / anthropic)
    LLM_MODEL=deepseek/deepseek-v4-flash

Supported providers and example model slugs:
    openrouter  →  deepseek/deepseek-v4-flash
                   deepseek/deepseek-v4-pro
                   qwen/qwen3-32b
                   moonshotai/kimi-k2
    groq        →  llama-3.3-70b-versatile
                   qwen/qwen3-32b
    google      →  gemini-2.5-flash
                   gemini-2.5-pro
    openai      →  gpt-4o-mini
                   gpt-4o
    anthropic   →  claude-sonnet-5
                   claude-haiku-4-5-20251001
"""
from __future__ import annotations

import logging

from regulation_advisor.config import settings

logger = logging.getLogger(__name__)


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
):
    """
    LLM provider factory — reads LLM_PROVIDER from config/environment by default.

    ``provider``/``model``/``api_key`` override the corresponding settings value
    for this call only; global settings are never mutated. This is the BYOK
    entry point — callers pass a caller-supplied ``api_key`` to build a
    throwaway client for one request without touching the shared default LLM.

    Returns a LangChain chat model object that is compatible with
    ``llm.bind_tools()``, ``llm.invoke()``, and LangChain chains.
    """
    provider = provider or settings.llm_provider
    model = model or settings.llm_model
    logger.info("Building LLM: provider=%s model=%s", provider, model)

    timeout = settings.llm_request_timeout_seconds

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url=settings.openrouter_base_url,
            api_key=api_key or settings.openrouter_api_key,
            timeout=timeout,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key or settings.google_api_key, timeout=timeout
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model, api_key=api_key or settings.openai_api_key, timeout=timeout
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model, api_key=api_key or settings.anthropic_api_key, timeout=timeout
        )
    # default: groq
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=api_key or settings.groq_api_key, timeout=timeout)
