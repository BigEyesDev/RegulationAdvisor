"""Unit tests for regulation_advisor.llm.build_llm().

Run:
    uv run pytest tests/unit/test_llm.py -v
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from regulation_advisor.llm import build_llm


def test_default_uses_settings():
    llm = build_llm()
    assert isinstance(llm, (ChatOpenAI, ChatGoogleGenerativeAI, ChatGroq, ChatAnthropic))


def test_provider_override_returns_matching_client_type():
    llm = build_llm(provider="groq", model="llama-3.3-70b-versatile", api_key="fake-key")
    assert isinstance(llm, ChatGroq)


def test_openai_provider_returns_chat_openai():
    llm = build_llm(provider="openai", model="gpt-4o-mini", api_key="fake-key")
    assert isinstance(llm, ChatOpenAI)
    assert llm.openai_api_key.get_secret_value() == "fake-key"


def test_anthropic_provider_returns_chat_anthropic():
    llm = build_llm(provider="anthropic", model="claude-sonnet-5", api_key="fake-key")
    assert isinstance(llm, ChatAnthropic)
    assert llm.anthropic_api_key.get_secret_value() == "fake-key"


def test_api_key_override_is_used_not_settings_key():
    llm = build_llm(provider="groq", model="llama-3.3-70b-versatile", api_key="byok-fake-key")
    assert llm.groq_api_key.get_secret_value() == "byok-fake-key"


def test_model_override_is_used():
    llm = build_llm(provider="groq", model="some-other-model", api_key="fake-key")
    assert llm.model_name == "some-other-model"


def test_no_override_does_not_mutate_settings():
    from regulation_advisor.config import settings

    original_provider = settings.llm_provider
    build_llm(provider="google", model="gemini-2.5-flash", api_key="fake-key")
    assert settings.llm_provider == original_provider
