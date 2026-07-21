"""Unit tests for the Gradio BYOK routing helper (_agent_for_call)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from regulation_advisor.config import settings
from regulation_advisor.ui.gradio_app import _BYOK_MODELS, _agent_for_call, _ByokRequiredError


def test_no_key_returns_shared_agent():
    shared = MagicMock(name="shared-agent")
    with patch("regulation_advisor.ui.gradio_app._get_agent", return_value=shared):
        assert _agent_for_call(None) is shared
        assert _agent_for_call("") is shared


def test_key_builds_throwaway_agent_not_shared():
    shared = MagicMock(name="shared-agent")
    throwaway = MagicMock(name="throwaway-agent")
    with (
        patch("regulation_advisor.ui.gradio_app._get_agent", return_value=shared),
        patch(
            "regulation_advisor.agent.graph.build_agent_graph", return_value=throwaway
        ) as mock_build,
    ):
        result = _agent_for_call("sk-user-supplied")
        assert result is throwaway
        assert result is not shared
        mock_build.assert_called_once_with(provider=None, model=None, api_key="sk-user-supplied")


def test_key_with_provider_and_model_passes_them_through():
    throwaway = MagicMock(name="throwaway-agent")
    with patch(
        "regulation_advisor.agent.graph.build_agent_graph", return_value=throwaway
    ) as mock_build:
        result = _agent_for_call(
            "sk-user-supplied", provider="groq", model="llama-3.3-70b-versatile"
        )
        assert result is throwaway
        mock_build.assert_called_once_with(
            provider="groq", model="llama-3.3-70b-versatile", api_key="sk-user-supplied"
        )


def test_no_key_and_no_default_key_raises_byok_required(monkeypatch):
    monkeypatch.setattr(type(settings), "has_default_llm_key", False)
    with pytest.raises(_ByokRequiredError):
        _agent_for_call(None)


def test_supplied_key_still_works_when_no_default_key(monkeypatch):
    monkeypatch.setattr(type(settings), "has_default_llm_key", False)
    throwaway = MagicMock(name="throwaway-agent")
    with patch(
        "regulation_advisor.agent.graph.build_agent_graph", return_value=throwaway
    ):
        assert _agent_for_call("sk-user-supplied") is throwaway


def test_byok_models_cover_every_schema_provider():
    from regulation_advisor.api.schemas import ChatRequest

    schema_providers = set(
        ChatRequest.model_fields["provider"].annotation.__args__[0].__args__
    )
    assert set(_BYOK_MODELS.keys()) == schema_providers
    for models in _BYOK_MODELS.values():
        assert models, "every provider must offer at least one model choice"
