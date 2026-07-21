"""Unit tests for the Gradio BYOK routing helper (_agent_for_call)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from regulation_advisor.config import settings
from regulation_advisor.ui.gradio_app import _agent_for_call, _ByokRequiredError


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
        mock_build.assert_called_once_with(api_key="sk-user-supplied")


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
