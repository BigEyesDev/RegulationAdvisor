"""Unit tests for the Gradio BYOK routing helper (_agent_for_call)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from regulation_advisor.ui.gradio_app import _agent_for_call


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
