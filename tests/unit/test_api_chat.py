"""
Unit tests for POST /api/chat/sync.

We inject a mock agent via routes.set_agent() so no real LLM or index is needed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regulation_advisor.api import routes
from regulation_advisor.api.routes import router


@pytest.fixture()
def client_with_agent():
    """Build a minimal test app and inject a fake agent."""
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Article 5 prohibits social scoring.")],
        "retrieved_chunks": [],
        "confidence_score": 0.95,
    })

    app = FastAPI()
    app.include_router(router)

    original = routes._agent
    routes.set_agent(mock_agent)
    with TestClient(app) as c:
        yield c, mock_agent
    routes.set_agent(original)


def test_chat_sync_returns_200(client_with_agent):
    client, _ = client_with_agent
    response = client.post("/api/chat/sync", json={"message": "What is Article 5?"})
    assert response.status_code == 200


def test_chat_sync_answer_in_response(client_with_agent):
    client, _ = client_with_agent
    response = client.post("/api/chat/sync", json={"message": "What is Article 5?"})
    data = response.json()
    assert data["answer"] == "Article 5 prohibits social scoring."


def test_chat_sync_session_id_echoed(client_with_agent):
    client, _ = client_with_agent
    response = client.post(
        "/api/chat/sync",
        json={"message": "test", "session_id": "my-session-123"},
    )
    assert response.json()["session_id"] == "my-session-123"


def test_chat_sync_without_agent_returns_503():
    app = FastAPI()
    app.include_router(router)

    original = routes._agent
    routes.set_agent(None)
    try:
        with TestClient(app) as client:
            response = client.post("/api/chat/sync", json={"message": "hello"})
            assert response.status_code == 503
    finally:
        routes.set_agent(original)


def test_chat_sync_empty_message_returns_422(client_with_agent):
    client, _ = client_with_agent
    response = client.post("/api/chat/sync", json={"message": ""})
    assert response.status_code == 422
