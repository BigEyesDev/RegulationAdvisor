"""
Integration tests for the Week 4 API stack.

These tests build a minimal FastAPI app with the router (no lifespan) and
inject a mock agent. They verify that all routes work together correctly as
a unit — schemas → routes → schemas — without any real LLM or index.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regulation_advisor.api import metrics_store, routes
from regulation_advisor.api.routes import router


@pytest.fixture()
def mock_agent():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={
        "messages": [MagicMock(content="Article 5 prohibits social scoring by public authorities.")],
        "retrieved_chunks": [],
        "confidence_score": 0.92,
    })
    agent.astream_events = AsyncMock(return_value=iter([]))
    return agent


@pytest.fixture()
def client(mock_agent):
    app = FastAPI()
    app.include_router(router)
    original = routes._agent
    routes.set_agent(mock_agent)
    with TestClient(app) as c:
        yield c
    routes.set_agent(original)


@pytest.fixture()
def scores_path(tmp_path, monkeypatch):
    path = tmp_path / "scores.json"
    monkeypatch.setattr(metrics_store, "_SCORES_PATH", path)
    return path


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, client):
        assert client.get("/api/health").json()["status"] == "ok"

    def test_version_is_0_4_0(self, client):
        assert client.get("/api/health").json()["version"] == "0.4.0"


# ── Chat sync ─────────────────────────────────────────────────────────────────

class TestChatSync:
    def test_200_with_valid_message(self, client):
        r = client.post("/api/chat/sync", json={"message": "What is Article 5?"})
        assert r.status_code == 200

    def test_answer_in_response(self, client):
        r = client.post("/api/chat/sync", json={"message": "What is Article 5?"})
        assert "Article 5" in r.json()["answer"]

    def test_session_id_default(self, client):
        r = client.post("/api/chat/sync", json={"message": "test"})
        assert r.json()["session_id"] == "default"

    def test_session_id_custom(self, client):
        r = client.post("/api/chat/sync", json={"message": "test", "session_id": "abc"})
        assert r.json()["session_id"] == "abc"

    def test_422_on_empty_message(self, client):
        assert client.post("/api/chat/sync", json={"message": ""}).status_code == 422

    def test_422_on_message_too_long(self, client):
        r = client.post("/api/chat/sync", json={"message": "x" * 2001})
        assert r.status_code == 422

    def test_503_when_agent_not_ready(self):
        app = FastAPI()
        app.include_router(router)
        original = routes._agent
        routes.set_agent(None)
        try:
            with TestClient(app) as c:
                r = c.post("/api/chat/sync", json={"message": "hello"})
                assert r.status_code == 503
        finally:
            routes.set_agent(original)


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_404_before_any_evaluation(self, client, scores_path):
        assert client.get("/api/metrics").status_code == 404

    def test_200_after_scores_written(self, client, scores_path):
        scores_path.write_text(json.dumps({
            "version": "v0.4",
            "evaluated_at": "2026-07-15T12:00:00Z",
            "metrics": {
                "faithfulness": 0.82,
                "answer_relevancy": 0.75,
                "context_precision": 0.71,
                "context_recall": 0.68,
            },
            "total_qa_pairs": 20,
            "acceptable": True,
        }))
        r = client.get("/api/metrics")
        assert r.status_code == 200
        assert r.json()["faithfulness"] == 0.82

    def test_acceptable_flag_correct(self, client, scores_path):
        scores_path.write_text(json.dumps({
            "version": "v0.4", "evaluated_at": "x",
            "metrics": {"faithfulness": 0.85, "answer_relevancy": 0.72,
                        "context_precision": 0.70, "context_recall": 0.65},
            "total_qa_pairs": 20, "acceptable": True,
        }))
        assert client.get("/api/metrics").json()["acceptable"] is True


# ── Evaluate ──────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_started_status(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_evaluation_running", False)
        r = client.post("/api/evaluate")
        assert r.json()["status"] == "started"

    def test_already_running_status(self, client, monkeypatch):
        monkeypatch.setattr(routes, "_evaluation_running", True)
        r = client.post("/api/evaluate")
        assert r.json()["status"] == "already_running"


# ── Metrics store roundtrip ───────────────────────────────────────────────────

class TestMetricsStore:
    def test_load_returns_none_when_no_file(self, scores_path):
        assert metrics_store.load() is None

    def test_save_and_load(self, scores_path):
        metrics_store.save(
            faithfulness=0.83, answer_relevancy=0.74,
            context_precision=0.70, context_recall=0.67,
            total_qa_pairs=20,
        )
        result = metrics_store.load()
        assert result is not None
        assert result.faithfulness == 0.83
        assert result.total_qa_pairs == 20

    def test_acceptable_true_when_above_thresholds(self, scores_path):
        metrics_store.save(
            faithfulness=0.81, answer_relevancy=0.71,
            context_precision=0.65, context_recall=0.60,
            total_qa_pairs=20,
        )
        assert metrics_store.load().acceptable is True

    def test_acceptable_false_when_below_faithfulness(self, scores_path):
        metrics_store.save(
            faithfulness=0.79, answer_relevancy=0.75,
            context_precision=0.70, context_recall=0.65,
            total_qa_pairs=20,
        )
        assert metrics_store.load().acceptable is False
