"""
Unit tests for GET /api/metrics and POST /api/evaluate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regulation_advisor.api import metrics_store, routes
from regulation_advisor.api.routes import router


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def scores_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary scores file and point metrics_store at it."""
    path = tmp_path / "baseline_scores.json"
    monkeypatch.setattr(metrics_store, "_SCORES_PATH", path)
    return path


def test_metrics_404_when_no_file(client, scores_file):
    response = client.get("/api/metrics")
    assert response.status_code == 404


def test_metrics_returns_scores_from_file(client, scores_file):
    scores_file.write_text(json.dumps({
        "version": "v0.4",
        "evaluated_at": "2026-07-15T10:00:00Z",
        "metrics": {
            "faithfulness": 0.84,
            "answer_relevancy": 0.79,
            "context_precision": 0.72,
            "context_recall": 0.68,
        },
        "total_qa_pairs": 20,
        "acceptable": True,
    }))
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["faithfulness"] == 0.84
    assert data["version"] == "v0.4"
    assert data["acceptable"] is True


def test_metrics_store_save_and_load_roundtrip(scores_file, monkeypatch):
    monkeypatch.setattr(metrics_store, "_SCORES_PATH", scores_file)
    metrics_store.save(
        faithfulness=0.85,
        answer_relevancy=0.76,
        context_precision=0.71,
        context_recall=0.69,
        total_qa_pairs=20,
    )
    result = metrics_store.load()
    assert result is not None
    assert result.faithfulness == 0.85
    assert result.acceptable is True  # 0.85 >= 0.80 and 0.76 >= 0.70


def test_evaluate_returns_started(client, monkeypatch):
    monkeypatch.setattr(routes, "_evaluation_running", False)
    response = client.post("/api/evaluate")
    assert response.status_code == 200
    assert response.json()["status"] == "started"


def test_evaluate_returns_already_running_when_busy(client, monkeypatch):
    monkeypatch.setattr(routes, "_evaluation_running", True)
    response = client.post("/api/evaluate")
    assert response.json()["status"] == "already_running"
