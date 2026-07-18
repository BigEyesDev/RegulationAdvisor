"""
Unit tests for GET /api/health.

We build a minimal FastAPI app with just the router (no lifespan) so the
test runs without needing an index file or API keys.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from regulation_advisor.api.routes import router
from regulation_advisor.config import Settings

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_version():
    response = client.get("/api/health")
    assert response.json()["version"] == "0.6.3"


def test_health_vector_store_backend_is_valid():
    response = client.get("/api/health")
    backend = response.json()["vector_store_backend"]
    assert backend in ("faiss", "chromadb")


def test_health_backend_matches_default_config():
    response = client.get("/api/health")
    assert response.json()["vector_store_backend"] == Settings.model_fields["vector_store_backend"].default
