"""
API routes for RegulationAdvisor.

_agent is set once at startup by api/app.py lifespan.
Routes read it at request time — never at import time.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

from regulation_advisor.api.schemas import HealthResponse
from regulation_advisor.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Set by api/app.py lifespan via set_agent()
_agent = None


def set_agent(agent: object) -> None:
    global _agent
    _agent = agent


@router.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.4.0",
        vector_store_backend=settings.vector_store_backend,
    )
