"""
RegClassifier — fine-tuned regulation classifier.
Stub implementation — replace with real inference once the model is trained.
"""
from __future__ import annotations

import logging
from regulation_advisor.models import RegulationFinding

logger = logging.getLogger(__name__)


class RegClassifier:
    """Classifies regulation text into risk tier, obligation type, and urgency."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        # TODO: load fine-tuned model with peft + transformers
        logger.info("Loading RegClassifier from %s", path)

    def classify(self, text: str) -> RegulationFinding:
        # TODO: replace with real model inference
        logger.warning("RegClassifier is a stub — returning placeholder result")
        return RegulationFinding(
            article="unknown",
            risk_tier="Minimal",
            obligation_type="TRANSPARENCY",
            urgency="2026",
            confidence=0.0,
            reasoning="Classifier not yet trained.",
        )
