"""
RegClassifier — fine-tuned regulation classifier.
Built in Week 6. Placeholder until then.
"""
from __future__ import annotations

import logging
from regulation_advisor.models import RegulationFinding

logger = logging.getLogger(__name__)


class RegClassifier:
    """
    Classifies regulation text into risk tier, obligation type, and urgency.
    Week 1-5: not used.
    Week 6: fine-tune and replace the stub below with real inference.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        # TODO Week 6: load fine-tuned model with peft + transformers
        logger.info("Loading RegClassifier from %s", path)

    def classify(self, text: str) -> RegulationFinding:
        # TODO Week 6: replace with real model inference
        logger.warning("RegClassifier is a stub — returning placeholder result")
        return RegulationFinding(
            article="unknown",
            risk_tier="Minimal",
            obligation_type="TRANSPARENCY",
            urgency="2026",
            confidence=0.0,
            reasoning="Classifier not yet trained.",
        )
