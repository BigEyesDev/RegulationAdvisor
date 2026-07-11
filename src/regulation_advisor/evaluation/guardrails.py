"""
Guardrail layer — Chain of Responsibility pattern.

Each handler checks one thing. Chain them together with build_guardrail_chain().
Add new checks by creating a new GuardrailHandler subclass.
"""
from __future__ import annotations

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from regulation_advisor.models import RegulationChunk

logger = logging.getLogger(__name__)

LEGAL_CLAIM_PHRASES = ["you must", "it is illegal", "the fine is", "you are required to",
                       "you must immediately"]


@dataclass
class GuardrailResult:
    passed: bool
    warnings: list[str] = field(default_factory=list)
    confidence_score: float = 1.0


class GuardrailHandler(ABC):
    def __init__(self, next_handler: GuardrailHandler | None = None) -> None:
        self._next = next_handler

    @abstractmethod
    def check(self, answer: str, chunks: list[RegulationChunk],
               confidence: float) -> GuardrailResult: ...

    def _pass_to_next(self, answer: str, chunks: list[RegulationChunk],
                      confidence: float) -> GuardrailResult:
        if self._next:
            return self._next.check(answer, chunks, confidence)
        return GuardrailResult(passed=True, confidence_score=confidence)


class FaithfulnessCheck(GuardrailHandler):
    def __init__(self, threshold: float = 0.7, next_handler: GuardrailHandler | None = None):
        super().__init__(next_handler)
        self._threshold = threshold

    def check(self, answer, chunks, confidence):
        if confidence < self._threshold:
            return GuardrailResult(
                passed=False, confidence_score=confidence,
                warnings=[f"⚠️ Low confidence ({confidence:.2f}) — verify against the regulation."],
            )
        return self._pass_to_next(answer, chunks, confidence)


class CitationVerificationCheck(GuardrailHandler):
    def check(self, answer, chunks, confidence):
        cited = set(re.findall(r"Article\s+(\d+)", answer, re.IGNORECASE))
        available = {c.article_number for c in chunks}
        hallucinated = cited - available
        result = self._pass_to_next(answer, chunks, confidence)
        if hallucinated:
            result.passed = False
            result.warnings.append(f"⚠️ Cited articles not in retrieved context: {hallucinated}")
        return result


class LegalClaimFlagCheck(GuardrailHandler):
    def check(self, answer, chunks, confidence):
        result = self._pass_to_next(answer, chunks, confidence)
        if any(p in answer.lower() for p in LEGAL_CLAIM_PHRASES):
            result.warnings.append(
                "ℹ️ This answer contains legal claims. This is AI-generated guidance, "
                "not legal advice. Verify with a qualified lawyer."
            )
        return result


def build_guardrail_chain(faithfulness_threshold: float = 0.7) -> GuardrailHandler:
    """Returns the full chain: faithfulness → citation → legal claim."""
    return FaithfulnessCheck(
        threshold=faithfulness_threshold,
        next_handler=CitationVerificationCheck(
            next_handler=LegalClaimFlagCheck()
        ),
    )
