"""
Thin file-based cache for RAGAS evaluation scores.

Reads from and writes to evals/baseline_scores.json — the same file
that scripts/run_evaluation.py produces. The API reads it on demand;
the background evaluation task overwrites it when a new run completes.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from regulation_advisor.api.schemas import MetricsResponse

logger = logging.getLogger(__name__)

_SCORES_PATH = Path("evals/baseline_scores.json")


def load() -> MetricsResponse | None:
    """Return the latest scores, or None if no evaluation has run yet."""
    if not _SCORES_PATH.exists():
        return None
    with open(_SCORES_PATH) as f:
        data = json.load(f)
    m = data.get("metrics", {})
    return MetricsResponse(
        version=data.get("version", "unknown"),
        faithfulness=m.get("faithfulness"),
        answer_relevancy=m.get("answer_relevancy"),
        context_precision=m.get("context_precision"),
        context_recall=m.get("context_recall"),
        total_qa_pairs=data.get("total_qa_pairs", 0),
        acceptable=data.get("acceptable", False),
        evaluated_at=data.get("evaluated_at", "unknown"),
    )


def save(faithfulness: float, answer_relevancy: float,
         context_precision: float, context_recall: float,
         total_qa_pairs: int, version: str = "v0.4") -> None:
    """Write a completed evaluation run to baseline_scores.json."""
    acceptable = faithfulness >= 0.80 and answer_relevancy >= 0.70
    payload = {
        "version": version,
        "week": 4,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        },
        "total_qa_pairs": total_qa_pairs,
        "acceptable": acceptable,
        "thresholds": {"faithfulness": 0.80, "answer_relevancy": 0.70},
    }
    _SCORES_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Saved evaluation scores to %s (acceptable=%s)", _SCORES_PATH, acceptable)
