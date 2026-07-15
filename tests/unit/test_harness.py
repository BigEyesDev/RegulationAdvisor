"""Tests for RAGASResult — no LLM, no network calls."""
import json
from pathlib import Path
from regulation_advisor.evaluation.harness import RAGASResult


def test_summary_includes_pass_label():
    result = RAGASResult(faithfulness=0.85, answer_relevancy=0.80,
                         context_precision=0.75, context_recall=0.70)
    assert "[PASS]" in result.summary()


def test_summary_includes_fail_label():
    result = RAGASResult(faithfulness=0.60, answer_relevancy=0.80,
                         context_precision=0.75, context_recall=0.70)
    assert "[FAIL]" in result.summary()


def test_is_acceptable_passes_above_thresholds():
    result = RAGASResult(faithfulness=0.81, answer_relevancy=0.71,
                         context_precision=0.50, context_recall=0.50)
    assert result.is_acceptable()


def test_is_acceptable_fails_on_low_faithfulness():
    result = RAGASResult(faithfulness=0.79, answer_relevancy=0.90,
                         context_precision=0.90, context_recall=0.90)
    assert not result.is_acceptable()


def test_save_writes_valid_json(tmp_path: Path):
    result = RAGASResult(faithfulness=0.85, answer_relevancy=0.80,
                         context_precision=0.75, context_recall=0.70)
    out = tmp_path / "scores.json"
    result.save(out)
    data = json.loads(out.read_text())
    assert data["faithfulness"] == 0.85
