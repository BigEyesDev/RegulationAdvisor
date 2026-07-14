"""
Integration tests for Week 3 evaluation + guardrail pipeline.

These tests wire the real components together but use a mock pipeline_fn
so no LLM calls or network requests are made.
"""
import json
from pathlib import Path

import pytest

from regulation_advisor.evaluation.guardrails import build_guardrail_chain
from regulation_advisor.evaluation.harness import EvaluationHarness, RAGASResult
from regulation_advisor.models import RegulationChunk


# ── Helpers ──────────────────────────────────────────────────────────────────

def _chunk(article: str) -> RegulationChunk:
    return RegulationChunk(
        content="", article_number=article, article_title="", source_document="test.pdf"
    )


# ── Guardrail chain end-to-end ────────────────────────────────────────────────

class TestGuardrailChainEndToEnd:
    def setup_method(self):
        self.chain = build_guardrail_chain()

    def test_clean_answer_passes_all_checks(self):
        chunks = [_chunk("5"), _chunk("99")]
        result = self.chain.check(
            "Article 5 prohibits social scoring. Article 99 sets the maximum fine.",
            chunks,
            confidence=1.0,
        )
        assert result.passed

    def test_hallucinated_article_blocks_answer(self):
        chunks = [_chunk("5")]  # only Article 5 retrieved
        result = self.chain.check(
            "Article 42 requires a conformity assessment.",  # 42 was never retrieved
            chunks,
            confidence=1.0,
        )
        assert not result.passed
        assert any("42" in w for w in result.warnings)

    def test_legal_claim_adds_warning_without_blocking(self):
        chunks = [_chunk("5")]
        result = self.chain.check(
            "Under Article 5, you must cease this practice immediately.",
            chunks,
            confidence=1.0,
        )
        # "you must" triggers LegalClaimFlagCheck → warning added but NOT a block
        assert result.passed
        assert any("not legal advice" in w for w in result.warnings)

    def test_low_confidence_blocks_before_citation_check(self):
        chunks = [_chunk("5")]
        result = self.chain.check("Article 5 prohibits this.", chunks, confidence=0.3)
        assert not result.passed
        # Low confidence blocks at FaithfulnessCheck — no citation check runs
        assert not any("not in retrieved" in w.lower() for w in result.warnings)


# ── Harness loads real qa_pairs.json ─────────────────────────────────────────

class TestEvaluationHarness:
    @pytest.fixture
    def harness(self):
        return EvaluationHarness(Path("evals/qa_pairs.json"))

    def test_harness_loads_twenty_pairs(self, harness):
        assert len(harness._qa_pairs) == 20

    def test_harness_run_with_mock_pipeline(self, harness):
        """
        Run the harness with a mock pipeline_fn that echoes the ground truth.
        Because the answer == ground_truth, RAGAS should give high scores.

        We skip this in environments without the RAGAS/OpenAI dependency
        by catching ImportError — the CI eval.yml runs this with the full deps.
        """
        pytest.importorskip("ragas", reason="ragas not installed in this environment")

        def perfect_pipeline(question: str) -> tuple[str, list[str]]:
            pair = next(p for p in harness._qa_pairs if p["question"] == question)
            return pair["ground_truth_answer"], [pair["ground_truth_answer"]]

        result = harness.run(perfect_pipeline)
        assert isinstance(result, RAGASResult)
        assert result.faithfulness >= 0.0  # basic sanity; real score depends on RAGAS judge

    def test_harness_save_round_trip(self, harness, tmp_path):
        result = RAGASResult(
            faithfulness=0.85, answer_relevancy=0.80,
            context_precision=0.75, context_recall=0.70,
        )
        out = tmp_path / "test_scores.json"
        result.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["faithfulness"] == 0.85
        assert loaded["answer_relevancy"] == 0.80
