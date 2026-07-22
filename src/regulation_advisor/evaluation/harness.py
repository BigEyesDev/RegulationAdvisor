"""RAGAS evaluation harness."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class _LocalEmbeddingsAdapter:
    """
    Bridges SentenceTransformerEmbedder to LangChain's Embeddings interface
    (embed_documents/embed_query), so RAGAS never needs an OpenAI embeddings
    call. Local, free, and reuses the same model already used for retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder

        self._embedder = SentenceTransformerEmbedder(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0].tolist()


def _build_ragas_judge():
    """
    RAGAS's default judge construction (unset llm=/embeddings=) is broken in
    the installed ragas/langchain-openai combo: its auto-factory returns a
    "modern"-interface embeddings object (embed_text) into the "legacy"
    metrics path this harness uses (ragas.metrics, which calls embed_query) —
    an AttributeError, not anything specific to which provider is used.

    Passing explicit llm=/embeddings= bypasses that broken auto-factory
    entirely. gpt-4o-mini as judge (fast, cheap, supports RAGAS's 3-way
    consistency sampling); local sentence-transformers embeddings (the
    actual fix — no OpenAI embeddings call at all). Uses settings.openai_api_key
    directly rather than relying on RAGAS's implicit OPENAI_API_KEY env-var
    read, which also fixes "Missing credentials" when only .env (not the
    shell environment) has the key.
    """
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from regulation_advisor.config import settings

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(_LocalEmbeddingsAdapter())
    return judge_llm, judge_embeddings


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def summary(self) -> str:
        status = "PASS" if self.is_acceptable() else "FAIL"
        return (f"[{status}] Faithfulness={self.faithfulness:.3f} | "
                f"Relevancy={self.answer_relevancy:.3f} | "
                f"Precision={self.context_precision:.3f} | "
                f"Recall={self.context_recall:.3f}")

    def is_acceptable(self) -> bool:
        # Faithfulness threshold is stricter (0.80) because hallucinated legal
        # claims are more dangerous than generic chatbot errors.
        return self.faithfulness >= 0.80 and self.answer_relevancy >= 0.70

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Saved RAGAS results to %s", path)


class EvaluationHarness:
    def __init__(self, qa_pairs_path: Path) -> None:
        with open(qa_pairs_path) as f:
            self._qa_pairs: list[dict] = json.load(f)
        logger.info("Loaded %d Q&A pairs from %s", len(self._qa_pairs), qa_pairs_path)

    def run(self, pipeline_fn: Callable[[str], tuple[str, list[str]]]) -> RAGASResult:
        """
        pipeline_fn: takes a question, returns (answer_str, list_of_context_strings)
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        data: dict[str, list] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for pair in self._qa_pairs:
            answer, contexts = pipeline_fn(pair["question"])
            data["question"].append(pair["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(pair["ground_truth_answer"])

        from ragas.run_config import RunConfig

        judge_llm, judge_embeddings = _build_ragas_judge()
        # Default max_workers=16 fires enough concurrent judge calls to trip
        # OpenAI's per-minute rate limit immediately; the resulting 429
        # backoff waits (40-50s+) then exceed the default 180s job timeout
        # before the retry even finishes, so jobs fail with TimeoutError
        # instead of actually retrying successfully. Fewer workers, more
        # patience.
        run_config = RunConfig(max_workers=3, timeout=300, max_wait=90)
        result = evaluate(
            dataset=Dataset.from_dict(data),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
        )
        # result[metric_name] is a per-question list, not an aggregate — some
        # entries are NaN when a judge call fails (rate limit, context
        # overflow, timeout), so average with nanmean and log how many of
        # each metric actually scored rather than silently dropping them.
        return RAGASResult(
            faithfulness=_mean_score(result["faithfulness"], "faithfulness"),
            answer_relevancy=_mean_score(result["answer_relevancy"], "answer_relevancy"),
            context_precision=_mean_score(result["context_precision"], "context_precision"),
            context_recall=_mean_score(result["context_recall"], "context_recall"),
        )


def _mean_score(scores: list[float], metric_name: str) -> float:
    import numpy as np

    valid = int(np.sum(~np.isnan(scores)))
    total = len(scores)
    if valid < total:
        logger.warning(
            "%s: %d/%d judge calls failed (rate limit, timeout, or context "
            "overflow) — averaging over the %d that succeeded",
            metric_name, total - valid, total, valid,
        )
    return float(np.nanmean(scores))
