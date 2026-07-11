"""RAGAS evaluation harness — built in Week 3."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def summary(self) -> str:
        return (f"Faithfulness: {self.faithfulness:.3f} | "
                f"Relevancy: {self.answer_relevancy:.3f} | "
                f"Precision: {self.context_precision:.3f} | "
                f"Recall: {self.context_recall:.3f}")

    def is_acceptable(self, threshold: float = 0.7) -> bool:
        return self.faithfulness >= threshold and self.answer_relevancy >= threshold


class EvaluationHarness:
    def __init__(self, qa_pairs_path: Path) -> None:
        with open(qa_pairs_path) as f:
            self._qa_pairs: list[dict] = json.load(f)
        logger.info("Loaded %d Q&A pairs from %s", len(self._qa_pairs), qa_pairs_path)

    def run(self, pipeline_fn: Callable[[str], tuple[str, list[str]]]) -> RAGASResult:
        """
        pipeline_fn: takes a question, returns (answer_str, list_of_context_strings)
        """
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        data: dict[str, list] = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for pair in self._qa_pairs:
            answer, contexts = pipeline_fn(pair["question"])
            data["question"].append(pair["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(pair["ground_truth_answer"])

        result = evaluate(
            dataset=Dataset.from_dict(data),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return RAGASResult(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
        )
