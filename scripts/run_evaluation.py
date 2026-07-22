"""
Run RAGAS evaluation against the full agent pipeline and save the scorecard.

Usage:
    python scripts/run_evaluation.py

Output:
    evals/baseline_scores.json   — scorecard from this run
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regulation_advisor.agent.graph import build_agent_graph
from regulation_advisor.agent.tools import set_retriever
from regulation_advisor.config import settings
from regulation_advisor.evaluation.harness import EvaluationHarness
from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.retrieval.store import build_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
QA_PATH = _ROOT / "evals" / "qa_pairs.json"
OUTPUT_PATH = _ROOT / "evals" / "baseline_scores.json"


def make_pipeline_fn(agent):
    """Wrap the agent so harness.run() can call it with just a question."""

    def pipeline_fn(question: str) -> tuple[str, list[str]]:
        # A unique thread_id per question — a shared one would let LangGraph's
        # MemorySaver accumulate every prior question's retrieved chunks and
        # answer into each subsequent call, ballooning context size until the
        # judge LLM's context window overflows (seen: 130k-165k tokens by the
        # last few questions in a 20-question run).
        config = {"configurable": {"thread_id": f"ragas-eval-{hash(question)}"}}
        result = agent.invoke({"messages": [("human", question)]}, config=config)
        answer = result["messages"][-1].content
        # Extract tool message contents as context strings for RAGAS
        contexts = [
            m.content for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
        return answer, contexts or [answer]

    return pipeline_fn


def main() -> None:
    logger.info("Loading vector store (%s)…", settings.vector_store_backend)
    store = build_vector_store()
    store.load(_ROOT / settings.index_dir)
    set_retriever(Retriever(store=store, embedder=SentenceTransformerEmbedder()))

    logger.info("Building agent...")
    agent = build_agent_graph()

    logger.info("Loading evaluation harness from %s", QA_PATH)
    harness = EvaluationHarness(QA_PATH)

    logger.info("Running evaluation — this calls the LLM for each Q&A pair...")
    result = harness.run(make_pipeline_fn(agent))

    print("\n" + result.summary())
    result.save(OUTPUT_PATH)
    logger.info("Done. Results saved to %s", OUTPUT_PATH)

    if not result.is_acceptable():
        logger.warning("Scores are below threshold — review results before shipping.")
        sys.exit(1)


if __name__ == "__main__":
    main()
