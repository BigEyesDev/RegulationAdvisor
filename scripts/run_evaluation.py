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

from regulation_advisor.evaluation.harness import EvaluationHarness
from regulation_advisor.agent.graph import build_agent_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

QA_PATH = Path("evals/qa_pairs.json")
OUTPUT_PATH = Path("evals/baseline_scores.json")


def make_pipeline_fn(agent):
    """Wrap the agent so harness.run() can call it with just a question."""
    config = {"configurable": {"thread_id": "ragas-eval"}}

    def pipeline_fn(question: str) -> tuple[str, list[str]]:
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
