"""
Regression check for scope-gate + citation behavior.

Runs a fixed set of good/trick/bad/gray-area questions (evals/regression_questions.json)
against the live agent graph and checks that each response matches its expected
shape (answered-with-citation vs. rejected). Run this after any change to the
system prompt, agent graph, or tools — not a substitute for the RAGAS eval
(scripts/run_evaluation.py), which scores answer quality, not scope behavior.

Usage:
    uv run python scripts/run_regression_questions.py
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regulation_advisor.agent.graph import build_agent_graph
from regulation_advisor.agent.tools import set_retriever
from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.retrieval.store import build_vector_store
from regulation_advisor.config import settings

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = _ROOT / "evals" / "regression_questions.json"

# Two independent, individually-imperfect signals that a response was a scope-gate
# rejection rather than a substantive answer:
#   1. No "Article N" citation — the prompt requires one for every real answer, and
#      forbids constructing an answer when rejecting.
#   2. Explicit refusal language — but the model phrases this differently every run,
#      and sometimes cites an article while explaining *why* something is out of
#      scope (e.g. "this isn't a high-risk use under Article 6"), which defeats
#      signal 1 alone. Either signal firing is treated as a rejection.
ARTICLE_CITATION = re.compile(r"article\s+\d+", re.IGNORECASE)
REJECT_PHRASES = [
    "out of scope", "outside the scope", "outside of scope",
    "cannot answer this question", "does not describe a coherent activity",
    "cannot and will not construct an answer", "frivolous",
    "not a genuine compliance", "not a genuine use case",
    "has nothing to do with", "unrelated to", "no ai system",
    "does not concern", "does not involve",
]


def check(case: dict, answer: str) -> tuple[bool, str]:
    cited = bool(ARTICLE_CITATION.search(answer))
    lower = answer.lower()
    refused = any(p in lower for p in REJECT_PHRASES)
    rejected = refused or not cited

    if case["expect"] == "reject":
        if rejected:
            return True, "rejected as expected"
        return False, "expected a rejection but got a substantive-looking answer"

    if case["expect"] == "reject_or_answer":
        # Ambiguous cases where either a flat refusal or a reasoned exemption
        # citation is acceptable — just confirm it didn't crash or go silent.
        return bool(answer.strip()), "got a non-empty response (either outcome accepted)"

    # expect == "answer"
    if rejected:
        return False, "expected a substantive answer but it looks like the scope gate rejected it"
    must_contain_any = case.get("must_contain_any", [])
    if must_contain_any and not any(m.lower() in lower for m in must_contain_any):
        return False, f"answer missing expected citation(s): {must_contain_any}"
    return True, "answered with expected citation(s)"


def main() -> None:
    cases = json.loads(QUESTIONS_PATH.read_text())

    logger.info("Loading vector store (%s)…", settings.vector_store_backend)
    store = build_vector_store()
    store.load(_ROOT / settings.index_dir)
    set_retriever(Retriever(store=store, embedder=SentenceTransformerEmbedder()))

    agent = build_agent_graph()

    results = []
    for case in cases:
        config = {"configurable": {"thread_id": f"regression-{case['id']}"}}
        result = agent.invoke({"messages": [("human", case["question"])]}, config=config)
        answer = result["messages"][-1].content
        passed, reason = check(case, answer)
        results.append((case["id"], case["type"], passed, reason, answer))

    print("\n=== Regression results ===")
    n_failed = 0
    for case_id, case_type, passed, reason, answer in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            n_failed += 1
        print(f"[{status}] {case_id} ({case_type}): {reason}")
        if not passed:
            print(f"    --- answer ---\n    {answer[:500]}\n")

    print(f"\n{len(results) - n_failed}/{len(results)} passed.")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
