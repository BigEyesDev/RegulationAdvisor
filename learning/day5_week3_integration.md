# Day 6 — Week 3 Integration

## What we built
- `evals/baseline_scores.json` — placeholder for the real scorecard (populated by `run_evaluation.py`)
- `tests/integration/test_week3_pipeline.py` — 7 integration tests wiring all Week 3 components together

---

## What "integration" means vs "unit testing"

### Unit testing
Test one thing in isolation. If a function breaks, you know exactly which function.

```python
# Unit test — tests CitationVerificationCheck alone
def test_hallucinated_article_flagged():
    result = build_guardrail_chain(0.0).check("See Article 99.", [_chunk("5")], 0.9)
    assert any("not in retrieved" in w.lower() for w in result.warnings)
```

### Integration testing
Test multiple things working together. Catches bugs that only appear
when real components are connected.

```python
# Integration test — tests the full chain: FaithfulnessCheck → CitationVerificationCheck → LegalClaimFlagCheck
def test_legal_claim_adds_warning_without_blocking(self):
    chunks = [_chunk("5")]
    result = self.chain.check(
        "Under Article 5, you must cease this practice immediately.",
        chunks,
        confidence=1.0,
    )
    assert result.passed              # LegalClaimFlagCheck does NOT block
    assert any("not legal advice" in w for w in result.warnings)  # but it DOES warn
```

This test is an integration test because it checks the interaction between
LegalClaimFlagCheck (which adds a warning) and the pass/fail state set by
the upstream checks. If you tested LegalClaimFlagCheck alone, you might
not notice that its warning logic accidentally sets `passed=False`.

---

## The analogy: unit vs integration in manufacturing

Think of building a car.

**Unit test:** Test each part in isolation.
- Does the engine start? ✓
- Do the brakes grip? ✓
- Does the steering wheel turn? ✓

**Integration test:** Test the parts together in the car.
- Does braking while turning cause the car to skid? (brake + steering interaction)
- Does the engine stall when the air conditioning is turned on? (engine + AC interaction)

Passing all unit tests does not guarantee the car works. You need integration tests
for the interactions between components.

In this codebase:
- `FaithfulnessCheck` unit-tested in `test_guardrails.py`
- `CitationVerificationCheck` unit-tested in `test_guardrail_integration.py`
- The full chain integration-tested in `test_week3_pipeline.py`

---

## The most important test: `test_low_confidence_blocks_before_citation_check`

```python
def test_low_confidence_blocks_before_citation_check(self):
    chunks = [_chunk("5")]
    result = self.chain.check("Article 5 prohibits this.", chunks, confidence=0.3)
    assert not result.passed
    # Low confidence blocks at FaithfulnessCheck — no citation check runs
    assert not any("not in retrieved" in w.lower() for w in result.warnings)
```

This tests the SHORT-CIRCUIT behaviour of the chain. When `FaithfulnessCheck`
returns `passed=False` early, it does NOT call `self._pass_to_next()`.
Therefore `CitationVerificationCheck` never runs. Therefore no "not in retrieved"
warning should appear.

If this test fails, it means a refactoring broke the chain's early-exit logic —
a subtle bug that would only surface in code review, not in the unit tests.

---

## The `pytest.importorskip` pattern

```python
def test_harness_run_with_mock_pipeline(self, harness):
    pytest.importorskip("ragas", reason="ragas not installed in this environment")
    ...
```

`pytest.importorskip("ragas")` does two things:
1. Tries to import `ragas`
2. If the import fails, it SKIPS the test (marks it as skipped, not failed)

Why is this important?

The full `ragas` library requires Python 3.9+, several hundred megabytes of
dependencies, and an OpenAI API key to run. In a CI environment that only
runs unit tests, or on a machine that has not run `pip install ragas`,
the test should not fail — it should skip.

The test runner output will show `SKIPPED` instead of `FAILED`.
`SKIPPED` means: "this test was not run because a precondition was missing."
`FAILED` means: "this test ran and the assertion was wrong."

These are very different and should never be confused.

---

## The `baseline_scores.json` file

```json
{
  "version": "v0.3",
  "week": 3,
  "note": "Run `python scripts/run_evaluation.py` to populate this file with real scores.",
  "metrics": {
    "faithfulness": null,
    "answer_relevancy": null,
    "context_precision": null,
    "context_recall": null
  },
  "thresholds": {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70
  },
  "total_qa_pairs": 20
}
```

The `null` values are intentional. They signal "not yet measured".
A human running the project for the first time should see `null` and
know they need to run `python scripts/run_evaluation.py` to populate the file.

Do NOT commit fake numbers (like `"faithfulness": 0.85`).
Fake numbers mislead future developers who compare their changes
against a baseline that was never actually measured.

After running `run_evaluation.py`:
```json
{
  "faithfulness": 0.82,
  "answer_relevancy": 0.74,
  "context_precision": 0.68,
  "context_recall": 0.71
}
```

These are your real Week 3 numbers. When Week 4 improves the retrieval
(ChromaDB + better chunking), you run the evaluation again and compare
against these numbers to prove the improvement is real.

---

## The complete Week 3 test inventory

After this week, running `pytest tests/ -v` should show:

| File | Tests | What they cover |
|---|---|---|
| `test_config.py` | 6 | Settings loading |
| `test_chunkers.py` | 4 | ArticleAwareChunker and RecursiveChunker |
| `test_loaders.py` | 3 | DocumentLoaderFactory routing |
| `test_tools.py` | 3 | search_regulations, query_structured_data |
| `test_agent_graph.py` | 3 | LangGraph compilation and routing |
| `test_eval_dataset.py` | 3 | qa_pairs.json structure |
| `test_harness.py` | 5 | RAGASResult thresholds and save() |
| `test_guardrails.py` | 3 | Individual guardrail handlers |
| `test_guardrail_integration.py` | 5 | Article parsing + chain behaviour |
| `test_week3_pipeline.py` | 7 | Full integration of all Week 3 components |
| **Total** | **42** | |

---

## How to run the complete Week 3 evaluation (real numbers)

Step 1: Make sure the FAISS index is built
```bash
python scripts/ingest.py
```

Step 2: Run RAGAS batch evaluation
```bash
python scripts/run_evaluation.py
```
This populates `evals/baseline_scores.json` with real scores.

Step 3: Run all tests
```bash
pytest tests/ -v
```

Step 4: Verify the promptfoo suite locally
```bash
npx promptfoo eval --config evals/promptfoo.yaml
```

When all four steps pass, Week 3 is complete.

---

## What you now know vs what a beginner does not

After this week, you understand:
- Why evaluation datasets are written before code (ground truth must be independent)
- What each RAGAS metric measures and why faithfulness has a stricter threshold
- How the Chain of Responsibility pattern lets you add/remove guardrail checks trivially
- How LangGraph's checkpointer stores conversation state you can read after streaming
- Why promptfoo regression tests complement RAGAS scores (structural vs statistical)
- What a mutation test is and why you should deliberately break your CI to prove it works
- The difference between unit tests (isolation) and integration tests (interaction)
- Why `pytest.importorskip` is better than `try/except` around test bodies

These are the things that separate an ML engineer who builds models
from an AI engineer who builds production AI systems.
The model is 10% of the work. This week was the other 90%.
