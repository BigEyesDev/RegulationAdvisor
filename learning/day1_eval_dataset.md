# Day 1 — Building the Evaluation Dataset

## What we built
`evals/qa_pairs.json` — 20 question-and-answer pairs with verified ground truth.
`tests/unit/test_eval_dataset.py` — 3 structural tests that guard the dataset.

---

## Why an evaluation dataset exists at all

Imagine you hire a new legal assistant and on their first day they answer 10 questions.
On day 30 you make some changes to how they work (maybe give them a new reference book),
and now you want to know: are they better or worse than before?

Without a written record of what the correct answers are, you cannot compare.
You are just guessing.

An evaluation dataset is exactly that written record. It is a list of:
- A question someone might actually ask
- The provably correct answer (you verified it yourself against the source document)
- Which article the answer comes from

Every time you change anything in the system — the prompt, the chunking strategy,
the model — you run the system against this dataset and compare numbers.
That is the only honest way to know if your change helped or hurt.

---

## The three fields in each Q&A pair

```json
{
  "question": "What AI practices are completely prohibited?",
  "ground_truth_answer": "Article 5 prohibits: social scoring...",
  "expected_article": "5"
}
```

**question** — A realistic thing a compliance officer or developer would type.
Write it in natural language, not as a keyword search. Bad: "Article 5 prohibited list".
Good: "Can an employer use AI to detect employees' emotions at work?"

**ground_truth_answer** — The correct answer according to the actual EU AI Act PDF.
This is NOT what the AI says. This is what you wrote after reading the regulation yourself.
It is the yardstick you measure the AI against.

**expected_article** — The article number where the answer lives.
This lets you verify that retrieval actually found the right chunk, not just any chunk.

---

## What "ground truth" means and why it matters

Ground truth is a term borrowed from map-making. Surveyors would walk the actual land to
measure it (ground truth), then check whether the map (model output) matched reality.

In machine learning, ground truth = the correct label a human has verified.
The AI's output is always compared against the ground truth, never against itself.

If you let the AI verify its own answers, you get circular reasoning:
the AI says it is correct, and you have no way to catch when it is wrong.
Ground truth breaks that loop.

---

## Coverage strategy — why 20 specific questions

A good evaluation dataset covers the system's full responsibility, not just its easy cases.

| Category | Questions | Why |
|---|---|---|
| Prohibited practices (Art. 5) | 4 | The most consequential errors — a wrong answer could get someone prosecuted |
| Penalties (Art. 99) | 3 | Numbers must be exact — 7% not 10%, 35M not 30M |
| Enforcement timeline (Art. 113) | 3 | Deadlines are date-sensitive; the model must get years right |
| High-risk classification (Art. 6, Annex III) | 3 | The most common question type from developers |
| Obligations for high-risk (Art. 9–14) | 4 | The longest section of the Act — easy to hallucinate |
| GDPR interaction + Transparency (Art. 2, 50) | 2 | Cross-regulation — common in practice |
| GPAI models (Art. 51) | 1 | Emerging area, newer provisions |

The balance matters. If all 20 questions were about Article 5, you would get a
good score on prohibited practices but have no idea whether the system handles penalties
or timelines correctly.

---

## The test file — `tests/unit/test_eval_dataset.py`

```python
def test_minimum_pair_count():
    assert len(load_pairs()) >= 20
```

This ensures nobody accidentally deletes entries from the file.
If the count drops below 20, the test fails loudly rather than silently
running a smaller evaluation and producing misleadingly good scores.

```python
def test_all_pairs_have_required_keys():
    for pair in load_pairs():
        assert REQUIRED_KEYS <= pair.keys()
```

The `<=` operator on sets means "is a subset of". It checks that every
required key exists in the pair's keys. If you add a new entry and
forget the `expected_article` field, this test catches it.

```python
def test_no_empty_fields():
    for pair in load_pairs():
        for key in REQUIRED_KEYS:
            assert pair[key].strip()
```

`.strip()` removes whitespace. `assert ""` is falsy in Python, so
an empty string fails this assertion. This prevents entries like
`"ground_truth_answer": "  "` slipping through.

---

## How this connects to the rest of Week 3

This dataset is used in three places:

1. **Day 2 (RAGAS harness)** — the harness loads `qa_pairs.json` and runs every
   question through the pipeline, comparing the AI's answer against `ground_truth_answer`
   to compute faithfulness, relevancy, and retrieval scores.

2. **Days 4–5 (promptfoo)** — the same questions become regression test cases.
   Promptfoo checks structural things: does the answer contain "Article 5"?
   Does it mention "35,000,000"? These run on every pull request to catch regressions.

3. **Week 4 and beyond** — when you swap from FAISS to ChromaDB, or change
   your chunk size, you re-run this dataset and compare the new scores against
   `evals/baseline_scores.json` to prove the change was an improvement.

The dataset is permanent infrastructure. Treat it like a test suite — never delete
entries, only add and improve them.

---

## Common mistakes to avoid

**Mistake 1: Copying the answer from the AI and calling it ground truth.**
If the AI got it wrong, you now have wrong ground truth. The whole evaluation
becomes circular. Always write the answer yourself after reading the PDF.

**Mistake 2: Asking questions the system was never designed to handle.**
"What did Ursula von der Leyen say about AI?" is not a compliance question.
Every question should be something a real developer or compliance officer
would ask about the regulation.

**Mistake 3: Only including easy questions.**
If every question has an obvious single-article answer, you will get
a high score that does not reflect real-world performance where questions
span multiple articles or require interpretation.

**Mistake 4: Never updating the dataset.**
When you find a question the system answers badly in production,
add it to the dataset. The dataset should grow as you discover edge cases.
