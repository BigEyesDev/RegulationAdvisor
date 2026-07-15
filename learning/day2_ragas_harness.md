# Day 2 — The RAGAS Evaluation Harness

## What we built
- `src/regulation_advisor/evaluation/harness.py` — enhanced with `save()` and stricter thresholds
- `scripts/run_evaluation.py` — one command to run the full evaluation
- `tests/unit/test_harness.py` — 5 tests covering thresholds, output format, and file saving

---

## What RAGAS is — and why it exists

RAGAS stands for **Retrieval Augmented Generation Assessment**.
It is a library that measures how well a RAG system is working across four dimensions.

Think of it like a restaurant health inspection. An inspector doesn't just eat the food
and say "tastes fine". They check the kitchen temperature, the food storage,
the cleanliness of surfaces, and the staff hygiene. Each dimension is measured separately
because they can fail independently — the food can taste great and still make you sick.

RAGAS does the same for a RAG system: it measures four independent dimensions,
each of which can fail even if the others look fine.

---

## The four RAGAS metrics explained

### Faithfulness

**What it measures:** Is every claim in the AI's answer actually supported by
the retrieved text chunks?

**Analogy:** A journalist writing an article cites their sources. Faithfulness
measures whether the article's claims actually appear in those sources.
If the journalist says "the CEO earned $10M" but their sources say "$8M",
that is a faithfulness failure — even if the rest of the article is perfect.

**Why it matters here:** Legal advice that cites the wrong number (35M vs 10M,
7% vs 3%) has real consequences. Faithfulness is the single most important metric
for this application, which is why we set its threshold at 0.80, stricter than
the 0.70 we use for everything else.

**Range:** 0.0 (every claim is hallucinated) → 1.0 (every claim is grounded in context)

---

### Answer Relevancy

**What it measures:** Does the answer actually address the question that was asked?

**Analogy:** Someone asks "how do I file a GDPR complaint?" and the system responds
with a detailed explanation of what GDPR stands for. Technically true, but irrelevant.
Answer relevancy catches this.

**Why it matters:** Users ask specific questions and need specific answers.
A system that always responds with related-but-not-quite-right information is
frustrating and useless for compliance decisions.

**Range:** 0.0 (answer is completely off-topic) → 1.0 (answer directly addresses the question)

---

### Context Precision

**What it measures:** Among all the chunks the retriever fetched, what fraction
were actually relevant to answering the question?

**Analogy:** You ask a librarian for books about GDPR fines. They bring you
12 books: 5 about GDPR fines, 4 about general GDPR history, and 3 about
unrelated EU law. Context precision = 5 / 12 = 0.42. Low.
A good librarian brings you mostly relevant books.

**Why it matters:** If your retriever returns a lot of noise, the LLM
has to find the signal. This increases the chance of confused answers
and wastes the context window.

---

### Context Recall

**What it measures:** Did the retriever surface all the chunks that were
needed to fully answer the question?

**Analogy:** The same librarian brings only 5 books about GDPR fines,
but there are actually 20 relevant books in the library. Context recall = 5 / 20 = 0.25.
The answer might be incomplete because important source material was never retrieved.

**Why it matters:** Some questions require multiple articles working together.
"What are the obligations AND penalties for high-risk AI?" needs Article 9 (obligations)
AND Article 99 (penalties). If the retriever only finds one, the answer is incomplete.

---

## The code changes to `harness.py`

### `save()` method

```python
def save(self, path: Path) -> None:
    path.write_text(json.dumps(asdict(self), indent=2))
    logger.info("Saved RAGAS results to %s", path)
```

`asdict()` is from Python's `dataclasses` module. It converts a dataclass instance
into a plain dictionary. `json.dumps()` then converts that dictionary to a JSON string.
`indent=2` makes the JSON human-readable (with indentation).

This saves your scores permanently. Without `save()`, every time you run the evaluation
you would see the numbers on screen, but you would have no record of what
the score was before your last change. You need the saved file to prove your
Week 4 improvements actually improved anything.

### Why `is_acceptable()` has no parameter

```python
# Before (had a threshold parameter):
def is_acceptable(self, threshold: float = 0.7) -> bool:
    return self.faithfulness >= threshold and self.answer_relevancy >= threshold

# After (thresholds encoded in the function):
def is_acceptable(self) -> bool:
    return self.faithfulness >= 0.80 and self.answer_relevancy >= 0.70
```

Having `threshold` as a parameter was a leaky abstraction. The threshold is not
a caller concern — it is a business decision about what constitutes acceptable
legal AI. Encoding it in the function makes the business rule explicit and
prevents callers from accidentally lowering the bar.

The comment explains WHY faithfulness gets 0.80 instead of 0.70:

```python
# Faithfulness threshold is stricter (0.80) because hallucinated legal
# claims are more dangerous than generic chatbot errors.
```

---

## The run script — `scripts/run_evaluation.py`

The most interesting part is `make_pipeline_fn`:

```python
def make_pipeline_fn(agent):
    config = {"configurable": {"thread_id": "ragas-eval"}}

    def pipeline_fn(question: str) -> tuple[str, list[str]]:
        result = agent.invoke({"messages": [("human", question)]}, config=config)
        answer = result["messages"][-1].content
        contexts = [
            m.content for m in result["messages"]
            if hasattr(m, "type") and m.type == "tool"
        ]
        return answer, contexts or [answer]

    return pipeline_fn
```

**Why this wrapping pattern?**

RAGAS's `harness.run()` expects a simple function with the signature
`(question: str) -> (answer, contexts)`. But the agent is complex:
it is a LangGraph state machine that returns a whole state dictionary
with a list of messages.

`make_pipeline_fn` is an adapter. It wraps the complex agent interface
inside the simple signature RAGAS expects. This is the Adapter design pattern:
you do not change either side; you write a thin bridge between them.

**What are `contexts`?**

RAGAS needs the source text that was given to the LLM. In this system,
that is the tool call results — when the agent calls `search_regulations`,
the results come back as `ToolMessage` objects in the conversation history.

We find them with:
```python
contexts = [
    m.content for m in result["messages"]
    if hasattr(m, "type") and m.type == "tool"
]
```

This reads: "for every message in the final agent state, if it's a tool message,
grab its content". Those tool message contents are the regulation chunks
the LLM received — exactly what RAGAS needs to check faithfulness.

**The `or [answer]` fallback:**

If no tool messages exist (the agent answered from its training data without
calling any tools), we fall back to passing the answer itself as the context.
RAGAS can still compute answer_relevancy and context_recall in this case,
though faithfulness will be trivially 1.0 (the answer is "grounded" in itself).

---

## How to run the evaluation

```bash
python scripts/run_evaluation.py
```

This will:
1. Build the agent (loads the LLM, FAISS index, all tools)
2. Run every question in `evals/qa_pairs.json` through the agent (20 LLM calls)
3. Send all answers + contexts + ground truths to RAGAS (another LLM call per pair)
4. Print the scorecard
5. Save to `evals/baseline_scores.json`
6. Exit with code 1 if scores are below threshold (useful for CI)

**Expected cost:** ~20 × 2 calls = ~40 API calls total. With Groq free tier, this is fine.

---

## The test file — `tests/unit/test_harness.py`

No LLM needed here. We test the data structures in isolation.

```python
def test_is_acceptable_fails_on_low_faithfulness():
    result = RAGASResult(faithfulness=0.79, answer_relevancy=0.90, ...)
    assert not result.is_acceptable()
```

This tests the exact boundary: 0.79 is one tick below the 0.80 threshold.
This confirms the threshold is 0.80, not 0.7 or 0.85. If someone changes
the threshold in the code, this test will fail and force them to consciously
update the test too — making the change deliberate, not accidental.

```python
def test_save_writes_valid_json(tmp_path: Path):
    result = RAGASResult(...)
    out = tmp_path / "scores.json"
    result.save(out)
    data = json.loads(out.read_text())
    assert data["faithfulness"] == 0.85
```

`tmp_path` is a pytest built-in fixture. It gives each test a fresh temporary
directory that is automatically cleaned up after the test. This means the test
does real file I/O (not mocked) without leaving files on disk.

---

## What these scores tell you about your system

When you run this for the first time, you will see numbers.
Here is how to interpret them:

| Scenario | What it means | What to do |
|---|---|---|
| Faithfulness < 0.70 | The LLM is making up facts not in the retrieved text | Fix the system prompt to say "only answer from context" |
| Relevancy < 0.70 | The LLM is going off-topic | Check if the question type is too broad for your prompt |
| Precision < 0.50 | Your retriever returns too much noise | Reduce `retrieval_k` or improve chunking |
| Recall < 0.50 | Your retriever misses important chunks | Increase `retrieval_k` or improve embeddings |

Write down the exact numbers. They are your baseline. Every change you make
in Week 4 gets measured against them.
