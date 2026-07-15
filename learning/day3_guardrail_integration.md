# Day 3 — Guardrail Integration

## What we built
- `src/regulation_advisor/evaluation/guardrails.py` — (already existed from Week 2 skeleton; this day wires it in)
- `src/regulation_advisor/ui/gradio_app.py` — updated to run guardrails after every streamed answer
- `tests/unit/test_guardrail_integration.py` — 5 tests covering the article extraction and chain behaviour

---

## The problem: AI systems can say wrong things confidently

An AI system that answers compliance questions has a specific danger:
it can state legal requirements with total confidence — correct grammar,
correct tone, wrong facts.

Example of a dangerous hallucination:

> "Under Article 42, employers must perform a conformity assessment before
> deploying emotion recognition software."

Article 42 is about conformity assessments — but it applies to high-risk AI systems
in general, not specifically to emotion recognition. More critically, the system
might cite an article that is nowhere in the regulation. If a compliance officer
acts on this, their company could face a fine for failing to follow the actual requirements.

The guardrail layer is the last line of defence before the answer reaches the user.

---

## What a guardrail chain is — the analogy

Think of an airport security line. You pass through multiple checkpoints:

1. **Boarding pass check** — Does this person have a valid ticket?
2. **X-ray machine** — Is there anything dangerous in the bag?
3. **Metal detector** — Is the person carrying metal?

Each checkpoint is independent. Failing any one of them stops you.
Passing all of them means you can board.

The guardrail chain works the same way. Each handler checks one specific thing.
If a check fails, the answer is flagged and no further checks run.
If all checks pass, the answer reaches the user.

```
FaithfulnessCheck → CitationVerificationCheck → LegalClaimFlagCheck
```

The first two can **block** (answer is not delivered as-is).
The third only **annotates** (adds a disclaimer but never blocks — users should always
see a legal disclaimer even on good answers).

---

## The three guardrail handlers

### FaithfulnessCheck

```python
class FaithfulnessCheck(GuardrailHandler):
    def __init__(self, threshold: float = 0.7, next_handler=None):
        super().__init__(next_handler)
        self._threshold = threshold

    def check(self, answer, chunks, confidence):
        if confidence < self._threshold:
            return GuardrailResult(passed=False, ...)
        return self._pass_to_next(answer, chunks, confidence)
```

This check uses the `confidence` score to decide if the answer is reliable.
In batch evaluation (RAGAS), confidence = faithfulness score from the RAGAS metric.
In the live UI, we pass `confidence=1.0` because we do not have a real-time
RAGAS score — this effectively skips the faithfulness check in production.
(The faithfulness check runs in batch evaluation via `scripts/run_evaluation.py`.)

### CitationVerificationCheck

```python
class CitationVerificationCheck(GuardrailHandler):
    def check(self, answer, chunks, confidence):
        cited = set(re.findall(r"Article\s+(\d+)", answer, re.IGNORECASE))
        available = {c.article_number for c in chunks}
        hallucinated = cited - available
        result = self._pass_to_next(answer, chunks, confidence)
        if hallucinated:
            result.passed = False
            result.warnings.append(...)
        return result
```

This is the most important runtime check.

`cited` = all article numbers the LLM mentioned in its answer.
`available` = all article numbers that were present in the retrieved text.
`hallucinated` = cited - available = articles the LLM mentioned that it never actually read.

Set subtraction in Python: `{5, 42} - {5, 9, 14}` = `{42}`. Article 42 was cited but never retrieved.

### LegalClaimFlagCheck

```python
class LegalClaimFlagCheck(GuardrailHandler):
    def check(self, answer, chunks, confidence):
        result = self._pass_to_next(answer, chunks, confidence)
        if any(p in answer.lower() for p in LEGAL_CLAIM_PHRASES):
            result.warnings.append("⚠️ This answer contains legal claims...")
        return result
```

This handler never blocks. It just adds a disclaimer when the answer uses
phrases like "you must", "it is illegal", "the fine is". These phrases indicate
the system is giving direct legal advice, which it is not qualified to give.

---

## The Chain of Responsibility design pattern

This is a classic software design pattern. Instead of one big `check_everything()`
function with nested if-statements, each check is its own class that:

1. Does its check
2. Either returns early (blocking) or calls `self._next.check(...)` to pass
   control to the next handler

Why this structure?

**Adding a new check is trivial.** Want to add a "length check" that warns if
the answer is under 50 words? Create `LengthCheck(GuardrailHandler)`,
implement `check()`, and add it to `build_guardrail_chain()`. Zero changes
to the other handlers.

**Removing a check is trivial.** Edit `build_guardrail_chain()`. Done.

**Testing each check is trivial.** Each handler is independent. You can test
`CitationVerificationCheck` without involving `FaithfulnessCheck` at all.

**Compare this to the alternative:**

```python
# Bad: one function, hard to extend, hard to test
def check_answer(answer, chunks, confidence):
    if confidence < 0.7:
        return "blocked: low faithfulness"
    cited = re.findall(...)
    if hallucinated:
        return "blocked: hallucinated citation"
    if any legal phrases:
        return "warning: legal claim"
    return "ok"
```

As you add more checks this function grows and becomes hard to maintain.
The chain pattern keeps each rule in its own class, making the codebase
grow in a healthy, organized way.

---

## How the UI wiring works

The tricky part: how do we get the retrieved chunks to run the citation check,
when the UI only shows the streamed LLM response?

The answer is the LangGraph **checkpointer**. After `agent.stream()` finishes,
the entire conversation history for this thread — including all tool call results —
is stored in `MemorySaver`. We can read it back:

```python
def _context_chunks_from_state(agent, config: dict) -> list[RegulationChunk]:
    state = agent.get_state(config)
    tool_texts = " ".join(
        m.content for m in state.values.get("messages", [])
        if hasattr(m, "type") and m.type == "tool"
    )
    article_numbers = set(re.findall(r"Article\s+(\d+[a-z]?)", tool_texts, re.IGNORECASE))
    return [
        RegulationChunk(content="", article_number=a, article_title="", source_document="")
        for a in article_numbers
    ]
```

Step by step:

1. `agent.get_state(config)` — loads the full state for this thread from MemorySaver.
   `config` contains the `thread_id` that identifies which conversation we want.

2. `state.values.get("messages", [])` — gets all messages (human, AI, and tool messages).

3. The filter `m.type == "tool"` — finds only the tool response messages.
   When the agent calls `search_regulations("prohibited AI practices")`,
   the tool returns a string containing the retrieved regulation text.
   That string is stored as a ToolMessage in the conversation history.

4. `re.findall(r"Article\s+(\d+[a-z]?)", tool_texts, re.IGNORECASE)` —
   extracts all article numbers mentioned in the retrieved text.
   For example, if the tool returned "[Article 5 — eu_ai_act.pdf]\n...",
   this finds "5".

5. We build lightweight `RegulationChunk` objects with just the article number
   (content is empty because we only need the number for the citation check).

The result: we know exactly which articles were actually retrieved.
If the LLM's answer cites "Article 42" but only "Article 5" was retrieved,
the citation check will flag it.

---

## The respond() function after this change

```python
partial = ""
for chunk, _ in agent.stream(...):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        partial += chunk.content
        yield partial          # user sees tokens as they arrive

# Streaming is done. Now inspect the complete answer.
chunks = _context_chunks_from_state(agent, config)
guard = _guardrails.check(partial, chunks, confidence=1.0)
if guard.warnings:
    yield partial + "\n\n" + "\n\n".join(guard.warnings)
```

**Why yield inside the loop and then again at the end?**

Gradio's `ChatInterface` with a generator function works like this:
each time `yield` is called, Gradio replaces what it currently shows
with the new value. Yielding inside the loop gives the user the live
typing effect. Yielding at the end (with warnings appended) updates
the final displayed text to include the guardrail warnings.

**Why `confidence=1.0`?**

We do not have a real faithfulness score at inference time.
The faithfulness metric requires an LLM-as-judge call (what RAGAS does),
which would add latency to every user message. So in the live UI,
we pass `1.0` which makes `FaithfulnessCheck` always pass.
The citation and legal claim checks still run and still provide value.
Real faithfulness scoring happens in batch mode via `run_evaluation.py`.

---

## Why this is guarding `try/except` around `get_state`

```python
try:
    state = agent.get_state(config)
    ...
except Exception:
    return []
```

`get_state` can fail if the thread has no state yet (first message in a session)
or if MemorySaver has been wiped. When it fails, we return an empty chunk list.
With an empty list, `available = {}`, so `hallucinated = cited` — every cited article
looks hallucinated. That would wrongly flag every first message.

Returning `[]` early prevents this. The citation check is skipped for messages
where we cannot read the state, which is safe: we just lose the guardrail for
that message rather than producing a false positive.

---

## The test file — `tests/unit/test_guardrail_integration.py`

The test file extracts the article parsing logic into a plain function
(not dependent on agent/gradio/network) and tests it directly:

```python
def _parse_articles_from_tool_text(tool_text: str) -> list[RegulationChunk]:
    article_numbers = set(re.findall(r"Article\s+(\d+[a-z]?)", tool_text, re.IGNORECASE))
    return [RegulationChunk(...) for a in article_numbers]
```

This is a key testing principle: extract the testable logic from the
untestable infrastructure (agent, Gradio). The logic is a pure function
with no side effects. You can run it thousands of times in a test suite
without any API calls.

```python
def test_guardrail_blocks_when_cited_article_not_in_context():
    chunks = _parse_articles_from_tool_text("Article 5 — prohibited practices...")
    result = chain.check("Under Article 99, the fine is EUR 35M.", chunks, confidence=1.0)
    assert not result.passed
```

This is the most important test. It proves that if the LLM cites Article 99
but only Article 5 was in the retrieved context, the guardrail blocks it.
