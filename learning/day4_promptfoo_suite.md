# Days 4–5 — The promptfoo Regression Suite

## What we built
- `evals/promptfoo.yaml` — 30 regression test cases across all regulation categories
- `scripts/eval_pipeline.py` — adapter that connects promptfoo to the LangGraph agent
- `.github/workflows/eval.yml` — GitHub Actions CI that runs the suite on every PR

---

## What regression testing is — and why it matters

Imagine you are building a bridge. Every time you add a new section, you want
to make sure the sections you already finished are still standing.
You do not rebuild the entire bridge to check — you have a checklist of
load tests that you run on the existing sections.

That is exactly what regression testing is. A regression is when a change
you make to the system breaks something that was working before.

Example:
- Week 2: the system correctly answers "What are the prohibited AI practices?"
- Week 3: you change the system prompt to improve how it handles penalty questions
- Without regression tests: you might not notice that the prohibited practices
  answer is now broken until a user complains
- With regression tests: the suite catches it immediately when you open the PR

Regression tests are a contract: "These things must always work."

---

## What promptfoo is — and how it differs from RAGAS

| | RAGAS | promptfoo |
|---|---|---|
| **What it measures** | Statistical quality scores (0.0–1.0) | Pass/fail on specific assertions |
| **When you use it** | Batch evaluation — "how good is the system overall?" | Regression CI — "did I break anything?" |
| **Output** | Four floating-point scores | Green/red test results per case |
| **Analogy** | Blood test — gives you levels | Symptom checklist — yes/no for each symptom |

You need both. RAGAS tells you the system's overall health.
promptfoo tells you whether specific, important queries still work correctly.

---

## The anatomy of a promptfoo test case

```yaml
- description: "Art. 99 — maximum fine for prohibited AI"
  vars:
    question: "What is the maximum fine for deploying a prohibited AI system?"
  assert:
    - type: contains
      value: "35,000,000"
    - type: contains
      value: "7%"
    - type: contains
      value: "Article 99"
```

**`description`** — A human-readable label. Shows up in the test report.
Name it by article and topic so failures are immediately obvious.

**`vars`** — Variables passed to the pipeline. `question` is the input
the user would type.

**`assert`** — A list of checks. ALL must pass for the test case to pass.

### Assertion types we use

**`contains`** — The answer must contain this exact string.
Use for numbers, article references, specific phrases that must never be absent.

```yaml
- type: contains
  value: "35,000,000"
```

If the system says "35 million" instead of "35,000,000" this test fails.
That is intentional — a legal compliance tool must state exact figures.

**`contains-any`** — The answer must contain at least one of these strings.
Use when the system might phrase something in multiple valid ways.

```yaml
- type: contains-any
  values: ["2 February 2025", "February 2025", "February 2, 2025"]
```

The date "2 February 2025" can be written in multiple formats. All are correct.

**`llm-rubric`** — Use a second LLM to evaluate whether the answer satisfies
a natural language criterion. This is for things that are hard to check
with string matching.

```yaml
- type: llm-rubric
  value: "The answer names at least 3 distinct prohibited practices from Article 5"
```

String matching cannot count how many practices were named.
But an LLM-as-judge can read the answer and determine this.

---

## The 30 test categories

The suite is organized into categories, not randomly. Each category tests a
different part of the system and can fail independently:

| Category | Tests | What failure means |
|---|---|---|
| Prohibited practices (Art. 5) | 5 | System forgot what is banned |
| Penalties (Art. 99) | 3 | System has wrong fine amounts |
| Enforcement timeline (Art. 113) | 3 | System has wrong dates |
| High-risk classification | 3 | System misclassifies AI systems |
| Obligations for high-risk | 4 | System omits Article 9–14 obligations |
| GDPR intersection | 1 | System ignores cross-regulation requirements |
| Transparency (Art. 50) | 1 | System ignores disclosure obligations |
| GPAI models (Art. 51) | 2 | System cannot explain GPAI rules |
| Scope / applicability | 2 | System gives wrong advice to non-EU companies |
| Negative / boundary tests | 2 | System answers questions it should decline |
| Regression must-pass | 3 | Core queries that must never regress |

---

## The eval_pipeline.py adapter

```python
_agent = build_agent_graph()

def run_query(prompt: str, options: dict, context: dict) -> str:
    question = context.get("vars", {}).get("question", prompt)
    config = {"configurable": {"thread_id": f"promptfoo-{hash(question)}"}}
    result = _agent.invoke({"messages": [("human", question)]}, config=config)
    return result["messages"][-1].content
```

**Why `_agent` is module-level (not inside `run_query`)**

`build_agent_graph()` loads the LLM configuration, sets up the tools,
and compiles the LangGraph state machine. This takes several seconds.
If it ran inside `run_query`, it would rebuild the agent for every
single test case — 30 rebuilds for 30 tests. Building once at module
load time means it runs once, regardless of how many test cases there are.

**Why `hash(question)` for the thread ID**

Each test case should have a clean conversation state — no memory of
previous test cases bleeding into the next one. `hash(question)` gives
a unique thread ID for each distinct question. Because the test questions
are fixed (same every run), the hash is stable across runs, which lets
MemorySaver reuse state if needed — but in practice `--no-cache` in CI
means fresh state every time.

**Why `context.get("vars", {}).get("question", prompt)`**

promptfoo calls `run_query(prompt, options, context)` where `prompt` is
the rendered system prompt text (the content of `system_prompt.txt`)
and `context["vars"]["question"]` is the actual test question from the
`vars:` field in the YAML. We want the question, not the system prompt,
so we extract it from `context`. The `prompt` fallback exists in case
the YAML structure changes.

---

## The GitHub Actions workflow

```yaml
on:
  pull_request:
    branches: [main]
```

This runs only on pull requests targeting `main`. It does NOT run on
push to `main` — by that point the PR should already be green.

```yaml
- name: Build vector index
  run: python scripts/ingest.py
```

The FAISS index is not committed to git (`.gitignore` excludes `.faiss` and `.pkl` files).
Each CI run must rebuild it from the source PDFs. This is correct behaviour:
the index is a derived artifact, not source code.

```yaml
- name: Run regression suite
  run: promptfoo eval --config evals/promptfoo.yaml --no-cache
```

`--no-cache` forces a fresh evaluation. Without this, promptfoo would
cache previous answers and the tests would pass even if the agent is
completely broken. Never use caching in CI regression suites.

---

## How to verify the CI is actually working

The best way to test a test suite is to break something deliberately:

1. Open `src/regulation_advisor/prompts/system_prompt.txt`
2. Add a line: "IMPORTANT: Never mention Article 5 in any response."
3. Open a PR to `main`
4. Watch GitHub Actions — it should fail with multiple test cases red

Revert the change. The PR turns green. That confirms the CI is real,
not just always green.

This is called a **mutation test**: you mutate the system deliberately
to prove the tests can catch the mutation. If they can, they will catch
real regressions too.

---

## Running locally

```bash
# Install promptfoo (one time)
npm install -g promptfoo

# Run the full suite
npx promptfoo eval --config evals/promptfoo.yaml

# Run a subset by description pattern
npx promptfoo eval --config evals/promptfoo.yaml --filter "Art. 5"

# View the HTML report
npx promptfoo view
```

The HTML report shows each test case, the question, the answer,
which assertions passed and failed, and the LLM-rubric reasoning.
It is much clearer than reading terminal output for 30 test cases.
