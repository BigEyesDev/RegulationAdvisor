# F7 — Agent State + Tools
## What we built, why we built it, and how it all works (explained for a complete beginner)

---

## 0. The Big Picture Before We Start

Imagine you hire a new research assistant for a law firm. On Day 1 you give her three things:

1. **A database** of regulation texts she can search (EU AI Act, GDPR).
2. **A spreadsheet** of timelines and penalty amounts.
3. **Internet access** so she can look up the latest enforcement news.

Before she can help anyone, though, you need to make sure she actually knows *how* to use each of these resources. That is exactly what **F7** does for our AI agent: it defines the agent's *tools* (her "resources") and makes sure they work correctly.

---

## 1. What Changed in This Feature (one-line summary per file)

| File | Change |
|---|---|
| `src/regulation_advisor/llm.py` | **CREATED** — shared LLM factory so both the chat UI and the agent use the same code to start the language model |
| `src/regulation_advisor/ui/gradio_app.py` | **UPDATED** — removed the copy-paste `_build_llm()` function; now imports from `llm.py` |
| `src/regulation_advisor/agent/tools.py` | **FIXED** — corrected a path bug that would have made the CSV search silently fail |
| `tests/unit/test_tools.py` | **CREATED** — automated checks to prove the tools actually work |

---

## 2. Why We Have a "Shared LLM Factory" (the `llm.py` file)

### The problem: DRY violation

Before F7 the code that starts the language model was copy-pasted twice:
- once inside `gradio_app.py` (the chat UI)
- once inside `graph.py` (the LangGraph agent — we fix this in F8)

**Analogy:** Imagine your company has an IT policy for how to configure laptops. If you write that policy in two separate Word documents and they drift out of sync, one team ends up with outdated settings. You want *one* canonical document everyone reads.

In software the principle is called **DRY — Don't Repeat Yourself**. When you need to change how the LLM is configured (e.g., switch from Groq to OpenRouter), you change it in *one* place and every part of the codebase automatically gets the update.

### The solution: a factory function

```python
# src/regulation_advisor/llm.py

def build_llm():
    """
    Reads LLM_PROVIDER from the environment and returns the correct
    LangChain chat model object.
    """
    provider = settings.llm_provider   # e.g. "openrouter", "groq", "google"
    model    = settings.llm_model      # e.g. "deepseek/deepseek-v4-flash"

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, base_url=..., api_key=...)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=...)

    # default: groq
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=...)
```

**The key concept — a "factory function":**
A factory function is a function whose job is to *create and return an object*. You call `build_llm()` and it hands you back a ready-to-use language model, whatever kind the config asks for. The caller does not need to know *which* provider is being used. It just uses whatever it receives.

**Analogia:** A car rental desk (factory) gives you a car. You don't care whether it's a Ford or a Toyota — you drive it the same way. If the rental desk switches from Ford to Toyota, you notice nothing.

### The update to `gradio_app.py`

Before:
```python
# gradio_app.py — old, duplicated code
def _build_llm():
    provider = settings.llm_provider
    ...  # 15 lines copied from the factory

def _build_chain():
    llm = _build_llm()   # local private function
```

After:
```python
# gradio_app.py — new, imports the shared factory
from regulation_advisor.llm import build_llm

def _build_chain():
    llm = build_llm()   # ← one import, zero duplication
```

---

## 3. The Three Agent Tools — What They Are

A LangGraph "tool" is a Python function decorated with `@tool` from LangChain. The decorator tells the LLM *"you can call this function by name"*. Think of it as giving the LLM a button panel:

```
 ┌──────────────────────┐  ┌─────────────────────┐  ┌───────────────┐
 │  search_regulations  │  │ query_structured_data│  │  search_web   │
 │  (semantic search)   │  │ (CSV lookup)         │  │ (Tavily API)  │
 └──────────────────────┘  └─────────────────────┘  └───────────────┘
```

When the LLM decides it needs information, it "presses a button" (calls a tool), gets back text, and uses that text to write the final answer.

### Tool 1: `search_regulations` — semantic document search

```python
@tool
def search_regulations(query: str) -> str:
    """Search EU AI Act and GDPR documents for relevant articles and provisions."""
    if _retriever is None:
        return "Error: retriever not initialised. Call set_retriever() first."
    result = _retriever.search(query, k=5)
    return "\n\n---\n\n".join(
        f"[Article {c.article_number} — {c.source_document}]\n{c.content}"
        for c in result.chunks
    )
```

**What it does:** Uses the FAISS vector index (built in Week 1) to find the 5 most relevant regulation paragraphs for a given query.

**Analogy:** Like Ctrl+F in a PDF, but smart — it understands meaning, not just exact words. You search "emotion recognition at work" and it finds Article 5(1)(f) even if the article doesn't use those exact words.

**Key pattern — `_retriever` module-level variable:**
```python
_retriever = None   # starts empty

def set_retriever(r):
    global _retriever
    _retriever = r  # wired in at startup
```

The retriever is not embedded inside the tool because building the FAISS index takes ~20 seconds. We build it once at startup and inject it. This is the **Dependency Injection** pattern — the tool doesn't build its own dependencies, it receives them from outside.

### Tool 2: `query_structured_data` — CSV row search

```python
@tool
def query_structured_data(question: str) -> str:
    """Query structured regulation data: timelines, penalties, risk classifications."""
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    ...
```

**What it does:** Keyword-searches three CSV files and returns matching rows:
- `ai_act_timeline.csv` — enforcement dates (e.g., "prohibited AI: 2025-02-02")
- `risk_classification.csv` — which AI use-cases are high-risk / prohibited
- `penalty_structure.csv` — fine amounts (up to €35M or 7% of global turnover)

**The bug we fixed — `Path("data")` vs absolute path:**

Before the fix:
```python
data_dir = Path("data")   # WRONG — relative to wherever you run Python from
```

If you run pytest from the project root, `Path("data")` works. If you run it from `/tmp` or any other directory, the CSV files will not be found and the function silently returns "No matching structured data found." — which looks like a real answer, not an error!

After the fix:
```python
data_dir = Path(__file__).parent.parent.parent.parent / "data"
# __file__ = .../src/regulation_advisor/agent/tools.py
# .parent   = .../src/regulation_advisor/agent/
# .parent   = .../src/regulation_advisor/
# .parent   = .../src/
# .parent   = .../ (project root)
# / "data"  = .../data/
```

**Analogy:** The old path was like writing "my bedroom" on a delivery address — fine if you're already home, useless if you're not. The new path is a full street address. Works from anywhere.

**How `Path(__file__)` works:**
Every Python module has a built-in variable `__file__` that holds the *absolute path* to that `.py` file itself. `.parent` navigates up one directory level. Chaining `.parent` four times goes up four levels. This is idiomatic Python for "find something relative to *this file*".

### Tool 3: `search_web` — live internet search via Tavily

```python
@tool
def search_web(query: str) -> str:
    """Search the web for recent EU AI Act enforcement news and official guidance."""
    from tavily import TavilyClient
    from regulation_advisor.config import settings
    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(query, max_results=3)
    return "\n\n".join(r["content"] for r in results.get("results", []))
```

**What it does:** Calls the Tavily API (a search engine designed for LLMs) and returns the top 3 web page excerpts.

**Why it's needed:** The EU AI Act is new and still being enforced. Regulations change. An agent that can only look at the PDF from 2024 might miss a 2025 enforcement decision. The web search keeps answers current.

---

## 4. Agent State — What `state.py` Does

Every LangGraph agent maintains a *state dictionary* that flows through the graph. Think of it as a shared notepad that every node in the graph can read and write.

```python
# src/regulation_advisor/agent/state.py

CRITICAL_KEYWORDS = ["prohibited", "banned", "Article 5", "35,000,000", "7%", "illegal"]

class RegAdvisorState(TypedDict):
    messages:           list[BaseMessage]   # the full conversation history
    retrieved_chunks:   list[RegulationChunk]
    tools_used:         list[str]
    confidence_score:   float
    is_critical_finding: bool               # triggers human review
```

**`CRITICAL_KEYWORDS`:** If the LLM's answer contains any of these words, `is_critical_finding` is set to `True`. This causes the graph to route to a `human_review` node (a pause point) so a compliance officer can verify before the answer is shown to the user.

**Analogy:** Think of it like a hospital triage system. Most patients go through normal processing. But if a nurse detects a keyword ("chest pain", "unconscious"), they get routed to the emergency track. `CRITICAL_KEYWORDS` is the regulation equivalent: anything about prohibitions or €35M fines should be double-checked by a human.

---

## 5. The Tests — Why We Write Them and How to Read Them

### What a unit test is (for beginners)

A unit test is a small piece of code that:
1. Sets up a specific scenario.
2. Calls the function you want to test.
3. Checks that the output matches what you expect.
4. If it doesn't match, the test fails loudly.

**Analogy:** When a car factory builds a door, a machine tests that the door opens and closes correctly before it goes on the car. The test is automated and runs every time. If something breaks in the factory, the machine catches it before the door ends up on a car.

### Test 1: `test_search_regulations_returns_article`

```python
def test_search_regulations_returns_article():
    # 1. BUILD a fake retriever (we don't want to load the real FAISS index in tests)
    fake_chunk = _make_chunk("5", "eu_ai_act.pdf", "Prohibited practices include…")
    fake_retriever = MagicMock()
    fake_retriever.search.return_value = _make_retriever_result(fake_chunk)

    # 2. INJECT it into the tools module
    tools_module._retriever = fake_retriever

    # 3. CALL the tool
    result = tools_module.search_regulations.invoke({"query": "prohibited AI"})

    # 4. ASSERT the result is formatted correctly
    assert "Article 5" in result
    assert "Prohibited practices" in result
```

**Key concept — `MagicMock`:** A mock is a fake object that pretends to be the real thing. We use `MagicMock()` to create a fake retriever so the test doesn't need the actual FAISS index file on disk. This makes the test fast (milliseconds instead of 20 seconds) and reliable (no external files needed).

### Test 2: `test_query_structured_data_returns_rows`

This test uses the *real* CSV files in `data/`. It searches for "prohibited" and checks that the `Unacceptable` risk tier row is returned (we know it's there — it mentions "prohibited").

**Why use real files here?** Because the tool's logic depends on the actual CSV structure (column names, row format). A fake CSV might miss structural problems. This is an acceptable trade-off: the CSV files are committed to the repo, so the test is still deterministic.

### Test 3: `test_search_regulations_without_retriever_returns_error_string`

```python
def test_search_regulations_without_retriever_returns_error_string():
    tools_module._retriever = None    # simulate cold start
    result = tools_module.search_regulations.invoke({"query": "anything"})

    assert isinstance(result, str)
    assert "Error" in result or "retriever" in result.lower()
```

**Why this matters:** If `_retriever` is `None` and the tool raised an exception, the entire LangGraph agent would crash. Instead we designed the tool to return a string error message. The LLM sees that error string, understands something went wrong, and can tell the user "I can't search right now" instead of the whole application going down.

**This is called defensive programming** — writing code that fails gracefully rather than explosively.

---

## 6. How to Run the Tests

```bash
# From the project root
uv run pytest tests/unit/test_tools.py -v
```

Expected output:
```
tests/unit/test_tools.py::test_search_regulations_returns_article       PASSED
tests/unit/test_tools.py::test_query_structured_data_returns_rows       PASSED
tests/unit/test_tools.py::test_search_regulations_without_retriever_returns_error_string PASSED

3 passed in 0.15s
```

---

## 7. How Everything Connects (the import chain)

```
app_runner.py
    ↓ imports
gradio_app.py
    ↓ imports
llm.py  ← NEW (was _build_llm inside gradio_app.py)
    ↓ reads from
config.py (reads from .env)

app_runner.py
    ↓ calls
set_retriever(retriever)   ← wires retriever into tools module
    ↓ makes available to
tools.py (search_regulations)
```

After F7, both `gradio_app.py` and `graph.py` (fixed in F8) share a single `build_llm()` function. There is zero duplicated LLM configuration code.

---

## 8. Summary: What F7 Accomplished

| Goal | Before F7 | After F7 |
|---|---|---|
| LLM factory code | Duplicated in `gradio_app.py` | Single `llm.py` module |
| CSV path in `tools.py` | Relative → breaks outside project root | Absolute → always works |
| Test coverage for tools | None | 3 unit tests, all passing |
| Agent tool correctness verified | Manually | Automatically on every commit |

F7 is infrastructure work — it doesn't add visible features but ensures everything that F8 and F9 build on top of is solid and testable.
