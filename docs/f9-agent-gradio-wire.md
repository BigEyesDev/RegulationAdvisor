# F9 — Wire the LangGraph Agent into Gradio
## What we built, why we built it, and how it all works (explained for a complete beginner)

---

## 0. The Upgrade at a Glance

In Week 1 the chat UI and the reasoning logic were entangled: `gradio_app.py`
fetched documents, formatted context, called the LLM, and returned the answer — all
in one flat `respond()` function.

In Week 2 the agent takes over all of that. `gradio_app.py` becomes a thin shell:

```
Week 1 — Gradio app is the "brain"            Week 2 — Gradio app is the "display"
─────────────────────────────────             ───────────────────────────────────────
User message                                  User message
    ↓                                              ↓
gradio_app.respond()                          gradio_app.respond()
    ├── retriever.search()      ←removed           └── agent.invoke()   ←one line
    ├── _format_context()       ←removed               (all reasoning inside agent)
    ├── chain.invoke()          ←removed
    └── return answer                          return answer + optional ⚠️ banner
```

F9 is the step where we make that swap — and strip all the now-dead code from
`gradio_app.py` and `app_runner.py`.

---

## 1. Files Changed

| File | Change |
|---|---|
| `src/regulation_advisor/ui/gradio_app.py` | **Rewritten** — 137 → 52 lines; `build_ui(agent)` replaces `build_ui(retriever)` |
| `src/regulation_advisor/ui/app_runner.py` | **Updated** — startup sequence wires retriever into tools then builds agent |
| `pyproject.toml` | **Bumped** `0.1.0 → 0.2.0` |
| `CHANGELOG.md` | **Updated** — Week 2 release entry |

---

## 2. Why We Deleted Three Functions from `gradio_app.py`

Before F9 the file had three helpers:

| Helper | What it did | Why it's gone |
|---|---|---|
| `_build_llm()` | Created the LLM | Already moved to `llm.py` in F7 |
| `_build_chain()` | Built `ChatPromptTemplate \| LLM \| StrOutputParser` | Agent handles all LLM calls internally |
| `_format_context()` | Glued retrieved chunks into a context string | Agent calls `search_regulations` tool — which already formats the output |

Keeping those functions would be **dead code** — code that is never called but sits in the file confusing readers and maintenance. Good engineering removes dead code aggressively.

**Analogy:** Imagine you hired a project manager (the agent). Before, the receptionist (Gradio) was also doing the research, filing, and writing. Now the project manager does all that. You fire the receptionist's research and filing skills — she just greets visitors and passes messages to the project manager.

---

## 3. The New `gradio_app.py` — Line by Line

```python
"""
Gradio UI — RegulationAdvisor v0.2

v0.1 (Week 1): simple RAG chain — retrieve chunks → stuff context → LLM.
v0.2 (Week 2): LangGraph agent — the agent decides when to call tools and
               how many times, and surfaces a warning on critical findings.
"""
from __future__ import annotations

import logging
import gradio as gr

logger = logging.getLogger(__name__)

_CRITICAL_WARNING = (
    "\n\n---\n⚠️ **Critical finding** — this topic involves prohibited practices "
    "or significant penalties. Verify with a qualified legal professional before acting."
)
```

**What changed:**
- Only two imports (`logging`, `gradio`). Compare to before: `langchain_core`, `pydantic-settings`,
  `ChatPromptTemplate`, `StrOutputParser`, `Retriever` — none of those are needed anymore.
- `_CRITICAL_WARNING` is a module-level constant (not buried inside a function). Constants
  at the module level are easy to find, test, and update without touching any logic.

```python
def build_ui(agent) -> gr.Blocks:
    """
    Build the Gradio ChatInterface around a compiled LangGraph agent.

    Args:
        agent: A compiled LangGraph graph (returned by build_agent_graph()).

    Returns:
        A gr.Blocks object ready for demo.launch().
    """

    def respond(message: str, history: list) -> str:
        config = {"configurable": {"thread_id": "gradio-session"}}
        result = agent.invoke({"messages": [("human", message)]}, config=config)
        answer = result["messages"][-1].content
        if result.get("is_critical_finding"):
            answer += _CRITICAL_WARNING
        logger.info("Answered query (%d chars, critical=%s)", len(answer), result.get("is_critical_finding"))
        return answer
```

### Understanding `respond()` step by step

**Step 1 — `config = {"configurable": {"thread_id": "gradio-session"}}`**

The `thread_id` is the agent's memory key. Every call with the same `thread_id`
resumes the same conversation. The `MemorySaver` checkpointer (set up in F8) stores
all previous messages under this key.

**Analogy:** Think of `thread_id` as a file folder label at a doctor's office.
Every time you visit, the receptionist pulls your folder. The doctor reads your history.
If you had a different name (different `thread_id`), you'd get a blank folder.

In a multi-user production app you would use a per-user UUID here. For our single-user
Gradio demo, `"gradio-session"` is fine.

**Step 2 — `result = agent.invoke({"messages": [("human", message)]}, config=config)`**

`agent.invoke()` runs the full LangGraph decision loop:
- The agent node sees the new human message.
- It decides whether to call tools (search, CSV, web).
- If it calls tools, it reads the results, decides again.
- When it has enough, it writes the final answer.
- The loop terminates (returns to us) when `should_continue` returns `END` or
  the graph pauses at `human_review`.

`result` is the final state dictionary of the agent — it has the full `messages`
list and the `is_critical_finding` flag.

**Step 3 — `answer = result["messages"][-1].content`**

The last message in the state is the agent's final answer (an `AIMessage`). We
extract its text content.

**Step 4 — `if result.get("is_critical_finding"): answer += _CRITICAL_WARNING`**

If the agent detected prohibited AI practices, high fines, or banned uses in its
answer, it set `is_critical_finding = True` in the state. We append the warning
banner so the user knows to get a lawyer.

---

## 4. The New `app_runner.py` — The Startup Sequence

The startup sequence is now a clean 6-step waterfall. Here it is with annotations:

```python
if __name__ == "__main__":
    # Imports at the call site — avoids circular import issues at module load time
    from regulation_advisor.agent.tools import set_retriever
    from regulation_advisor.agent.graph import build_agent_graph
    from regulation_advisor.ui.gradio_app import build_ui

    # Step 1 — Build FAISS index if it doesn't exist (no-op on repeat runs)
    _ensure_index()

    # Step 2 — Load FAISS + embedding model into a Retriever object
    retriever = _load_retriever()

    # Step 3 — Inject retriever into the tools module
    #           search_regulations() uses this to answer queries
    set_retriever(retriever)

    # Step 4 — Build the LangGraph agent (compiles the state graph)
    agent = build_agent_graph()

    # Step 5 — Wrap the agent in a Gradio UI
    demo = build_ui(agent)

    # Step 6 — Launch the web server
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
```

### Why imports are inside `__main__` (lazy imports)

Heavy packages like `sentence-transformers`, `faiss`, and `langgraph` take time to
import. Putting them inside `if __name__ == "__main__"` means they are only loaded
when the script is the entry point — not when you import `app_runner` in tests.

**Analogy:** You don't carry all your camping gear to work every day. You pack it
when you're actually going camping. Lazy imports pack heavy dependencies only when
you're actually running the server.

### The Dependency Injection pattern — why `set_retriever(retriever)`?

The FAISS retriever is expensive to build (~20 seconds, ~500 MB of embedding model).
We build it once in `app_runner.py` and inject it into `tools.py` via `set_retriever()`.

The alternative — building the retriever inside `tools.py` — would mean it rebuilds
every time a tool is called, or on every test run. Neither is acceptable.

**Pattern name:** This is called **Dependency Injection** (DI). The tool says "give
me a retriever" rather than "I'll build my own retriever." The caller (app_runner)
controls what retriever is used. This makes `tools.py` testable without a real FAISS index.

---

## 5. What `agent.invoke()` Actually Does Inside (Full Trace)

Let's trace the query: *"Is emotion recognition in the workplace prohibited under the EU AI Act?"*

```
1. gradio_app.respond() calls:
   agent.invoke({"messages": [("human", "Is emotion recognition...")]}, config)

2. LangGraph loads the saved state for thread_id="gradio-session" from MemorySaver.
   (Empty on first call; contains prior messages on follow-up questions.)

3. agent_node runs:
   llm_with_tools.invoke(state["messages"])
   → LLM decides: "I need to search the regulations."
   → Returns AIMessage with tool_calls=[{name: "search_regulations", args: {query: "emotion recognition workplace"}}]

4. should_continue sees tool_calls → routes to "tools" node.

5. ToolNode executes search_regulations("emotion recognition workplace"):
   → _retriever.search() returns Article 5(1)(f)
   → Returns ToolMessage: "[Article 5 — eu_ai_act.pdf]\nEmotion recognition systems in
      workplaces are prohibited under Annex III..."

6. agent_node runs again with the tool result in messages:
   → LLM writes final answer: "Under Article 5(1)(f) of the EU AI Act, the use of AI
      systems for emotion recognition in the workplace is **prohibited**..."
   → "prohibited" is in CRITICAL_KEYWORDS → is_critical_finding = True

7. should_continue sees is_critical_finding=True → routes to "human_review".

8. interrupt_before=["human_review"] → graph pauses, saves state, returns to caller.

9. respond() receives result:
   answer = "Under Article 5(1)(f)..." + _CRITICAL_WARNING
   → Gradio displays answer + ⚠️ banner
```

The whole loop — two LLM calls, one tool call, state checkpointing — happens inside
that single `agent.invoke()` line in `respond()`.

---

## 6. Why This Architecture Is Better Than the Week 1 Chain

| Concern | Week 1 (chain) | Week 2 (agent) |
|---|---|---|
| How many tool calls? | Always exactly 1 (retriever.search) | As many as needed — LLM decides |
| Multi-turn memory? | None — every message is independent | Yes — `thread_id` + `MemorySaver` |
| Critical finding detection? | None | Yes — `is_critical_finding` flag + warning |
| Add a new tool? | Requires changing `respond()` code | Add tool to `TOOLS` list in `graph.py` |
| `gradio_app.py` knows about retrieval? | Yes — tightly coupled | No — only knows about the agent |
| Lines of code in `gradio_app.py` | 137 | 52 |

The reduction from 137 to 52 lines is not just cosmetic. It means:
- Less to read.
- Less to test.
- Less to break when you change the retrieval strategy in Week 3.

---

## 7. The Benchmark Query You Should Run Manually

After starting the app (`uv run python src/regulation_advisor/ui/app_runner.py`),
paste this into the chat:

> *"My company deploys an AI system for employee emotion recognition in the workplace.
> What are our obligations under the EU AI Act?"*

**Expected behaviour:**
1. Agent calls `search_regulations` → finds Article 5(1)(f) (prohibited practices).
2. Agent calls `query_structured_data` → finds enforcement date 2025-02-02.
3. Agent writes a final answer citing Article 5(1)(f) and the enforcement date.
4. `is_critical_finding = True` because the word "prohibited" appears in the answer.
5. The ⚠️ warning banner appears below the answer.

If you see the warning banner, F9 is working correctly end-to-end.

---

## 8. Summary: What F9 Accomplished

| Goal | Before F9 | After F9 |
|---|---|---|
| `gradio_app.py` complexity | 137 lines, 3 helpers, 6 imports | 52 lines, 1 function, 2 imports |
| `build_ui()` signature | `build_ui(retriever)` — coupled to retrieval | `build_ui(agent)` — decoupled |
| Multi-turn memory | None | Yes — `thread_id` per session |
| Critical finding warning | None | Appended automatically |
| `app_runner.py` startup | Build chain, pass retriever to UI | Wire retriever → build agent → UI |
| Version | 0.1.0 | 0.2.0 |
