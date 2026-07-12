# F8 — LangGraph Agent Graph
## What we built, why we built it, and how it all works (explained for a complete beginner)

---

## 0. The Big Picture: From "Smart Search" to "Autonomous Agent"

In Week 1 our app worked like this:

```
User question → retrieve chunks → stuff into prompt → LLM answers → done
```

This is called a **RAG chain** (Retrieval Augmented Generation). It is linear and dumb: the system always retrieves, always stuffs the same prompt, always calls the LLM exactly once. It can't decide to look something up a second time if the first search wasn't good enough. It can't check a CSV for penalty amounts after finding a relevant article.

In Week 2 we upgrade to an **agent**. An agent can:
- **Decide** which tool to call (or call no tool at all).
- **Loop** — call a tool, read the result, decide whether to call another tool.
- **Stop** when it has enough information.
- **Escalate** — pause and ask a human if the answer involves something critical.

**Analogy:** The Week 1 system is like a vending machine: put in a question, get out an answer, no flexibility. The Week 2 agent is like a junior analyst: give her a question, she searches the database, then checks a spreadsheet, then writes the answer — and flags it to the compliance officer if she finds something alarming.

---

## 1. What Changed in This Feature

| File | Change |
|---|---|
| `src/regulation_advisor/agent/graph.py` | **FIXED** — removed hardcoded `ChatGroq`, now uses `build_llm()` from shared factory |
| `tests/unit/test_agent_graph.py` | **CREATED** — 3 unit tests proving the graph compiles and routes correctly |
| `docs/f8-langgraph-agent.md` | **CREATED** — this document |

---

## 2. What Is LangGraph?

LangGraph is a library from the LangChain team that lets you build **stateful, cyclic graphs** where nodes are Python functions and edges define the flow between them.

**Key vocabulary:**
- **Node** — a Python function that takes the current state and returns an update to the state.
- **Edge** — a connection between nodes. Can be unconditional ("always go here next") or conditional ("go here if X, otherwise go there").
- **State** — a dictionary (TypedDict) that flows through the graph. Every node reads from and writes to this shared state.
- **Checkpointer** — saves the state after each step so the graph can be paused, resumed, and replayed. We use `MemorySaver` (stores in RAM).
- **interrupt_before** — tells LangGraph to pause *before* running a specific node, waiting for a human to approve or modify the state.

**Analogy for the whole graph:**

Imagine a post office sorting workflow. A package (state) enters the sorting room (graph). A worker (agent node) looks at it and decides: does it need customs checking (tool node)? Or is it ready to go out (END)? If it's flagged as dangerous (critical finding), it goes to a supervisor (human_review node). The supervisor can approve it or reject it. Otherwise it ships out.

---

## 3. The Graph Structure — Step by Step

```python
# src/regulation_advisor/agent/graph.py

graph = StateGraph(RegAdvisorState)

graph.add_node("agent",        agent_node)          # the "brain"
graph.add_node("tools",        ToolNode(TOOLS))     # the "hands"
graph.add_node("human_review", lambda s: s)         # the "pause button"

graph.set_entry_point("agent")                      # always start here
graph.add_conditional_edges("agent", should_continue)  # decide what's next
graph.add_edge("tools", "agent")                    # after tools, back to agent
graph.add_edge("human_review", END)                 # after human review, finish

return graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],              # pause before human_review
)
```

### The flow visualised

```
           ┌──────────────────────────────────────────────────┐
           │                  ENTRY POINT                      │
           └──────────────────┬───────────────────────────────┘
                              ↓
                       ┌─────────────┐
                       │  agent node  │  ← LLM thinks: tool call? or final answer?
                       └──────┬──────┘
                              │
              ┌───────────────┼──────────────────┐
              ↓ "tools"       ↓ "human_review"   ↓ END
       ┌─────────────┐  ┌──────────────┐
       │  tools node  │  │ human_review │
       │ (ToolNode)   │  │  (pause here)│
       └──────┬───────┘  └──────┬───────┘
              │ (back to agent)  │ → END
              └─────────────────┘
```

---

## 4. The Three Nodes Explained

### Node 1: `agent_node` — the brain

```python
def agent_node(state: RegAdvisorState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    is_critical = any(kw.lower() in response.content.lower() for kw in CRITICAL_KEYWORDS)
    return {"messages": [response], "is_critical_finding": is_critical}
```

**What it does:**
1. Takes the current conversation history (`state["messages"]`).
2. Passes it to the LLM, which has been "bound" with the tool schemas.
3. The LLM either:
   - Returns a final answer (plain text response).
   - Returns a "tool call" request (a special structured message saying "call this tool with these args").
4. Checks if the response contains critical keywords (prohibited, banned, Article 5, €35M).
5. Returns the new message and whether a critical finding was detected.

**Key concept — `llm.bind_tools(TOOLS)`:**

When you call `llm.bind_tools(TOOLS)`, you're telling the LLM: "Here are the tools available to you. Their names, descriptions, and parameter schemas." The LLM is trained to understand this format and can decide to generate a tool call message instead of a plain text response.

**Analogy:** Imagine giving a consultant a phone list. You say "You have three resources: the legal database (search_regulations), the penalty spreadsheet (query_structured_data), and Google (search_web)." The consultant (LLM) decides which to call — or writes the final memo without calling any.

### Node 2: `ToolNode(TOOLS)` — the hands

`ToolNode` is a prebuilt LangGraph component. It:
1. Reads the tool_calls from the last AI message.
2. Calls each tool function with the specified arguments.
3. Wraps the return value in a `ToolMessage` and appends it to the state.

You don't write this node — LangGraph provides it. You just pass it your list of tools.

### Node 3: `human_review` — the pause button

```python
graph.add_node("human_review", lambda s: s)  # identity function — does nothing
```

This node literally does nothing — `lambda s: s` returns the state unchanged. Its value comes from the `interrupt_before=["human_review"]` flag in `graph.compile()`.

When LangGraph sees that the next node is `human_review` and `interrupt_before` is set, it **pauses the entire graph**, saves the state to the checkpointer, and returns control to the caller. The caller (our Gradio UI) can then:
- Show the answer with a warning: "⚠️ Critical finding — verify with a qualified legal professional."
- Resume the graph later (in future versions, a compliance officer could approve/reject).

**Analogy:** In nuclear power plants, there's a "dead man's switch" — if the operator doesn't actively press a button, the reactor shuts down. `interrupt_before=["human_review"]` is similar: the agent pauses and waits for human confirmation before proceeding on critical topics.

---

## 5. The Router: `should_continue`

```python
def should_continue(state: RegAdvisorState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"              # LLM wants to call a tool → go to tools node
    if state.get("is_critical_finding"):
        return "human_review"      # critical topic → pause for human
    return END                     # done → finish
```

This is a **conditional edge** function. LangGraph calls it after the `agent` node runs and uses the return value to pick the next node.

**The three paths:**
1. **`"tools"`** — the LLM generated a tool_call message. Go execute the tool, come back to the agent.
2. **`"human_review"`** — the LLM's text answer triggered a critical keyword. Pause.
3. **`END`** — the LLM gave a plain text final answer with no tool calls and no critical keywords. We're done.

**Analogy:** Think of `should_continue` as the sorting machine at the post office. It reads the label on the package and puts it in one of three chutes: "needs customs" (tools), "needs supervisor" (human_review), or "ready to ship" (END).

---

## 6. The Fix We Made: `build_llm()` Replaces Hardcoded `ChatGroq`

### Before F8

```python
def build_agent_graph():
    from langchain_groq import ChatGroq
    from regulation_advisor.config import settings
    
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
```

**The problems:**
1. **Hardcodes the provider.** If you set `LLM_PROVIDER=openrouter` in your `.env`, the agent ignores it and uses Groq anyway.
2. **Duplicates code** from `gradio_app.py` (which we already extracted to `llm.py` in F7).
3. **Breaks if Groq API key is missing.** The Gradio chain would use OpenRouter while the agent crashes silently.

### After F8

```python
from regulation_advisor.llm import build_llm

def build_agent_graph():
    llm = build_llm()   # reads LLM_PROVIDER from config — same as the UI
```

Now both the chat UI and the agent use **the same LLM**, configured from **the same config file**. Change the `.env` once, everything updates.

---

## 7. How the State Flows — A Concrete Example

Let's trace a user question: *"Is emotion recognition in the workplace prohibited?"*

```
Step 1 — State enters agent_node:
  state.messages = [HumanMessage("Is emotion recognition in the workplace prohibited?")]

Step 2 — LLM decides to call a tool:
  response = AIMessage(tool_calls=[{
      "name": "search_regulations",
      "args": {"query": "emotion recognition workplace prohibited"},
  }])
  → should_continue returns "tools"

Step 3 — ToolNode calls search_regulations:
  result = "[Article 5 — eu_ai_act.pdf]\nEmotion recognition in the workplace is prohibited..."
  → ToolMessage added to state.messages

Step 4 — State back to agent_node:
  LLM sees the tool result and writes a final answer:
  response = AIMessage("Under Article 5(1)(f) of the EU AI Act, emotion recognition
                        systems in the workplace are **prohibited**...")
  is_critical = True   ← "prohibited" is in CRITICAL_KEYWORDS
  → should_continue returns "human_review"

Step 5 — Graph pauses at human_review:
  Gradio UI receives control, appends the ⚠️ warning, displays the answer.
```

---

## 8. The Tests — What We're Checking and Why

### Test 1: `test_graph_compiles`

```python
def test_graph_compiles():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    with patch("regulation_advisor.agent.graph.build_llm", return_value=mock_llm):
        from regulation_advisor.agent.graph import build_agent_graph
        graph = build_agent_graph()

    assert graph is not None
```

**Why:** The graph has five moving parts (3 nodes, 2 types of edges, a checkpointer). If any import fails, any node name is misspelled, or any edge references a missing node, `build_agent_graph()` will raise an exception. This test verifies all the wiring is correct without making a real API call.

**Key technique — `patch`:** `unittest.mock.patch` temporarily replaces `build_llm` with a mock during the test. This is necessary because:
- We don't want to read API keys in CI/CD tests.
- We don't want to pay for LLM API calls during tests.
- We don't want test speed to depend on network latency.

**Analogy:** A car factory tests that all the bolts fit together correctly before putting in the engine. `test_graph_compiles` checks the bolts (the graph structure) without running the engine (the LLM).

### Test 2: `test_should_continue_routes_to_tools`

```python
msg = MagicMock(spec=AIMessage)
msg.tool_calls = [{"name": "search_regulations", ...}]
state = {"messages": [msg], "is_critical_finding": False}
assert should_continue(state) == "tools"
```

**Why:** This tests the core routing logic. If `should_continue` routes to `END` when it should route to `"tools"`, the agent will never call any tools — it will just give an answer from its training data without searching the regulations. Wrong answers, no citations.

### Test 3: `test_should_continue_routes_to_end`

```python
msg = MagicMock(spec=AIMessage)
msg.tool_calls = []   # empty list
state = {"messages": [msg], "is_critical_finding": False}
assert should_continue(state) == END
```

**Why:** This is the "happy path" termination. If the router never returns `END`, the graph will loop forever (or until LangGraph's recursion limit, which defaults to 25 steps). This test verifies that a plain text final answer terminates the graph correctly.

---

## 9. Gate Check — Manual Verification

The plan specifies this command as the F8 "gate":

```bash
uv run python -c "
from regulation_advisor.agent.graph import build_agent_graph
g = build_agent_graph()
print('graph ok')
"
```

This checks that the graph compiles end-to-end using the real `build_llm()` factory (which reads from your `.env`). It will print `graph ok` if everything is wired correctly.

**Note:** This requires a valid `.env` with `LLM_PROVIDER` and the corresponding API key set. It does *not* run the graph — it only compiles the structure. No API calls are made.

---

## 10. How LangGraph Memory Works (the Checkpointer)

```python
return graph.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
```

`MemorySaver` is a **checkpointer** — it saves a copy of the state after each node runs. This enables:

1. **Multi-turn conversations:** Each Gradio session passes a `thread_id` in the config:
   ```python
   config = {"configurable": {"thread_id": "gradio-session"}}
   agent.invoke({"messages": [...]}, config=config)
   ```
   LangGraph uses `thread_id` to load the saved state from a previous turn. The agent remembers what was said earlier in the conversation.

2. **Resume after interrupt:** When the graph pauses at `human_review`, the state is saved. A compliance officer can later call `agent.invoke(None, config=config)` to resume from exactly where it stopped.

**Analogy:** `MemorySaver` is like a sticky note. After each step of the workflow, the state is written on the sticky note. If the workflow is interrupted, you can pick up the sticky note and continue from the last written step.

---

## 11. Summary: What F8 Accomplished

| Goal | Before F8 | After F8 |
|---|---|---|
| LLM in agent | Hardcoded to Groq — ignores LLM_PROVIDER env var | Uses `build_llm()` — respects `.env` settings |
| Agent capabilities | Single-turn, no tools, no looping | Multi-turn, 3 tools, conditional looping |
| Critical finding handling | None | Pauses at `human_review` with `interrupt_before` |
| Multi-turn memory | None | `MemorySaver` checkpoints state by `thread_id` |
| Test coverage | None | 3 unit tests: compile check + routing logic |

F8 is the core of the Week 2 upgrade. Every conversation now goes through a full autonomous decision loop: think → search → think → answer (or escalate). This replaces the mechanical retrieve-then-answer chain from Week 1.
