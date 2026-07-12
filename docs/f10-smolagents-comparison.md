# F10 — smolagents Comparison Agent
## What we built, why we built it, and how it all works (explained for a complete beginner)

---

## 0. Why Build the Same Agent Twice?

At the end of Week 2 we have a working LangGraph agent. So why build another one
in smolagents?

Two reasons:

1. **Learning by contrast.** You understand a tool much better when you compare it
   to an alternative. The differences reveal the design philosophy of each framework.

2. **LinkedIn Article #3.** The whole point is to write a technical article: "I built
   the same agent in two frameworks — here's what I learned." You can't write that
   article without actually doing the build.

Think of it like learning to cook: you don't really understand why a recipe calls
for butter until you try it with oil and see how the texture changes.

---

## 1. Files Changed

| File | Change |
|---|---|
| `src/regulation_advisor/agent/smolagents_agent.py` | **CREATED** — `build_smolagents_agent()` factory |
| `pyproject.toml` | Added `smolagents[litellm]` dependency |
| `requirements.txt` | Added `smolagents[litellm]` dependency |
| `docs/smolagents_comparison.md` | **Completed** — benchmark results, code comparison, production guide |

---

## 2. What Is smolagents?

smolagents is a minimalist agent framework from HuggingFace. Its design philosophy
is the opposite of LangGraph: **minimal surface area, zero boilerplate**.

Where LangGraph asks you to:
- Define a state schema (TypedDict)
- Define named nodes (Python functions)
- Define a graph (StateGraph)
- Define routing logic (should_continue)
- Compile the graph
- Call `.invoke()`

smolagents asks you to:
- List your tools
- Call `.run("your question")`

That's it.

**Analogy:** LangGraph is like assembling IKEA furniture with a full instruction
manual — you control every screw and know exactly where each part goes. smolagents
is like hiring a handyman — you say "put the bookshelf here" and he figures out the
screws himself.

Both get you a bookshelf. The difference is control vs convenience.

---

## 3. The Two Mental Models: Graph vs Loop

### LangGraph — Explicit Directed Graph

```
      [agent_node]
           │
    ┌──────┼──────────┐
    ↓      ↓           ↓
 [tools] [human_review] [END]
    │
    └──→ [agent_node]  (loop back)
```

You draw the graph. You write the routing. You control every transition.

### smolagents — Implicit Think-Act-Observe Loop

```
    ┌─────────────────────────────────────┐
    │  1. THINK: What tool should I call? │
    │  2. ACT:   Call the tool            │
    │  3. OBSERVE: Read the result        │
    │  4. Repeat until done (max_steps)   │
    └─────────────────────────────────────┘
```

smolagents runs this loop internally. You don't see it or control it directly.

**When the implicit loop is fine:** Single-turn questions where you just want an answer.

**When you need the explicit graph:** Multi-turn conversations, human approval steps,
different handling for different risk tiers.

---

## 4. The `smolagents_agent.py` Code — Explained Line by Line

```python
from smolagents import LangChainTool, LiteLLMModel, ToolCallingAgent

from regulation_advisor.agent.tools import query_structured_data, search_regulations, search_web
from regulation_advisor.config import settings
```

**Three imports from smolagents:**
- `LangChainTool` — wraps a LangChain `@tool` so smolagents can call it.
- `LiteLLMModel` — a model wrapper that talks to 100+ LLM providers via [LiteLLM](https://github.com/BerriAI/litellm).
- `ToolCallingAgent` — the agent class. Runs the think-act-observe loop.

**Why `LangChainTool` instead of rewriting tools in smolagents format?**

smolagents has its own `@tool` decorator. You could rewrite `search_regulations`,
`query_structured_data`, and `search_web` using it. But that would mean:
- 61 lines of tool logic copied and maintained in two places.
- A DRY violation (introduced in F7, immediately broken in F10).

`LangChainTool` is an adapter — it reads the existing LangChain tool's name,
description, and parameter schema, and exposes it to smolagents. Zero duplication.

**Analogy:** Imagine your company uses US plugs and you hire a contractor from Europe
with European tools. Instead of buying all-new US tools, you give him an adapter.
`LangChainTool` is the adapter.

---

```python
_LITELLM_PREFIXES = {
    "openrouter": "openrouter",
    "google": "gemini",
    "groq": "groq",
}

def _litellm_model_id() -> str:
    """Map LLM_PROVIDER / LLM_MODEL from settings to a LiteLLM model identifier."""
    prefix = _LITELLM_PREFIXES.get(settings.llm_provider, settings.llm_provider)
    return f"{prefix}/{settings.llm_model}"
```

LiteLLM uses a `provider/model` string format. Our settings store `llm_provider`
and `llm_model` separately, so we join them with a slash.

Examples:
- `llm_provider=openrouter`, `llm_model=deepseek/deepseek-v4-flash`
  → `"openrouter/deepseek/deepseek-v4-flash"`
- `llm_provider=groq`, `llm_model=llama-3.3-70b-versatile`
  → `"groq/llama-3.3-70b-versatile"`
- `llm_provider=google`, `llm_model=gemini-2.5-flash`
  → `"gemini/gemini-2.5-flash"`

The fallback `settings.llm_provider` handles any future providers not in the dict.

---

```python
def build_smolagents_agent() -> ToolCallingAgent:
    model = LiteLLMModel(model_id=_litellm_model_id())
    tools = [
        LangChainTool(search_regulations),
        LangChainTool(query_structured_data),
        LangChainTool(search_web),
    ]
    logger.info("Building smolagents agent: model=%s tools=%d", model.model_id, len(tools))
    return ToolCallingAgent(tools=tools, model=model, max_steps=5)
```

**`max_steps=5`:** The maximum number of tool calls the agent can make before
stopping. This is a safety guard — without it, a confused agent could loop
indefinitely. In the LangGraph version the equivalent is LangGraph's built-in
`recursion_limit` (default 25). We set 5 because our queries rarely need more
than 2-3 tool calls.

**Signature `build_smolagents_agent() -> ToolCallingAgent`:** Returns a concrete
type, not `Any`. This makes the code self-documenting and helps your IDE provide
autocompletion for `.run()`.

---

## 5. How to Use the Agent

```python
from regulation_advisor.agent.tools import set_retriever
from regulation_advisor.agent.smolagents_agent import build_smolagents_agent

# Must call this first — same as for the LangGraph agent
set_retriever(your_retriever)

agent = build_smolagents_agent()
answer = agent.run("What are the prohibited AI practices under Article 5?")
print(answer)
```

**Compared to the LangGraph agent:**

```python
# LangGraph
config = {"configurable": {"thread_id": "session-1"}}
result = langgraph_agent.invoke({"messages": [("human", question)]}, config)
answer = result["messages"][-1].content

# smolagents
answer = smolagents_agent.run(question)   # one line — no config, no state unwrapping
```

The smolagents call is simpler. The LangGraph call gives you access to the full
state (tool call history, is_critical_finding, all messages) and persistent memory.

---

## 6. The LiteLLM Layer — How It Works

LiteLLM is not an LLM itself. It is a **translation layer** that converts
OpenAI-format API calls to whatever format each provider uses.

```
Your code calls:  LiteLLMModel.chat(messages, tools)
                         │
                         ↓
LiteLLM translates to:   Groq format    / OpenRouter format   / Google format
                         │                │                      │
                         ↓                ↓                      ↓
                     Groq API         OpenRouter API         Gemini API
```

**Why does this matter?** LangChain has separate packages for each provider
(`langchain-groq`, `langchain-openai`, `langchain-google-genai`). LiteLLM gives
you all providers in one package with one API. smolagents uses LiteLLM so you
only need to install `smolagents[litellm]`.

**Analogy:** LiteLLM is like a universal remote control. Instead of having a
separate remote for your TV, soundbar, and streaming box, one remote controls all
three. You press "volume up" and LiteLLM figures out which signal each device needs.

---

## 7. Where smolagents Falls Short for This Project

### No persistent multi-turn memory

```python
# Question 1
agent.run("Is emotion recognition prohibited?")

# Question 2 — agent has forgotten Question 1
agent.run("What about educational settings?")
# Answers generically — doesn't know Q2 is about emotion recognition
```

To work around this you would have to manually build a history and re-inject it:

```python
history = []
q1 = "Is emotion recognition prohibited?"
a1 = agent.run(q1)
history.append(f"Q: {q1}\nA: {a1}")

q2 = "What about educational settings?"
context = "\n".join(history)
a2 = agent.run(f"{context}\n\nQ: {q2}")
```

This is manual, fragile, and you'd need to manage token limits yourself.
LangGraph handles all of this with one line: `{"configurable": {"thread_id": "..."}}`.

### No human-in-the-loop

LangGraph's `interrupt_before=["human_review"]` pauses the graph before sending
a critical finding to the user. smolagents has no equivalent mechanism. You could
check the output after `.run()` completes and add a warning, but you cannot pause
mid-execution and route to a different handler.

For a compliance tool like RegulationAdvisor, this is significant. The ability to
pause on "prohibited" findings and flag for legal review is a product requirement,
not a nice-to-have.

### No visual debugger

When something goes wrong in a LangGraph agent, LangGraph Studio shows you exactly
which node ran, what the state looked like at each step, and where the routing
went wrong. With smolagents you get print logs. For complex multi-step reasoning
this is a real productivity gap.

---

## 8. Where smolagents Wins

### Speed of prototyping

From zero to a working agent:

```python
# smolagents — 4 lines, no imports except the library
from smolagents import ToolCallingAgent, LiteLLMModel, tool

@tool
def search(query: str) -> str:
    """Search for something."""
    return "result"

agent = ToolCallingAgent(tools=[search], model=LiteLLMModel("openrouter/..."))
print(agent.run("search for EU AI Act"))
```

Compare to LangGraph: state schema, graph nodes, conditional edges, compilation — even
with our shared infrastructure, `build_agent_graph()` is 28 lines before you can run it.

### CodeAgent — code execution

smolagents has a `CodeAgent` class that generates and executes Python code as its
"tool". The LLM writes Python, the agent runs it in a sandbox, observes the output,
and iterates. This is fundamentally different from function-call tools — the agent
generates programs, not just function arguments.

LangGraph does not have a built-in equivalent. You'd have to build a "code execution"
node yourself.

---

## 9. Dependency Notes

We installed `smolagents[litellm]` — the `[litellm]` extra installs LiteLLM as well.
Without it you get the core smolagents package but `LiteLLMModel` is unavailable.

```toml
# pyproject.toml
"smolagents[litellm]",
```

```text
# requirements.txt (for HuggingFace Spaces)
smolagents[litellm]
```

`smolagents[litellm]` pulls in `litellm` which itself depends on `openai`. This is
fine — we already use `langchain-openai` which has the same dependency.

---

## 10. Summary: What F10 Accomplished

| Goal | Outcome |
|---|---|
| Identical 3-tool agent in smolagents | Done — `build_smolagents_agent()` |
| Zero tool code duplication | Done — `LangChainTool` adapter reuses `tools.py` |
| Model provider flexibility | Done — `_litellm_model_id()` maps from settings |
| Benchmark comparison documented | Done — `docs/smolagents_comparison.md` |
| LinkedIn Article #3 material | Ready — full comparison with code, tables, verdict |
| Version | `0.2.0 → 0.2.1` |

The concrete conclusion from the comparison:

> **smolagents is better for rapid prototyping and code-execution agents.
> LangGraph is better for production multi-turn, stateful, human-in-the-loop workflows.**
> For RegulationAdvisor, LangGraph is the right choice and the one we keep in production.
