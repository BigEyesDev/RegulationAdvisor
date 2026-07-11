# F4 — LangChain RAG Chain

> **Audience:** Complete beginners. No AI or LangChain background required.
> **What you will understand after reading this:** How we connect a retrieved set of legal
> documents to a language model so it can give cited, grounded answers — not hallucinated ones.

---

## The Problem This Feature Solves

After F3 we have a searchable index of 213 regulation chunks. But the index just returns
raw text — it cannot *talk* to a user. The user asks a question and gets back paragraphs of
legal text with no explanation.

F4 answers: **"How do we turn retrieved text into a coherent, cited answer?"**

The answer is a **RAG chain** (Retrieval-Augmented Generation):

```
User question
     │
     ▼
  Retriever (FAISS)  →  5 most relevant chunks
                                 │
                                 ▼
                     Language Model (Groq / Qwen)
                                 │
                                 ▼
                     "According to Article 5, the following..."
```

The LLM does not know the EU AI Act from memory. We **give it the relevant pages** at
query time and tell it: "Answer only from what I'm showing you, and cite the Article."

---

## Analogy: The Open-Book Exam

Imagine a student who has never studied EU law. You hand them 5 pages from the EU AI Act
and ask: "What AI practices are prohibited?"

- The student reads those 5 pages.
- They find the answer in Article 5.
- They write: "According to Article 5, the following practices are prohibited: ..."

That student **is** the language model. The 5 pages are the **context**. Our job is to:
1. Pick the right pages (retrieval — F3 did this)
2. Hand them to the student with clear instructions (this feature — F4)
3. Let the student write the answer

The key constraint: the student is told **"write only what's in the pages"**. This is
how we prevent hallucination — making up facts that sound plausible but aren't real.

---

## What Was Built in F4

### 1. `src/regulation_advisor/prompts/system_prompt.txt`

This is the "exam instructions" given to the LLM before every question.

```
You are an EU AI Act compliance advisor.
Answer based ONLY on the provided regulation text.
Always cite the specific Article number when referencing a rule or requirement.
If the answer is not in the context, say "I cannot find this in the provided regulation text."

Context:
{context}
```

**Why a separate file?**
- The prompt is a configuration artifact, not code. It can be edited without touching Python.
- In Week 3, we'll run automated evaluation (promptfoo) that loads this file directly.
- Non-engineers (lawyers, product managers) can tweak it without breaking anything.

**What is `{context}`?**
It's a placeholder. Before sending the prompt to the LLM, we fill it in with the actual
retrieved chunks. LangChain does this substitution automatically.

---

### 2. `_build_chain()` — The LangChain Pipeline

```python
def _build_chain():
    system_prompt = (_PROMPTS_DIR / "system_prompt.txt").read_text()
    llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()
```

Let's unpack each piece.

#### `ChatGroq`

Groq is an LLM inference provider — think of it as a very fast server that runs large
language models. `ChatGroq` is LangChain's connector to that server.

```python
llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
```

- `model`: which AI model to use (e.g. `qwen/qwen3-32b` — a 32-billion parameter model)
- `api_key`: your secret key from Groq's website (stored in `.env`, never in code)

**Analogy:** `ChatGroq` is like a phone call to a very smart lawyer. You tell them which
lawyer you want (`model`), you authenticate yourself (`api_key`), and then you can talk.

#### `ChatPromptTemplate`

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),   # the exam instructions + context
    ("human", "{question}"),     # the user's actual question
])
```

Modern LLMs work in a **chat format**: each message has a *role*:
- `"system"` — background instructions the LLM always has in mind
- `"human"` — what the user just said
- `"ai"` — what the LLM said back (used in multi-turn conversations)

`{question}` is another placeholder — filled in with the real user question at runtime.

**Analogy:** Imagine writing a script for someone making a phone call:
- `system`: "You are a legal assistant. Only answer from the documents I provide."
- `human`: "What does Article 5 say about prohibited AI?"

The script is the template. The actual words are substituted at call time.

#### The Pipe Operator `|`

```python
return prompt | llm | StrOutputParser()
```

This is LangChain's **LCEL** (LangChain Expression Language). The `|` (pipe) chains
components together — the output of each feeds into the next.

| Step | Input | Output |
|---|---|---|
| `prompt` | `{context, question}` dict | Formatted chat messages |
| `llm` | Chat messages | `AIMessage` object |
| `StrOutputParser()` | `AIMessage` | Plain Python `str` |

**Analogy:** Think of Unix pipes in a terminal:
```bash
cat file.txt | grep "Article" | head -5
```
Each `|` passes the output of the left command to the right command.
LangChain's `|` does the same thing with AI components.

The final output is a plain Python string — the LLM's answer.

---

### 3. `_format_context(chunks)` — Labelling the Evidence

Before handing chunks to the LLM, we format them so the LLM knows *where* each piece
of text came from. This is what enables citation.

```python
def _format_context(chunks) -> str:
    parts = []
    for chunk in chunks:
        header = f"[{chunk.source_document} — Article {chunk.article_number}]"
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)
```

**Example output (5 chunks → one context string):**
```
[eu_ai_act.pdf — Article 5]
The following AI practices shall be prohibited:
(a) the placing on the market, the putting into service or the use of an AI system...

---

[eu_ai_act.pdf — Article 6]
Classification rules for high-risk AI systems...
```

By labelling each chunk with `Article N`, we give the LLM the information it needs to
write: *"According to Article 5..."*. Without labels, the LLM might give a correct answer
but wouldn't know which article to cite.

---

### 4. `respond(message, history)` — The Heart of the Feature

```python
def respond(message: str, history: list) -> str:
    result = retriever.search(message, k=settings.retrieval_k)
    context = _format_context(result.chunks)
    answer = chain.invoke({"context": context, "question": message})
    return answer
```

This function is called by Gradio every time the user sends a message. It does exactly
three things:

1. **Search:** Call the FAISS retriever with the user's question → get 5 chunks
2. **Format:** Turn those chunks into a labelled context string
3. **Generate:** Run the LangChain chain with context + question → get the LLM's answer

**Full data flow example:**

```
User types: "What AI practices are completely prohibited?"
                │
                ▼
    retriever.search("What AI practices are completely prohibited?", k=5)
                │
                ▼
    Returns chunks: Article 5, Article 6, Article 7, Article 9, Article 10
                │
                ▼
    _format_context([...]) builds:
        "[eu_ai_act.pdf — Article 5]\nThe following AI practices shall be prohibited..."
        "[eu_ai_act.pdf — Article 6]\nClassification rules..."
        ...
                │
                ▼
    chain.invoke({
        "context": "<formatted chunks>",
        "question": "What AI practices are completely prohibited?"
    })
                │
                ▼
    LLM response:
    "According to Article 5 of the EU AI Act, the following AI practices are
     completely prohibited:
     (a) Subliminal manipulation techniques...
     (b) Exploitation of vulnerabilities...
     (c) Social scoring by public authorities..."
```

---

## Understanding the LangChain Architecture (Bigger Picture)

LangChain is a framework for building applications with LLMs. It provides:

| Concept | What it is | Our usage |
|---|---|---|
| **LLM** | The AI model | `ChatGroq(model=...)` |
| **PromptTemplate** | Message scaffolding with placeholders | `ChatPromptTemplate.from_messages(...)` |
| **OutputParser** | Converts LLM output to a Python type | `StrOutputParser()` → plain `str` |
| **Chain (LCEL)** | Composed pipeline of components | `prompt \| llm \| parser` |

**Why LangChain and not just calling the API directly?**

You could call the Groq API directly using Python's `requests` library. LangChain adds:
- **Swappability:** Change `ChatGroq` to `ChatGoogleGenerativeAI` in one line — the rest
  of the code stays the same (same interface, different implementation)
- **LCEL Chains:** Declarative composition with `|` instead of nested function calls
- **Observability:** Every chain invocation can be traced (LangSmith, upcoming in Week 3)
- **Streaming:** `chain.stream(...)` streams tokens instead of waiting for the full answer

---

## What Changed in This Feature

### Files Created
- `src/regulation_advisor/prompts/system_prompt.txt` — the instruction prompt for the LLM

### Files Modified
- `src/regulation_advisor/ui/gradio_app.py` — replaced the TODO stub with:
  - `_build_chain()` — builds the LangChain pipeline
  - `_format_context(chunks)` — formats retrieved chunks for the LLM
  - `build_ui(retriever)` — now builds the chain and defines `respond()` as a closure
  - `respond(message, history)` — the actual query handler (closure inside `build_ui`)

### Files NOT Changed
- `config.py` — all settings already existed (`llm_model`, `groq_api_key`, `retrieval_k`)
- `retriever.py` — unchanged; we call `retriever.search()` as already designed
- `models.py` — `RetrievalResult` and `RegulationChunk` used as-is

---

## Key Vocabulary

| Term | Plain English |
|---|---|
| **RAG** | Retrieval-Augmented Generation — retrieve relevant documents, then generate an answer using them |
| **LLM** | Large Language Model — the AI that generates text (e.g. Qwen, GPT-4) |
| **Groq** | A company providing very fast LLM inference via API |
| **LangChain** | A Python framework for building apps that use LLMs |
| **LCEL** | LangChain Expression Language — the `\|` pipe syntax for chaining components |
| **Prompt** | The text you send to an LLM to instruct it |
| **System message** | Background instructions given to the LLM before the user's question |
| **Context** | The retrieved regulation text injected into the system prompt |
| **Hallucination** | When an LLM makes up a fact that sounds real but isn't — RAG prevents this |
| **StrOutputParser** | Converts an `AIMessage` object to a plain Python string |
| **Closure** | A function defined inside another function that captures the outer function's variables |

---

## Gate Check

```bash
# This must print "import ok" with no errors
uv run python -c "from regulation_advisor.ui.gradio_app import build_ui; print('import ok')"
```

Expected output:
```
import ok
```

---

## Why Closures? (For Those Curious About the `respond` Function)

`respond` is defined *inside* `build_ui`:

```python
def build_ui(retriever: Retriever) -> gr.Blocks:
    chain = _build_chain()

    def respond(message: str, history: list) -> str:
        result = retriever.search(message, k=settings.retrieval_k)
        ...

    gr.ChatInterface(fn=respond, ...)
```

`respond` uses `retriever` and `chain` — but those are defined in `build_ui`'s scope,
not inside `respond` itself. Python allows this: inner functions can *close over*
(capture) variables from their outer scope. This is called a **closure**.

**Why not use a class?** We could have written `class ChatBot: def __init__(self, retriever): ...`
The closure approach is simpler for a single function. The class approach would be better
if we needed multiple methods (e.g., also `stream_respond`, `reset_history`).

**Why not a global variable?** We could have written `_retriever = None` at module level
and set it before launching. But global mutable state is fragile — if someone imports
`build_ui` in a test, they'd need to remember to set the global. The closure makes the
dependency explicit: `build_ui(retriever=...)` — you must pass a retriever in.

---

## Summary

F4 is the "brain" of the chatbot. Before F4, we had a searchable index (F3) but no way to
turn search results into an answer. After F4:

1. A user question arrives
2. FAISS finds the 5 most relevant regulation chunks
3. LangChain formats them with Article labels
4. The LLM reads those 5 chunks and writes a cited answer
5. Gradio displays the answer to the user

The key design principle is **grounding**: the LLM is explicitly told to answer only from
the provided text. This is the fundamental technique that makes LLM outputs trustworthy
in high-stakes domains like legal compliance.
