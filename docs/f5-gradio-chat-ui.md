# F5 — Gradio Chat UI v0.1

> **Audience:** Complete beginners. No web development or UI background required.
> **What you will understand after reading this:** How we build a working chatbot web interface
> in ~30 lines of Python, and how every piece of the system connects to make it work end-to-end.

---

## The Problem This Feature Solves

After F4, we have:
- A FAISS index of 213 regulation chunks (F3)
- A LangChain chain that takes a question + context and returns a cited answer (F4)

But there is no way for a user to actually *use* this. There is no interface.

F5 builds **the front-door**: a browser-based chat interface where a user types a question
and gets a cited legal answer in seconds.

```
Browser          Python (our code)
  │                    │
  │ "What is Article 5?"
  │ ──────────────────► │
  │                    │  1. FAISS: find 5 chunks
  │                    │  2. LangChain: generate answer
  │                    │  3. Return string
  │ "According to       │
  │  Article 5, the ... "
  │ ◄────────────────── │
```

---

## Analogy: The Receptionist + Back Office

Think of the app as a law firm:

| Role | What they do | Code equivalent |
|---|---|---|
| **Receptionist** | Talks to clients, asks their question, reads back the answer | Gradio `ChatInterface` |
| **Paralegal** | Searches the filing cabinet for relevant documents | `retriever.search()` |
| **Lawyer** | Reads those documents and writes a cited legal memo | LangChain chain (F4) |
| **Filing cabinet** | All 213 regulation chunks on disk | FAISS index (F3) |

Before F5 we had the back office (paralegal + lawyer + filing cabinet) but no receptionist.
F5 adds the receptionist.

---

## What Was Built in F5

### 1. `build_ui(retriever)` — The Chat Interface

```python
def build_ui(retriever: Retriever) -> gr.Blocks:
    chain = _build_chain()

    def respond(message: str, history: list) -> str:
        result = retriever.search(message, k=settings.retrieval_k)
        context = _format_context(result.chunks)
        return chain.invoke({"context": context, "question": message})

    with gr.Blocks(title="RegulationAdvisor v0.1") as demo:
        gr.Markdown(
            "## EU AI Act Compliance Advisor\n"
            "Ask any question about the EU AI Act or GDPR. "
            "Every answer cites the relevant Article."
        )
        gr.ChatInterface(fn=respond, title="")

    return demo
```

Let's unpack every line.

#### `gr.Blocks` — The Page Container

```python
with gr.Blocks(title="RegulationAdvisor v0.1") as demo:
    ...
```

`gr.Blocks` is Gradio's low-level layout container. Think of it as an HTML `<body>` tag —
everything on the page goes inside it.

- `title=` sets the browser tab title
- The `with` block: everything indented inside is added to the page

**What is Gradio?** Gradio is a Python library that lets you build web UIs with zero
HTML/CSS/JavaScript. You describe your interface in Python, and Gradio handles all the
web server, routing, and browser rendering automatically. It's the fastest way to go from
a Python function to a browser-usable tool.

#### `gr.Markdown` — Static Text

```python
gr.Markdown(
    "## EU AI Act Compliance Advisor\n"
    "Ask any question about the EU AI Act or GDPR. "
    "Every answer cites the relevant Article."
)
```

This renders formatted text at the top of the page. `##` means a second-level heading
(Markdown syntax). This gives the user context about what the tool does.

#### `gr.ChatInterface` — The Chat Widget

```python
gr.ChatInterface(fn=respond, title="")
```

This single line creates the entire chat UI:
- A message input box at the bottom
- A conversation history panel above it
- A "Send" button
- A "Clear" button to reset the conversation

**The only required argument is `fn=`** — the function to call when the user sends a message.

Gradio handles all the rest automatically:
- Storing conversation history between messages
- Displaying the user's message on the right side
- Displaying the AI's response on the left side
- Streaming tokens as they arrive (optional)
- Mobile-responsive layout

**Analogy:** `gr.ChatInterface` is like a pre-built IKEA desk. You don't build it from
scratch — you just assemble the pieces. `fn=respond` is you plugging in the lamp (the
actual intelligence). Gradio provides the desk (the UI shell).

---

### 2. The `respond` Function — How Gradio Calls Your Code

```python
def respond(message: str, history: list) -> str:
    result = retriever.search(message, k=settings.retrieval_k)
    context = _format_context(result.chunks)
    return chain.invoke({"context": context, "question": message})
```

`gr.ChatInterface` calls this function every time the user hits "Send".

**Arguments Gradio passes automatically:**
- `message` (str): the user's typed text — e.g. "What is Article 5?"
- `history` (list): all previous messages in this conversation session

**Return value:**
- Must return a `str` — the AI's response text

We don't use `history` here because our RAG system is *stateless*: each question is
answered independently from the regulation index. In Week 2, when we add the LangGraph
agent, history will be used for multi-turn conversations.

**Data flow inside `respond`:**
```
message = "What AI practices are completely prohibited?"
                │
                ▼
    retriever.search(message, k=5)
    → RetrievalResult(chunks=[Article 5, Article 6, Article 7, ...])
                │
                ▼
    _format_context(result.chunks)
    → "[eu_ai_act.pdf — Article 5]\nThe following AI practices..."
                │
                ▼
    chain.invoke({"context": ..., "question": message})
    → "According to Article 5, the prohibited AI practices include..."
                │
                ▼
    Gradio displays this string in the chat panel
```

---

### 3. `src/regulation_advisor/ui/app_runner.py` — The Startup Script

This is the entry point — the script you run to start the application.

```python
from regulation_advisor.retrieval.embeddings import SentenceTransformerEmbedder
from regulation_advisor.retrieval.retriever import Retriever
from regulation_advisor.retrieval.store import FAISSVectorStore
from regulation_advisor.ui.gradio_app import build_ui

def _load_retriever() -> Retriever:
    store = FAISSVectorStore()
    store.load(Path("data/index"))          # loads index.faiss + chunks.pkl
    embedder = SentenceTransformerEmbedder()
    return Retriever(store=store, embedder=embedder)

if __name__ == "__main__":
    retriever = _load_retriever()           # Step 1: load the FAISS index
    demo = build_ui(retriever)              # Step 2: build the Gradio UI
    demo.launch()                           # Step 3: start the web server
```

**Why a separate `app_runner.py` instead of putting everything in `gradio_app.py`?**

This follows the **Single Responsibility Principle**:
- `gradio_app.py` knows how to *build* a UI given a retriever — it's a library function
- `app_runner.py` knows how to *start* the application — it's the entry point

This separation means:
1. Tests can call `build_ui(mock_retriever)` without launching a web server
2. In Week 4, FastAPI mounts `build_ui()` as a sub-app — `app_runner.py` is replaced
   by FastAPI's launcher, and `gradio_app.py` stays unchanged

**`if __name__ == "__main__":`** — This is a Python convention that means:
"Only run this code if this file is executed directly (not imported by another file)."

If you type `python app_runner.py` → runs. If another file does `import app_runner` → does NOT run.

**`demo.launch()`** — Starts Gradio's built-in web server on `http://localhost:7860`.
Gradio opens a browser tab automatically. When you press `Ctrl+C`, the server stops.

---

## The Complete End-to-End Flow (Everything Together)

Here is what happens from the moment you run the script to the moment a user gets an answer:

### Startup (runs once)
```
uv run python src/regulation_advisor/ui/app_runner.py
        │
        ▼
1. FAISSVectorStore.load("data/index")
   → Reads index.faiss (320 KB) + chunks.pkl (562 KB) into memory
   → Now we have 213 chunks and their embedding vectors in RAM
        │
        ▼
2. SentenceTransformerEmbedder()
   → Loads the all-MiniLM-L6-v2 model (22 MB) into memory
        │
        ▼
3. Retriever(store=store, embedder=embedder)
   → Wires them together
        │
        ▼
4. build_ui(retriever)
   → _build_chain() connects to Groq API
   → Gradio page is constructed
        │
        ▼
5. demo.launch()
   → Web server starts at http://localhost:7860
   → Browser opens automatically
   → App is ready to receive questions
```

Total startup time: ~2-3 seconds (index load: ~0.2s, model load: ~1.5s)

### Per-Query (runs for every user message)
```
User types: "Is emotion recognition in the workplace allowed?"
        │
        ▼
Gradio calls: respond("Is emotion recognition...", history=[])
        │
        ▼
retriever.search("Is emotion recognition...", k=5)
   → embedder encodes question → [0.12, -0.34, ...] (384 numbers)
   → FAISS finds 5 nearest chunks by vector similarity
   → Returns: [Article 6, Article 7, Article 9, Article 45, Article 50]
        │
        ▼
_format_context([Article 6, Article 7, ...])
   → "[eu_ai_act.pdf — Article 6]\nClassification rules for high-risk AI systems..."
   → "[eu_ai_act.pdf — Article 7]\n..."
   → ... (joined by "---" separators)
        │
        ▼
chain.invoke({"context": ..., "question": "Is emotion recognition..."})
   → Groq API receives:
       SYSTEM: You are an EU AI Act compliance advisor...
               Context: [eu_ai_act.pdf — Article 6]...
       HUMAN:  Is emotion recognition in the workplace allowed?
   → Groq returns: "According to Article 6 and Article 9 of the EU AI Act..."
        │
        ▼
Gradio displays the answer in the chat panel
Total query time: ~500ms–2s (network latency to Groq API is the bottleneck)
```

---

## The 5 Benchmark Queries

The plan requires that all 5 of these return an answer with at least one "Article N" citation.

| # | Query | Expected Article(s) cited |
|---|---|---|
| 1 | "What AI practices are completely prohibited?" | Article 5 |
| 2 | "What are the penalties for deploying a prohibited AI system?" | Article 99 |
| 3 | "Is emotion recognition in the workplace allowed?" | Article 6, Article 9 |
| 4 | "What is a high-risk AI system?" | Article 6, Annex III |
| 5 | "When does the EU AI Act become fully enforceable?" | Article 113 |

If a query returns an answer without any "Article" in it, the retrieval failed to find
the right chunks. To debug: run `retriever.search(query, k=10)` directly in Python and
check which articles are returned.

---

## What Changed in This Feature

### Files Created
- `src/regulation_advisor/ui/app_runner.py` — startup entry point

### Files Modified
- `src/regulation_advisor/ui/gradio_app.py` (from F4) — `build_ui()` was already
  complete from F4. No changes needed in F5 — the implementation was architected so that
  F4 built everything and F5 verified it works end-to-end with the `app_runner.py`.

### Files NOT Changed
- All of F1–F4 — the pipeline is stable

---

## Understanding Why This Architecture Is Modular

Let's trace how easy it would be to change one piece without touching others:

**Scenario: Replace Groq with Google Gemini**
```python
# Only change in gradio_app.py:
# Before:
llm = ChatGroq(model=settings.llm_model, api_key=settings.groq_api_key)
# After:
llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.google_api_key)
```
`app_runner.py` doesn't change. `retriever.py` doesn't change. The UI doesn't change.

**Scenario: Replace FAISS with ChromaDB**
```python
# Only change in app_runner.py:
# Before:
store = FAISSVectorStore(); store.load(Path("data/index"))
# After:
store = ChromaDBVectorStore()  # already persisted, loads automatically
```
`gradio_app.py` doesn't change. The LangChain chain doesn't change.

**Scenario: Add streaming (show tokens as they arrive)**
```python
# Only change in gradio_app.py:
def respond(message, history):
    ...
    return chain.stream({"context": context, "question": message})  # stream instead of invoke
```
`app_runner.py` doesn't change.

This is the power of the **dependency injection pattern** (`build_ui(retriever)` —
you pass dependencies in, not hardcode them).

---

## Key Vocabulary

| Term | Plain English |
|---|---|
| **Gradio** | Python library that builds browser UIs with no HTML/CSS/JavaScript |
| **`gr.Blocks`** | Gradio's page container — think of it as the HTML page itself |
| **`gr.ChatInterface`** | A pre-built Gradio widget for chat conversations |
| **`gr.Markdown`** | A Gradio element that renders formatted text |
| **Entry point** | The script you run to start a program (`python app_runner.py`) |
| **Startup vs per-query** | Startup code runs once; per-query code runs on every user message |
| **`demo.launch()`** | Starts Gradio's web server so the browser can connect |
| **Stateless** | Each question is answered independently, without memory of past questions |
| **Dependency injection** | Passing an object into a function instead of creating it inside |
| **Single Responsibility Principle** | Each file/function does exactly one thing |

---

## How to Run the App

```bash
# Make sure the index is built first (F3)
uv run python scripts/ingest.py

# Start the chat UI
uv run python src/regulation_advisor/ui/app_runner.py
```

Expected startup output:
```
INFO — Loading FAISS index from data/index …
INFO — Added 213 chunks to FAISS index
INFO — Loading embedding model …
INFO — Launching Gradio UI at http://localhost:7860
Running on local URL: http://127.0.0.1:7860
```

Then open `http://localhost:7860` in your browser. Type a question. See a cited answer.

---

## Summary

F5 is the "window" of the system — the place where a human and the RAG pipeline meet.

Before F5: we had a powerful back-end (index + retrieval + LLM chain) but no way for a
non-technical person to use it.

After F5: anyone with a browser can type a question and get a cited, regulation-grounded
answer in about 1 second.

The key lesson: **a good chatbot UI is mostly plumbing**. Gradio's `ChatInterface` handles
99% of the UI complexity (layout, history, streaming, mobile). Our code only provides
the function `respond()` — which is ~5 lines of pure logic. The separation between UI
and business logic is what makes the system easy to extend in Weeks 2–4.
