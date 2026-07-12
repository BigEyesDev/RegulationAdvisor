---
title: RegulationAdvisor
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.20.0"
app_file: src/regulation_advisor/ui/app_runner.py
pinned: false
---

# RegulationAdvisor

**Ask compliance questions about the EU AI Act in plain English. Get answers that cite the exact Article.**

> EU AI Act becomes fully enforceable 2 August 2026.

**Live demo:** [huggingface.co/spaces/BigEyesDev/regulation-advisor](https://huggingface.co/spaces/BigEyesDev/regulation-advisor)

---

## What it does

You type a question. A LangGraph agent decides which tools to call — semantic search over the
regulation PDFs, a structured CSV lookup for timelines and penalties, or a live web search
for recent enforcement news. The LLM writes a cited answer grounded in the actual text.
Critical findings (prohibited practices, large fines) are flagged with a legal review warning.

```
"Is emotion recognition in the workplace allowed under the EU AI Act?"
        ↓
Agent calls search_regulations("emotion recognition workplace")
        ↓
Retrieves Article 5(1)(f) — prohibited practice
        ↓
Agent calls query_structured_data("enforcement date prohibited AI")
        ↓
Returns: enforcement date 2025-02-02 from ai_act_timeline.csv
        ↓
LLM writes answer citing Article 5(1)(f) + enforcement date
        ↓
⚠️ Critical finding warning appended — verify with legal professional
```

---

## Architecture (v0.2 — LangGraph agent)

```
User question
      │
      ▼
  agent node  ←──────────────────────┐
  LLM decides: call tool or answer?  │
      │                              │
      ├── tool call ──► ToolNode ────┘  (loops until done)
      │                 ├── search_regulations   (FAISS semantic search)
      │                 ├── query_structured_data (CSV: timelines, penalties)
      │                 └── search_web            (Tavily live search)
      │
      ├── critical finding ──► human_review node (interrupt — ⚠️ banner shown)
      │
      └── done ──► answer displayed in Gradio
```

Multi-turn memory: `MemorySaver` checkpoints state per `thread_id` — the agent
remembers the conversation across follow-up questions.

---

## Run locally

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), regulation PDFs in `data/`

```bash
# 1. Clone and install
git clone https://github.com/BigEyesDev/RegulationAdvisor.git
cd RegulationAdvisor
uv sync --group dev

# 2. Configure API keys
cp .env.example .env
# Edit .env — set at minimum:
#   OPENROUTER_API_KEY  (or GROQ_API_KEY / GOOGLE_API_KEY)
#   TAVILY_API_KEY      (needed for the web search tool)

# 3. Add the regulation documents to data/
#    eu_ai_act.pdf, gdpr.pdf, ai_act_timeline.csv,
#    risk_classification.csv, penalty_structure.csv
#    (see data/README.md for download links)

# 4. Build the FAISS index — one time only, ~20 seconds
uv run python scripts/ingest.py

# 5. Launch the agent chatbot
uv run python src/regulation_advisor/ui/app_runner.py
# Open http://localhost:7860
```

---

## Swap the LLM model

Edit two lines in `.env` — no code changes needed:

```bash
LLM_PROVIDER=openrouter          # openrouter | groq | google
LLM_MODEL=deepseek/deepseek-v4-flash
```

The agent and the Gradio UI both read from the same factory (`build_llm()` in `llm.py`).

---

## Run tests

```bash
# Unit tests — no data files or API keys needed (21 tests)
uv run pytest tests/unit/ -v

# Integration tests — requires FAISS index to be built first
uv run pytest tests/integration/ -v

# Week 2 gate checks specifically
uv run pytest tests/unit/test_tools.py tests/unit/test_agent_graph.py -v
```

---

## Deploy to HuggingFace Spaces

```
Settings → Variables and Secrets → add each key from .env.example
```

Required secrets for the agent to work:

| Secret name | Where to get it | Required for |
|---|---|---|
| `OPENROUTER_API_KEY` | openrouter.ai | LLM (default provider) |
| `TAVILY_API_KEY` | tavily.com (1000 req/month free) | `search_web` tool |

Optional (if switching providers):

| Secret name | Where to get it |
|---|---|
| `GROQ_API_KEY` | console.groq.com |
| `GOOGLE_API_KEY` | aistudio.google.com |

The Space also needs the PDF and CSV data files committed or uploaded — see `data/README.md`.
On cold start, `_ensure_index()` builds the FAISS index automatically from those files (~20 s).

---

## Tech stack

| Layer | Tool |
|---|---|
| Package manager | uv |
| Agent framework | LangGraph — stateful graph with `MemorySaver` checkpointing |
| LLM | DeepSeek V4 Flash via OpenRouter (swappable via `.env`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, no API) |
| Vector store | FAISS |
| Tools | LangChain `@tool` — semantic search, CSV lookup, Tavily web search |
| LLM framework | LangChain (tool binding, message types) |
| Document loading | LlamaIndex + PyMuPDF |
| UI | Gradio 6 |
| Deployment | HuggingFace Spaces |
