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
for recent enforcement news. The LLM writes a cited answer grounded in the actual regulation text.
A guardrail layer checks every response for hallucinated article citations and flags legal claims
before the answer reaches the user.

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
Guardrail checks: citation verified, legal claim flagged
        ↓
LLM writes answer citing Article 5(1)(f) + enforcement date
        ↓
⚠️ Critical finding warning appended — verify with legal professional
```

---

## Architecture (v0.4)

```
                        ┌─────────────────────────────────────┐
                        │           FastAPI (port 8000)        │
                        │                                      │
                        │  POST /api/chat      (SSE stream)   │
                        │  POST /api/chat/sync (sync)         │
                        │  GET  /api/metrics                  │
                        │  POST /api/evaluate                 │
                        │  GET  /health                       │
                        │                                      │
                        │  Gradio UI  mounted at  /           │
                        └──────────────┬──────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  LangGraph Agent │
                              │   (StateGraph)   │
                              └────────┬─────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               ▼                       ▼                        ▼
    search_regulations        query_structured_data         search_web
    (ChromaDB / FAISS         (CSV: timelines,             (Tavily live
     semantic search)          penalties, risk tiers)       enforcement news)
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Guardrail chain (Chain of Responsibility)  │
    │  FaithfulnessCheck                         │
    │  → CitationVerificationCheck               │
    │  → LegalClaimFlagCheck                     │
    └──────────────────────────────────────┘
               │
               ▼
        Answer + warnings
```

Multi-turn memory: `MemorySaver` checkpoints state per `thread_id`.

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

# 4. Build the vector index — one time only, ~20 seconds
uv run python scripts/ingest.py

# 5. Launch the FastAPI server (includes Gradio UI at /)
uv run uvicorn regulation_advisor.api.app:app --host 0.0.0.0 --port 8000 --reload
# Gradio UI:  http://localhost:8000/
# REST API:   http://localhost:8000/docs
```

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Streaming SSE chat — streams LLM tokens as `data:` events |
| `POST` | `/api/chat/sync` | Synchronous chat — returns full answer in one response |
| `GET` | `/api/metrics` | Latest RAGAS scores from the last evaluation run |
| `POST` | `/api/evaluate` | Triggers RAGAS evaluation in the background |
| `GET` | `/health` | Health check |

```bash
# Streaming chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What AI practices are prohibited under Article 5?"}'

# Sync chat
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the fine for a prohibited AI system?"}'

# Get evaluation scores
curl http://localhost:8000/api/metrics
```

---

## Swap the LLM or vector store

Edit `.env` — no code changes needed:

```bash
# Switch LLM provider and model
LLM_PROVIDER=openrouter          # openrouter | groq | google
LLM_MODEL=deepseek/deepseek-v4-flash

# Switch vector store (chromadb requires the sidecar to be running)
VECTOR_STORE_BACKEND=faiss       # faiss | chromadb
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

---

## Run with ChromaDB (persistent vectors)

```bash
# Start ChromaDB sidecar
docker run -p 8001:8000 chromadb/chroma:latest

# Set backend in .env
VECTOR_STORE_BACKEND=chromadb

# Ingest once — data persists across restarts
uv run python scripts/ingest.py

uv run uvicorn regulation_advisor.api.app:app --host 0.0.0.0 --port 8000
```

---

## Evaluation

RAGAS scores are tracked across all answers. Run the evaluation suite against the 20-question
golden dataset:

```bash
# Run RAGAS evaluation and save scores to evals/baseline_scores.json
uv run python scripts/run_evaluation.py

# Run the promptfoo 30-case regression suite (requires Node.js)
npx promptfoo eval --config evals/promptfoo.yaml
```

The promptfoo suite also runs automatically on every PR to `main` via GitHub Actions.

**Current baseline targets:**

| Metric | Target |
|--------|--------|
| Faithfulness | ≥ 0.80 |
| Answer Relevancy | ≥ 0.70 |
| Context Precision | ≥ 0.70 |
| Context Recall | ≥ 0.70 |

---

## Run tests

```bash
# Unit tests — no data files or API keys needed
uv run pytest tests/unit/ -v

# Integration tests — requires index to be built first
uv run pytest tests/integration/ -v

# Full suite with coverage
uv run pytest tests/ --cov=src -v
```

---

## Deploy to HuggingFace Spaces

Add secrets in **Settings → Variables and Secrets**:

| Secret | Source | Required for |
|--------|--------|-------------|
| `OPENROUTER_API_KEY` | openrouter.ai | LLM (default provider) |
| `TAVILY_API_KEY` | tavily.com | `search_web` tool |
| `GROQ_API_KEY` | console.groq.com | If using Groq provider |
| `GOOGLE_API_KEY` | aistudio.google.com | If using Google provider |

On cold start, the Spaces build runs `scripts/ingest.py` automatically (~20 s).

---

## Tech stack

| Layer | Tool |
|-------|------|
| Package manager | uv |
| API server | FastAPI + Uvicorn (SSE streaming) |
| Agent framework | LangGraph — stateful graph with `MemorySaver` checkpointing |
| LLM | DeepSeek V4 Flash via OpenRouter (swappable via `.env`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, no API cost) |
| Vector store | FAISS (dev) / ChromaDB (persistent) — swapped via config |
| Tools | LangChain `@tool` — regulation search, CSV lookup, Tavily web search |
| Document loading | LlamaIndex + PyMuPDF |
| Guardrails | Chain of Responsibility — faithfulness, citation, legal-claim checks |
| Evaluation | RAGAS + promptfoo 30-case regression suite |
| UI | Gradio 6 (Chat + Evaluation Dashboard tabs) |
| CI | GitHub Actions — promptfoo eval on every PR |
| Deployment | HuggingFace Spaces |
