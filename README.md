---
title: RegulationAdvisor
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# RegulationAdvisor

**Ask compliance questions about the EU AI Act in plain English. Get answers that cite the exact Article.**

> EU AI Act becomes fully enforceable 2 August 2026.

**Live demo:** not published yet — see [Run locally](#run-locally) below.

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

## Architecture (v0.6)

```
                        ┌─────────────────────────────────────┐
                        │           FastAPI (port 8000)        │
                        │                                      │
                        │  POST /api/chat      (SSE stream)   │
                        │  POST /api/chat/sync (sync)         │
                        │  GET  /api/metrics                  │
                        │  POST /api/evaluate  (disabled by    │
                        │                       default)       │
                        │  GET  /api/health                   │
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

**Bring your own key (BYOK):** every chat request (API or Gradio) can optionally
supply its own `provider` + `model` + `api_key` — OpenAI, Anthropic (Claude), Groq,
Google Gemini, or OpenRouter. When supplied, a throwaway agent is built for that one
request using the caller's key; nothing is cached, logged, or written to disk, and
the shared default agent (if this deployment funds one) is never touched. If a
deployment ships with no LLM key configured at all, it's BYOK-only by design —
requests without a key get a clear message instead of a call billed to the deployer.

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
#   One of: OPENROUTER_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY /
#           OPENAI_API_KEY / ANTHROPIC_API_KEY (matching LLM_PROVIDER)
#   TAVILY_API_KEY      (needed for the web search tool)
#
# Leave all five LLM keys blank to run BYOK-only locally too — every chat
# request without its own key gets a clear "bring your own key" message.

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
| `POST` | `/api/evaluate` | Triggers RAGAS evaluation in the background — **disabled by default** (`ENABLE_EVALUATE_ENDPOINT=false`); it would otherwise let anyone trigger a batch of paid LLM calls with no per-caller check |
| `GET` | `/api/health` | Health check |

`ChatRequest` (`/api/chat` and `/api/chat/sync`) accepts an optional
`api_key` / `provider` (`openrouter` \| `groq` \| `google` \| `openai` \|
`anthropic`) / `model` — see [BYOK](#architecture-v06) above.

```bash
# Streaming chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What AI practices are prohibited under Article 5?"}'

# Sync chat
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the fine for a prohibited AI system?"}'

# Sync chat with your own key
curl -X POST http://localhost:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the fine for a prohibited AI system?", "provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."}'

# Get evaluation scores
curl http://localhost:8000/api/metrics
```

---

## Swap the LLM or vector store

Edit `.env` — no code changes needed:

```bash
# Switch LLM provider and model
LLM_PROVIDER=openrouter          # openrouter | groq | google | openai | anthropic
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

Two independent suites, run locally (never triggered by public traffic — see
[REST API](#rest-api) above for why `/api/evaluate` is disabled by default):

```bash
# RAGAS: scores the agent's answers against a 20-question golden dataset
uv run python scripts/run_evaluation.py

# Scope-gate regression: 6 fixed questions checking the agent correctly
# answers in-scope questions and refuses out-of-scope ones
uv run python scripts/run_regression_questions.py

# promptfoo: 30-case regression suite (requires Node.js)
npx promptfoo eval --config evals/promptfoo.yaml
```

The promptfoo suite also runs automatically on every PR to `main` via GitHub Actions.

**Current baseline** (openrouter/deepseek-v4-flash, 20-question golden set,
gpt-4o-mini judge, local sentence-transformers embeddings):

| Metric | Target | Result |
|--------|--------|--------|
| Faithfulness | ≥ 0.80 | **0.865** ✅ |
| Answer Relevancy | ≥ 0.70 | **0.852** ✅ |
| Context Precision | ≥ 0.70 | **0.973** ✅ |
| Context Recall | ≥ 0.70 | **0.975** ✅ |

The 6-question scope-gate regression suite passes 6/6 as well. Getting a
trustworthy number here required fixing three separate bugs, in case you hit
the same ones: `scripts/run_evaluation.py` never wired up the retriever
(so the agent silently fell back to the LLM's own memorized knowledge
instead of the retrieved documents), RAGAS's default judge construction
is broken for this `ragas`/`langchain-openai` combo (its auto-factory
returns a mismatched embeddings interface), and each evaluation question
must get its own conversation `thread_id` (a shared one lets LangGraph's
checkpointer accumulate every prior question into each subsequent
call, ballooning context until the judge model's context window
overflows). All three are fixed in `evaluation/harness.py` and
`scripts/run_evaluation.py`.

### Fine-tuned RegClassifier: before/after

`RegClassifier` (risk-tier classification: Unacceptable / High / Limited /
Minimal) can run as a QLoRA-fine-tuned checkpoint or fall back to an
LLM-prompted classifier. On an 11-example held-out test set:

| | Accuracy | Macro F1 |
|---|---|---|
| Base model | 0.818 | 0.672 |
| Fine-tuned | 0.818 | 0.798 |

Accuracy is identical; macro F1 improved because the fine-tuned model is
more balanced across classes (better recall on the minority `Limited`
class), not because it's a clean win — recall on `High` actually dropped.
**n=11 is small enough that none of this should be read as a strong
result** — it's the honest number from the test set that exists today, not
a claim of validated real-world improvement.

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

The Space uses `sdk: docker` — HF builds from `Dockerfile` directly, pre-baking the FAISS index
into the image. No separate `app_file` is needed.

**Secrets** — add in Space Settings → Variables and Secrets (never commit to repo).
Decide first whether this deployment funds a shared default LLM key or runs
BYOK-only (leave all five LLM keys below unset — visitors without their own
key get a clear message instead of a call billed to you):

| Secret | Value | Required for |
|--------|-------|-------------|
| `OPENROUTER_API_KEY` / `GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | your provider key | Funding a shared default LLM (matching `LLM_PROVIDER`) — leave unset for BYOK-only |
| `TAVILY_API_KEY` | from tavily.com | `search_web` tool |
| `VECTOR_STORE_BACKEND` | `faiss` | Use baked-in FAISS index (no ChromaDB sidecar) |
| `ENABLE_EVALUATE_ENDPOINT` | `false` (or leave unset) | Keep `/api/evaluate` disabled — see [REST API](#rest-api) |

**Upload:**

```bash
huggingface-cli login
huggingface-cli upload <your-username>/<space-name> . --repo-type=space
```

The FAISS index is built at Docker image build time (`RUN python scripts/ingest.py` in
`Dockerfile`), so HF Spaces cold starts are fast — no per-request ingest delay.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Package manager | uv |
| API server | FastAPI + Uvicorn (SSE streaming) |
| Agent framework | LangGraph — stateful graph with `MemorySaver` checkpointing |
| LLM | DeepSeek V4 Flash via OpenRouter by default; OpenAI, Anthropic, Groq, Google Gemini, or OpenRouter via BYOK per-request or `.env` |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, no API cost) |
| Vector store | FAISS (dev) / ChromaDB (persistent) — swapped via config |
| Tools | LangChain `@tool` — regulation search, CSV lookup, Tavily web search |
| Document loading | LlamaIndex + PyMuPDF |
| Guardrails | Chain of Responsibility — faithfulness, citation, legal-claim checks |
| Evaluation | RAGAS + 6-question scope-gate regression + promptfoo 30-case suite |
| UI | Gradio 6 — Chat tab with BYOK provider/model/key selector |
| CI | GitHub Actions — promptfoo eval on every PR to `main` |
| Deployment | HuggingFace Spaces |
