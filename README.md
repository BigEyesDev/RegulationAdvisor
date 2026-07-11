# RegulationAdvisor

**Agentic RAG system for EU AI Act compliance.**
Ask compliance questions in natural language. Get cited, guardrailed answers.

> EU AI Act becomes fully enforceable 2 August 2026.

---

## Day 1 Setup (uv)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv

# 2. Extract the project and enter it
cd regulation-advisor

# 3. Pin Python version and create virtual environment
uv python pin 3.11
uv sync --group dev

# 4. Copy and fill in your API keys
cp .env.example .env
# Open .env and add: GROQ_API_KEY, TAVILY_API_KEY, HUGGINGFACE_TOKEN

# 5. Verify everything works
uv run pytest tests/unit/ -v
# Expected: 7 tests pass immediately (no data files needed)
```

---

## Daily Commands

```bash
# Run tests
uv run pytest tests/unit/ -v
uv run pytest tests/ --cov=src --cov-report=term-missing

# Add a new package
uv add langchain-openai

# Add a dev-only package
uv add --group dev httpx

# Run any script
uv run python scripts/ingest.py

# Start the app (Week 4+)
uv run uvicorn regulation_advisor.api.app:app --reload

# Lint and format
uv run ruff check src/
uv run ruff format src/
```

---

## Build Sequence

| Week | What gets built |
|------|----------------|
| 1 | Document ingestion + FAISS + Gradio UI → live on HF Spaces |
| 2 | LangGraph agent (3 tools) + human-in-the-loop + smolagents comparison |
| 3 | RAGAS evaluation + guardrail layer + promptfoo regression tests |
| 4 | FastAPI REST API + ChromaDB + streaming responses |
| 5 | Docker + HF Spaces + AWS ECS deployment |
| 6 | QLoRA fine-tuning → RegClassifier on HuggingFace Hub |
| 7 | Polish, articles, interview prep |

---

## Architecture

```
User Query
    ↓
Gradio UI / FastAPI (/api/chat)
    ↓
LangGraph Agent
    ├── RAG Search Tool      → ChromaDB + sentence-transformers
    ├── CSV Query Tool       → Timeline, penalties, risk classification
    └── Web Search Tool      → Tavily (enforcement news)
    ↓
RegClassifier (Week 6)       → Risk tier + obligation type + deadline
    ↓
Guardrail Layer              → Faithfulness + citation + legal claim checks
    ↓
Validated Answer with citations
```

## Tech Stack

| Layer | Tool |
|-------|------|
| Package manager | **uv** |
| LLM (free) | Qwen3-32B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector store | FAISS (dev) → ChromaDB (prod) |
| Orchestration | LangGraph |
| Document loading | LlamaIndex |
| Evaluation | RAGAS + promptfoo |
| API | FastAPI |
| UI | Gradio |
| Fine-tuning | Unsloth + QLoRA |
| Deployment | Docker + HF Spaces + AWS ECS |
