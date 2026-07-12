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

You type a question. The app finds the most relevant articles from the EU AI Act and GDPR,
passes them to an LLM, and returns a cited answer — grounded in the actual regulation text, not model memory.

```
"What AI practices are completely prohibited?"
        ↓
FAISS retrieves: Article 5, Article 6, Article 7 …
        ↓
LLM reads those articles and answers:
"According to Article 5, the following AI practices are prohibited: ..."
```

No hallucination. Every claim is traceable to a specific Article.

---

## Architecture

```
User question
      │
      ▼
SentenceTransformer encodes question → 384-dimensional vector
      │
      ▼
FAISS searches 213 pre-indexed regulation chunks (EU AI Act + GDPR)
      │
      ▼
Top 5 chunks passed to LLM with system prompt:
"Answer ONLY from the provided text. Always cite the Article number."
      │
      ▼
Cited answer displayed in Gradio ChatInterface
```

---

## Run locally

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), an [OpenRouter API key](https://openrouter.ai)

```bash
# 1. Clone and install
git clone https://github.com/BigEyesDev/RegulationAdvisor.git
cd RegulationAdvisor
uv sync --group dev

# 2. Configure API keys
cp .env.example .env
# Edit .env — set OPENROUTER_API_KEY

# 3. Download the regulation documents into data/
#    eu_ai_act.pdf, gdpr.pdf, and three CSV files
#    (see data/README.md for download links)

# 4. Build the FAISS index (~20 seconds, one time only)
uv run python scripts/ingest.py

# 5. Launch the chatbot
uv run python src/regulation_advisor/ui/app_runner.py
# Open http://localhost:7860
```

---

## Swap the LLM model

Edit two lines in `.env` — no code changes needed:

```bash
LLM_PROVIDER=openrouter          # openrouter | groq | google
LLM_MODEL=deepseek/deepseek-v4-flash   # any model slug for that provider
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Package manager | uv |
| LLM | DeepSeek V4 Flash via OpenRouter |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (local, no API) |
| Vector store | FAISS |
| LLM framework | LangChain (LCEL chain) |
| Document loading | LlamaIndex + PyMuPDF |
| UI | Gradio 6 |
| Deployment | HuggingFace Spaces |

---

## Run tests

```bash
uv run pytest tests/unit/ -v          # 14 unit tests, no data files needed
uv run pytest tests/integration/ -v  # requires index to be built first
```
