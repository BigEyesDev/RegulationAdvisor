# Changelog

All notable changes to RegulationAdvisor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2026-07-12

**Week 1 complete: full RAG pipeline from raw PDFs to a live Gradio chatbot.**

### Added

#### Ingestion (F1 – F3)
- `PDFLoader`, `CSVLoader`, `MarkdownLoader` with `DocumentLoaderFactory` (Strategy pattern)
- `ArticleAwareChunker` — regex-based chunker that splits legal text at `Article N` boundaries,
  extracting `article_number` and `article_title` as metadata on every chunk
- `RecursiveCharacterChunker` — configurable fallback chunker (size / overlap from `.env`)
- `SentenceTransformerEmbedder` — local `all-MiniLM-L6-v2` model, no API key required
- `FAISSVectorStore` — in-memory vector store with `save()` / `load()` round-trip
- `ChromaDBVectorStore` — stub for Week 4 migration (swap via `VECTOR_STORE_BACKEND` in `.env`)
- `Retriever` — wraps embedder + store; `search(query, k)` returns `RetrievalResult`
- `run_ingestion()` pipeline — reads all files in `data/`, chunks, embeds, saves FAISS index
- `scripts/ingest.py` — CLI entry point for building the index
- 213 chunks indexed: 114 from EU AI Act, 99 from GDPR
- 14 unit tests + 2 integration tests (skip gracefully if index not built)
- Config-driven chunker strategy: `CHUNKER_STRATEGY=article_aware` or `recursive`

#### RAG Chain (F4)
- `src/regulation_advisor/prompts/system_prompt.txt` — grounding prompt that forces
  Article citations and prohibits answers outside the provided context
- `_build_llm()` — provider factory supporting `openrouter`, `groq`, and `google`;
  switch provider and model with two lines in `.env`, no code changes needed
- `_build_chain()` — `ChatPromptTemplate | LLM | StrOutputParser` LCEL pipeline
- `_format_context()` — labels each retrieved chunk as `[source — Article N]` for citation
- `respond()` — closure that wires retrieval → context formatting → LLM generation

#### Gradio UI (F5)
- `build_ui(retriever)` — `gr.Blocks` with `gr.ChatInterface`; returns cited answers
- `src/regulation_advisor/ui/app_runner.py` — startup entry point:
  auto-ingests on cold start, resolves paths via `__file__`, launches with
  `server_name="0.0.0.0"` for container/HF Spaces compatibility

#### HuggingFace Spaces (F6)
- `README.md` — HF Spaces YAML front-matter (`sdk: gradio 6.20.0`, `app_file`)
- `requirements.txt` — pip-compatible dependency list for HF Spaces install
- `_ensure_index()` in `app_runner.py` — auto-runs ingestion on HF cold start

### Changed
- LLM default switched from `groq / qwen-qwen3-32b` (6k TPM free tier) to
  `openrouter / deepseek/deepseek-v4-flash` (1M token context, no TPM cap)
- `config.py` — added `openrouter_base_url` setting
- `.gitignore` — internal learning docs and `data/index/` excluded from git

### Infrastructure
- Package manager: `uv` with `pyproject.toml`
- Linting: `ruff` (E, F, I, UP, B rules)
- Type checking: `mypy --strict`
- Tests: `pytest` with `asyncio_mode = auto`

---

## [0.0.1] — 2026-07-11

**Day 1 scaffold.**

- Project structure: `src/regulation_advisor/` layout
- `config.py` — `pydantic-settings` with `.env` support
- `models.py` — `RegulationChunk`, `RegulationFinding`, `RetrievalResult`
- `loaders.py`, `chunkers.py`, `pipeline.py`, `embeddings.py`, `store.py`, `retriever.py`
- 14 unit tests passing on first commit
- GitHub repository initialised
