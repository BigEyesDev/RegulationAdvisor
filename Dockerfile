# ── Stage 1: dependency installer ────────────────────────────────────────────
# uv's own image already has uv on PATH — no pip install needed.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy lock file alongside pyproject so --frozen can verify the lock is current.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Bring only the installed packages from the builder — no uv, no build tools.
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY data/ data/
COPY evals/ evals/
COPY scripts/ scripts/

ENV PATH="/app/.venv/bin:$PATH"

# Build the FAISS vector index once at image-build time.
# VECTOR_STORE_BACKEND defaults to "faiss" so this uses local files only.
RUN python scripts/ingest.py

EXPOSE 8000
CMD ["uvicorn", "regulation_advisor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
