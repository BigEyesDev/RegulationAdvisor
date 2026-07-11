# Use uv's official Docker image as the base for the builder stage
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first (Docker cache optimization)
# If pyproject.toml doesn't change, this layer is cached
COPY pyproject.toml ./

# Install dependencies into /app/.venv
# --frozen: fail if uv.lock is out of date (ensures reproducibility)
# --no-dev: skip dev dependencies in production
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# ── Final image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ src/
COPY data/ data/
COPY evals/ evals/
COPY scripts/ scripts/

# Make the venv's Python the default
ENV PATH="/app/.venv/bin:$PATH"

# Pre-build the vector index at container build time
RUN python scripts/ingest.py

EXPOSE 8000
CMD ["uvicorn", "regulation_advisor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
