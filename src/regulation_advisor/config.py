"""
Central configuration.
All settings come from environment variables or .env file.
Nothing is hardcoded anywhere else in the codebase.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    # Provider: "openrouter" | "groq" | "google"
    # Switch by changing LLM_PROVIDER in .env — no code changes needed.
    llm_provider: str = "openrouter"
    llm_model: str = "deepseek/deepseek-v4-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Some provider/model combos occasionally stall on a single request for
    # 30-60s+ with no error — this bounds how long any one call can hang.
    llm_request_timeout_seconds: int = 45

    # Web search
    tavily_api_key: str = ""

    # OpenAI (used by RAGAS evaluation harness)
    openai_api_key: str = ""

    # HuggingFace
    huggingface_token: str = ""

    # Embeddings (local — no API key needed)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 5
    # "article_aware" uses ArticleAwareChunker (recommended for legal PDFs)
    # "recursive"     uses RecursiveCharacterChunker with chunk_size / chunk_overlap
    chunker_strategy: str = "article_aware"

    # Vector store — swap "faiss" → "chromadb" for persistence
    vector_store_backend: str = "faiss"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Path where the FAISS index is saved / loaded (ChromaDB ignores this)
    index_dir: str = "data/index"

    # RegClassifier checkpoint: local path (outputs/reg_classifier/final) or a
    # HuggingFace Hub repo id (BigEyesDev/reg-classifier-qwen3-1.7b).
    # Leave empty to use the LLM-prompted fallback classifier instead.
    classifier_checkpoint: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def has_default_llm_key(self) -> bool:
        """
        Whether the configured LLM_PROVIDER has a key to serve requests with,
        with no caller-supplied api_key involved.

        False in a deployment that intentionally ships with no LLM keys in
        its secrets (e.g. a public Space run BYOK-only, so no visitor's usage
        is ever billed to the deployer) — callers use this to require BYOK
        instead of attempting a call that would fail on an empty key.
        """
        key_by_provider = {
            "openrouter": self.openrouter_api_key,
            "groq": self.groq_api_key,
            "google": self.google_api_key,
        }
        return bool(key_by_provider.get(self.llm_provider))


# Single instance — import this everywhere
settings = Settings()
