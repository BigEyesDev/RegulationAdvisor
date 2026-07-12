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

    # Web search
    tavily_api_key: str = ""

    # OpenAI (used by RAGAS evaluation in Week 3)
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

    # Vector store — swap "faiss" → "chromadb" in Week 4
    vector_store_backend: str = "faiss"
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Single instance — import this everywhere
settings = Settings()
