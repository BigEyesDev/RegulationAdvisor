"""
Day 1 gate check — verifies Settings loads correctly from environment.
"""
from regulation_advisor.config import settings


def test_llm_provider_default():
    assert settings.llm_provider == "groq"


def test_llm_model_default():
    assert settings.llm_model == "qwen/qwen3-32b"


def test_embedding_model_default():
    assert settings.embedding_model == "all-MiniLM-L6-v2"


def test_vector_store_backend_default():
    assert settings.vector_store_backend == "faiss"


def test_retrieval_defaults():
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.retrieval_k == 5
