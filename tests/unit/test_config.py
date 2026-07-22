"""
Gate check — verifies Settings class defaults and that the live settings load
without error.

We test Settings.model_fields (code-level defaults) rather than the live
settings instance so the tests are independent of whatever .env file is on
disk. The .env file is user/machine configuration — not something we assert.
"""
from regulation_advisor.config import Settings, settings


def test_llm_provider_default():
    assert Settings.model_fields["llm_provider"].default == "openrouter"


def test_llm_model_default():
    assert Settings.model_fields["llm_model"].default == "deepseek/deepseek-v4-flash"


def test_embedding_model_default():
    assert Settings.model_fields["embedding_model"].default == "all-MiniLM-L6-v2"


def test_vector_store_backend_default():
    assert Settings.model_fields["vector_store_backend"].default == "faiss"


def test_retrieval_defaults():
    assert Settings.model_fields["chunk_size"].default == 1000
    assert Settings.model_fields["chunk_overlap"].default == 200
    assert Settings.model_fields["retrieval_k"].default == 5


def test_settings_loads_without_error():
    assert settings.llm_provider != ""
    assert settings.llm_model != ""
    assert settings.embedding_model != ""


def test_has_default_llm_key_false_when_provider_key_empty():
    s = Settings(
        groq_api_key="", openrouter_api_key="", google_api_key="", llm_provider="groq"
    )
    assert s.has_default_llm_key is False


def test_has_default_llm_key_true_when_provider_key_set():
    s = Settings(
        groq_api_key="fake", openrouter_api_key="", google_api_key="", llm_provider="groq"
    )
    assert s.has_default_llm_key is True


def test_has_default_llm_key_only_checks_active_provider():
    # A key for a *different* provider than llm_provider must not count.
    s = Settings(
        groq_api_key="", openrouter_api_key="fake", google_api_key="", llm_provider="groq"
    )
    assert s.has_default_llm_key is False


def test_has_default_llm_key_true_for_openai_provider():
    s = Settings(openai_api_key="fake", llm_provider="openai")
    assert s.has_default_llm_key is True


def test_has_default_llm_key_true_for_anthropic_provider():
    s = Settings(anthropic_api_key="fake", llm_provider="anthropic")
    assert s.has_default_llm_key is True


def test_enable_evaluate_endpoint_defaults_false():
    assert Settings.model_fields["enable_evaluate_endpoint"].default is False
