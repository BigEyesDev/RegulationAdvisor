from pathlib import Path
import pytest
from regulation_advisor.ingestion.loaders import DocumentLoaderFactory

def test_factory_raises_for_unsupported_extension():
    with pytest.raises(ValueError):
        DocumentLoaderFactory.create(Path("file.xyz"))

def test_factory_supports_pdf():
    assert DocumentLoaderFactory.supports(Path("file.pdf")) is True

def test_factory_does_not_support_json():
    assert DocumentLoaderFactory.supports(Path("data.json")) is False
