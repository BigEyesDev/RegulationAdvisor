"""
Document loaders — Factory pattern.

Add new file types by adding a class + registering it in DocumentLoaderFactory._loaders.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class DocumentLoader(Protocol):
    def load(self, path: Path) -> list[str]: ...


class PDFLoader:
    def load(self, path: Path) -> list[str]:
        from llama_index.readers.file import PyMuPDFReader
        documents = PyMuPDFReader().load(file_path=str(path))
        logger.info("Loaded %d pages from %s", len(documents), path.name)
        return [doc.text for doc in documents]


class CSVLoader:
    def load(self, path: Path) -> list[str]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [", ".join(f"{k}: {v}" for k, v in row.items())
                    for row in csv.DictReader(f)]
        logger.info("Loaded %d rows from %s", len(rows), path.name)
        return rows


class MarkdownLoader:
    def load(self, path: Path) -> list[str]:
        return [path.read_text(encoding="utf-8")]


class DocumentLoaderFactory:
    _loaders: dict[str, type] = {
        ".pdf": PDFLoader,
        ".csv": CSVLoader,
        ".md": MarkdownLoader,
        ".txt": MarkdownLoader,
    }

    @classmethod
    def create(cls, path: Path) -> DocumentLoader:
        loader_class = cls._loaders.get(path.suffix.lower())
        if not loader_class:
            raise ValueError(f"No loader for '{path.suffix}'. Supported: {list(cls._loaders)}")
        return loader_class()

    @classmethod
    def supports(cls, path: Path) -> bool:
        return path.suffix.lower() in cls._loaders
