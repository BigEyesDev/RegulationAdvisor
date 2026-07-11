"""
Chunking strategies — Strategy pattern.

Swap chunkers by changing the class — interface is identical across all.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from regulation_advisor.models import RegulationChunk

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    def chunk(self, text: str, source: str) -> list[RegulationChunk]: ...


class RecursiveCharacterChunker:
    """Baseline. Fast but breaks articles mid-sentence. Use for comparison only."""

    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> list[RegulationChunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        texts = RecursiveCharacterTextSplitter(
            chunk_size=self.size, chunk_overlap=self.overlap
        ).split_text(text)
        return [RegulationChunk(content=t, article_number="unknown",
                                article_title="", source_document=source) for t in texts]


class ArticleAwareChunker:
    """
    Recommended for legal text.
    Splits by EU AI Act Article boundaries — each chunk = one complete article.
    """

    # Leading \n anchors the match to a standalone "Article N" header line,
    # avoiding false matches on inline citations like "see Article 5(1)".
    ARTICLE_PATTERN = re.compile(r"\n(Article\s+(\d+[a-z]?)\s*\n([^\n]+)\n)", re.IGNORECASE)

    def chunk(self, text: str, source: str) -> list[RegulationChunk]:
        chunks: list[RegulationChunk] = []
        matches = list(self.ARTICLE_PATTERN.finditer(text))

        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[match.start():end].strip()
            if len(content) >= 50:
                chunks.append(RegulationChunk(
                    content=content,
                    article_number=match.group(2),
                    article_title=match.group(3).strip(),
                    source_document=source,
                ))
        logger.info("ArticleAwareChunker: %d chunks from %s", len(chunks), source)
        return chunks
