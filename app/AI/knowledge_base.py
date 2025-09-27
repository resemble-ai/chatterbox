"""Utilities for loading and searching documents from the local data directory."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[\w']+")


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


@dataclass
class DocumentChunk:
    """Represents a small chunk of a larger document."""

    text: str
    source: str
    order: int
    score: float = 0.0


class LocalKnowledgeBase:
    """Loads plain-text files from a folder and provides naive semantic search."""

    def __init__(
        self,
        data_dir: Path,
        *,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        encoding: str = "utf-8",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._data_dir = data_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding = encoding
        self._documents: List[DocumentChunk] = []
        self._load_documents()

    @property
    def documents(self) -> Sequence[DocumentChunk]:
        return tuple(self._documents)

    def _load_documents(self) -> None:
        if not self._data_dir.exists():
            logger.warning("Data directory %s was not found. Knowledge base is empty.", self._data_dir)
            return

        order = 0
        for file_path in sorted(self._data_dir.glob("**/*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".txt", ".md", ".rst"}:
                continue
            text = file_path.read_text(encoding=self._encoding)
            for chunk in self._split_into_chunks(text):
                clean_chunk = chunk.strip()
                if not clean_chunk:
                    continue
                order += 1
                self._documents.append(
                    DocumentChunk(text=clean_chunk, source=str(file_path.name), order=order)
                )

    def _split_into_chunks(self, text: str) -> Iterable[str]:
        tokens = text.split()
        if not tokens:
            return []
        step = self._chunk_size - self._chunk_overlap
        if step <= 0:
            step = self._chunk_size
        chunks = []
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + self._chunk_size)
            chunks.append(" ".join(tokens[start:end]))
            if end == len(tokens):
                break
        return chunks

    def search(self, query: str, *, top_k: int = 5) -> List[DocumentChunk]:
        """Return top document chunks relevant to the query."""

        if not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_token_set = set(query_tokens)
        results: List[DocumentChunk] = []
        for doc in self._documents:
            doc_tokens = _tokenize(doc.text)
            if not doc_tokens:
                continue
            doc_token_set = set(doc_tokens)
            overlap = len(query_token_set & doc_token_set)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(query_token_set) * len(doc_token_set))
            results.append(DocumentChunk(text=doc.text, source=doc.source, order=doc.order, score=score))

        results.sort(key=lambda item: (item.score, -item.order), reverse=True)
        return results[:top_k]
