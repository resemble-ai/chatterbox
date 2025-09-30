"""Utilities for loading and searching documents from the local data directory."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


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
        self._tokenized_documents: List[List[str]] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._document_embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
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
                tokens = _tokenize(clean_chunk)
                if not tokens:
                    continue
                order += 1
                self._documents.append(
                    DocumentChunk(text=clean_chunk, source=str(file_path.name), order=order)
                )
                self._tokenized_documents.append(tokens)

        if self._documents:
            self._build_embeddings()

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

        if not self._documents or self._document_embeddings.size == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        total = float(sum(query_counts.values()))
        if total == 0.0:
            return []

        query_vector = np.zeros(len(self._vocab), dtype=np.float32)
        for token, count in query_counts.items():
            index = self._vocab.get(token)
            if index is None:
                continue
            query_vector[index] = (count / total) * self._idf[index]

        norm = float(np.linalg.norm(query_vector))
        if norm == 0.0:
            return []
        query_vector /= norm

        scores = self._document_embeddings @ query_vector
        if scores.size == 0:
            return []

        ranked_indices = np.argsort(scores)[::-1]
        results: List[DocumentChunk] = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score <= 0.0:
                break
            doc = self._documents[idx]
            results.append(
                DocumentChunk(text=doc.text, source=doc.source, order=doc.order, score=score)
            )

        return results

    def _build_embeddings(self) -> None:
        vocab: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokenized_documents:
            document_frequency.update(set(tokens))
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if not vocab:
            self._vocab = {}
            self._idf = np.zeros(0, dtype=np.float32)
            self._document_embeddings = np.zeros((len(self._documents), 0), dtype=np.float32)
            return

        vocab_size = len(vocab)
        doc_count = len(self._tokenized_documents)
        tf_matrix = np.zeros((doc_count, vocab_size), dtype=np.float32)

        for row, tokens in enumerate(self._tokenized_documents):
            token_counts = Counter(tokens)
            total_tokens = float(sum(token_counts.values()))
            if total_tokens == 0.0:
                continue
            for token, count in token_counts.items():
                index = vocab[token]
                tf_matrix[row, index] = count / total_tokens

        idf_values = np.array(
            [document_frequency[token] for token in vocab], dtype=np.float32
        )
        idf = np.log((1.0 + doc_count) / (1.0 + idf_values)) + 1.0
        embeddings = tf_matrix * idf

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms

        self._vocab = vocab
        self._idf = idf
        self._document_embeddings = embeddings
