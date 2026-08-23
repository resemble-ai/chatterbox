"""Natural phrase boundaries for progressive narration synthesis."""

import re
from dataclasses import dataclass
from typing import Literal


Boundary = Literal["clause", "sentence", "paragraph", "final"]


@dataclass(frozen=True)
class NarrationChunk:
    text: str
    boundary: Boundary


def _split_long_sentence(
    sentence: str,
    maximum_characters: int,
    minimum_phrase_characters: int,
) -> list[str]:
    """Split a long sentence at its latest useful clause mark, then words."""
    pieces: list[str] = []
    remaining = sentence.strip()
    while len(remaining) > maximum_characters:
        window = remaining[: maximum_characters + 1]
        candidates = [
            match.end()
            for match in re.finditer(r"[,;:\u2014\u2013](?:\s+|$)", window)
            if match.end() >= minimum_phrase_characters
        ]
        split_at = candidates[-1] if candidates else window.rfind(" ")
        if split_at < minimum_phrase_characters:
            split_at = maximum_characters
        piece = remaining[:split_at].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def narration_chunks(
    text: str,
    maximum_characters: int = 140,
    minimum_phrase_characters: int = 55,
) -> list[NarrationChunk]:
    """Split narration into playable phrases and describe each trailing beat."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text.strip()) if paragraph.strip()]
    chunks: list[NarrationChunk] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = re.findall(r"\S.*?(?:[.!?]+(?:[\"')\]]*)|$)", paragraph, re.DOTALL)
        paragraph_chunks: list[NarrationChunk] = []
        for raw_sentence in sentences:
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            pieces = _split_long_sentence(sentence, maximum_characters, minimum_phrase_characters)
            for piece_index, piece in enumerate(pieces):
                boundary: Boundary = "sentence" if piece_index == len(pieces) - 1 else "clause"
                paragraph_chunks.append(NarrationChunk(piece, boundary))
        if paragraph_chunks and paragraph_index < len(paragraphs) - 1:
            paragraph_chunks[-1] = NarrationChunk(paragraph_chunks[-1].text, "paragraph")
        chunks.extend(paragraph_chunks)

    if chunks:
        chunks[-1] = NarrationChunk(chunks[-1].text, "final")
    return chunks
