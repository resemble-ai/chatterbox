"""Split narration into bounded chunks without dropping words."""
import re


def split_text(text: str, max_chars: int = 250) -> list[str]:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError('max_chars must be positive')
    text = ' '.join(text.split())
    if not text:
        raise ValueError('Text must not be empty')
    chunks = []
    current = ''
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(sentence) <= max_chars:
            units = [sentence]
        else:
            units = sentence.split()
        for unit in units:
            if len(unit) > max_chars:
                raise ValueError('A single word exceeds max_chunk_chars')
            candidate = f'{current} {unit}' if current else unit
            if len(candidate) > max_chars:
                chunks.append(current)
                current = unit
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks
