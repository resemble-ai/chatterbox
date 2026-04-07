import re

import nltk


# Map ISO 639-1 codes to nltk language names for sent_tokenize
_NLTK_LANGUAGE_MAP = {
    "cs": "czech",
    "da": "danish",
    "nl": "dutch",
    "en": "english",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "de": "german",
    "el": "greek",
    "it": "italian",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "es": "spanish",
    "sv": "swedish",
    "tr": "turkish",
}

nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, language: str = "en") -> list[str]:
    """Split text into sentence-like chunks for TTS."""
    nltk_language = _NLTK_LANGUAGE_MAP.get(language.lower(), "english")
    try:
        parts = nltk.tokenize.sent_tokenize(text, language=nltk_language)
    except LookupError:
        raise RuntimeError(
            "NLTK punkt_tab data not found. Download it with: "
            "python -c \"import nltk; nltk.download('punkt_tab')\""
        ) from None
    result = []
    for part in parts:
        part = re.sub(r"^[-–—]\s+", "", part.strip())
        if part:
            result.append(part)
    return result
