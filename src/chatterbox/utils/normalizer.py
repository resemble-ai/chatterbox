import re

from num2words import num2words


_NUM2WORDS_LANGS = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr",
    "he", "hi", "it", "ja", "ko", "nl", "no", "pl",
    "pt", "ru", "sv", "tr",
}

# Languages that use period as decimal separator (comma is the thousands separator)
# All others in _NUM2WORDS_LANGS use comma as decimal (period is the thousands separator)
_PERIOD_DECIMAL_LANGS = {"ar", "en", "hi", "ja", "ko"}


def normalize_numbers(text: str, language: str = "en") -> str:
    """Replace numbers with language-aware words when supported."""

    lang = language.lower()
    if lang not in _NUM2WORDS_LANGS:
        return text

    def _replace(match):
        raw = match.group()
        if lang in _PERIOD_DECIMAL_LANGS:
            # Comma is thousands separator: 1,000 → 1000; period stays as decimal
            normalized = re.sub(r",(\d{3})(?!\d)", r"\1", raw)
        else:
            # Period is thousands separator: 1.000 → 1000; comma is decimal → period
            normalized = re.sub(r"\.(\d{3})(?!\d)", r"\1", raw)
            normalized = normalized.replace(",", ".")
        try:
            value = float(normalized)
            if value.is_integer():
                value = int(value)
            return num2words(value, lang=lang)
        except (NotImplementedError, TypeError, ValueError, OverflowError):
            return raw

    # Match full numbers including multiple separator groups (e.g. 1,000,000 or 1.000.000,50)
    return re.sub(r"\d+(?:[,\.]\d+)*", _replace, text)


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize text for TTS inference."""
    text = re.sub(r"\s*-{2,}.*$", "", text.strip(), flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    text = normalize_numbers(text, language=language)
    return text.strip()
