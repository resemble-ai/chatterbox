import logging
import json

import torch
from pathlib import Path
from unicodedata import category, normalize
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(" ", SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(" ", "")
        txt = txt.replace(SPACE, " ")
        txt = txt.replace(EOT, "")
        txt = txt.replace(UNK, "")
        return txt


# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
_kakasi = None
_dicta = None
_russian_stresser = None


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi

    try:
        if _kakasi is None:
            import pykakasi

            _kakasi = pykakasi.kakasi()

        result = _kakasi.convert(text)
        out = []

        for r in result:
            inp = r["orig"]
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif (
                all([is_katakana(c) for c in inp]) if inp else False
            ):  # Safety check for empty inp
                out.append(r["orig"])

            else:
                out.append(inp)

        normalized_text = "".join(out)

        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata

        normalized_text = unicodedata.normalize("NFKD", normalized_text)

        return normalized_text

    except ImportError:
        logger.warning("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta

    try:
        if _dicta is None:
            from dicta_onnx import Dicta

            _dicta = Dicta()

        return _dicta.add_diacritics(text)

    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""

    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ("\uac00" <= char <= "\ud7af"):
            return char

        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ""

        return initial + medial + final

    # Decompose syllables and normalize punctuation
    result = "".join(decompose_hangul(char) for char in text)
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""

    def __init__(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()

    def _load_cangjie_mapping(self, model_dir=None):
        """Load Cangjie mapping from HuggingFace model repository."""
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID, filename="Cangjie5_TC.json", cache_dir=model_dir
            )

            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)

        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")

    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from spacy_pkuseg import pkuseg

            self.segmenter = pkuseg()
        except ImportError:
            logger.warning(
                "pkuseg not available - Chinese segmentation will be skipped"
            )
            self.segmenter = None

    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)

    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text

        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def add_russian_stress(text: str) -> str:
    """Russian text normalization: adds stress marks to Russian text."""
    global _russian_stresser

    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser

            _russian_stresser = RussianTextStresser()

        return _russian_stresser.stress_text(text)

    except ImportError:
        logger.warning(
            "russian_text_stresser not available - Russian stress labeling skipped"
        )
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text


class MTLTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        self.check_vocabset_sot_eot()
        # Enable debug logging for tokenizer
        self._debug = True

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def preprocess_text(
        self,
        raw_text: str,
        language_id: str = None,
        lowercase: bool = True,
        normalize_unicode: bool = True,
    ):
        """
        Text preprocessor that handles lowercase conversion and Unicode normalization.

        Uses NFC normalization (composed form) instead of NFKD (decomposed) because:
        1. The vocabulary contains composed accented characters (á, é, í, ó, ú, ñ, etc.)
        2. NFKD decomposes these into base + combining marks, using different tokens
        3. The model was likely trained on NFC-normalized text (more common)
        4. Using composed form produces better pronunciation for Spanish, French, Portuguese, etc.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if normalize_unicode:
            # Use NFC (Canonical Decomposition, followed by Canonical Composition)
            # This keeps accented characters as single codepoints (e.g., á stays as U+00E1)
            # rather than decomposing to base + combining mark (a + U+0301)
            preprocessed_text = normalize("NFC", preprocessed_text)

        return preprocessed_text

    def text_to_tokens(
        self,
        text: str,
        language_id: str = None,
        lowercase: bool = True,
        normalize_unicode: bool = True,
    ):
        text_tokens = self.encode(
            text,
            language_id=language_id,
            lowercase=lowercase,
            normalize_unicode=normalize_unicode,
        )
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(
        self,
        txt: str,
        language_id: str = None,
        lowercase: bool = True,
        normalize_unicode: bool = True,
    ):
        original_txt = txt
        txt = self.preprocess_text(
            txt,
            language_id=language_id,
            lowercase=lowercase,
            normalize_unicode=normalize_unicode,
        )

        # DEBUG: Log preprocessing results
        if self._debug:
            logger.info(
                f"[MTLTokenizer DEBUG] Original text: {repr(original_txt[:200])}"
            )
            logger.info(
                f"[MTLTokenizer DEBUG] After preprocess (NFC): {repr(txt[:200])}"
            )
            # Show character-by-character differences for debugging
            if original_txt.lower() != txt:
                diff_chars = []
                original_lower = original_txt.lower()
                for i, (c1, c2) in enumerate(zip(original_lower, txt)):
                    if c1 != c2:
                        diff_chars.append(
                            f"pos {i}: '{c1}' (U+{ord(c1):04X}) -> '{c2}' (U+{ord(c2):04X})"
                        )
                if diff_chars:
                    logger.info(
                        f"[MTLTokenizer DEBUG] Character changes: {diff_chars[:10]}"
                    )

        # Language-specific text processing
        if language_id == "zh":
            txt = self.cangjie_converter(txt)
        elif language_id == "ja":
            txt = hiragana_normalize(txt)
        elif language_id == "he":
            txt = add_hebrew_diacritics(txt)
        elif language_id == "ko":
            txt = korean_normalize(txt)
        elif language_id == "ru":
            txt = add_russian_stress(txt)

        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"

        txt = txt.replace(" ", SPACE)

        # DEBUG: Log final text and tokenization
        if self._debug:
            logger.info(
                f"[MTLTokenizer DEBUG] Final text for tokenization: {repr(txt[:200])}"
            )

        token_ids = self.tokenizer.encode(txt).ids

        # DEBUG: Log token count and decoded result
        if self._debug:
            logger.info(f"[MTLTokenizer DEBUG] Token count: {len(token_ids)}")
            # Decode tokens to see what the model will "see"
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            logger.info(f"[MTLTokenizer DEBUG] Decoded tokens: {repr(decoded[:200])}")
            # Show any UNK tokens
            unk_id = self.tokenizer.get_vocab().get(UNK, -1)
            unk_positions = [i for i, t in enumerate(token_ids) if t == unk_id]
            if unk_positions:
                logger.warning(
                    f"[MTLTokenizer DEBUG] UNK tokens at positions: {unk_positions}"
                )

        return token_ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(" ", "").replace(SPACE, " ").replace(EOT, "").replace(UNK, "")
        return txt
