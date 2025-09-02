import logging
import json
import re

import torch
from pathlib import Path
from unicodedata import category
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

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt


# Model repository
REPO_ID = "ResembleAI/chatterbox-multilingual"

# Global instances for optional dependencies
_kakasi = None
_dicta = None


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
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)
        
        normalized_text = "".join(out)
        
        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
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
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)
    result = re.sub(r'[…~？！，：；（）「」『』]', '.', result)  # Korean punctuation
    
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
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
                cache_dir=model_dir
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
            from pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        code = self.word2cj.get(glyph, None)
        if code is None:
            return None
        
        index = self.cj2word[code].index(glyph)
        index_suffix = str(index) if index > 0 else ""
        return code + index_suffix
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert Arabic numerals (1-99) to Chinese characters."""
        digit_map = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                     '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
        
        pattern = re.compile(r'(?<!\d)(\d{1,2})(?!\d)')
        
        def convert_number(match):
            num = int(match.group(1))
            
            if num == 0:
                return '零'
            elif 1 <= num <= 9:
                return digit_map[str(num)]
            elif num == 10:
                return '十'
            elif 11 <= num <= 19:
                return '十' + digit_map[str(num % 10)]
            elif 20 <= num <= 99:
                tens, ones = divmod(num, 10)
                if ones == 0:
                    return digit_map[str(tens)] + '十'
                else:
                    return digit_map[str(tens)] + '十' + digit_map[str(ones)]
            else:
                return match.group(1)
        
        return pattern.sub(convert_number, text)
    
    def convert_chinese_text(self, text: str) -> str:
        """Convert Chinese characters in text to Cangjie tokens."""
        text = re.sub('[、，：；〜－（）｟｠]', ',', text)
        text = re.sub('(。|…)', '.', text)
        text = self._normalize_numbers(text)
        
        # Skip segmentation for simple sequences (numbers, punctuation, short phrases)
        if self.segmenter is not None:
            # This avoids over-segmenting number sequences like "一, 二, 三"
            is_simple_sequence = (
                len([c for c in text if category(c) == "Lo"]) <= 15 and  # Max 15 Chinese chars
                text.count(',') >= 2  # Contains multiple commas (likely enumeration)
            )
            
            # Only segment complex Chinese text (longer sentences without enumeration patterns)
            if not is_simple_sequence and len(text) > 10:
                chinese_chars = sum(1 for c in text if category(c) == "Lo")
                total_chars = len([c for c in text if c.strip()])
                
                if chinese_chars > 5 and chinese_chars / total_chars > 0.7:
                    segmented_words = self.segmenter.cut(text)
                    text = " ".join(segmented_words)
        
        output = []
        for char in text:
            if category(char) == "Lo":  # Chinese character
                cangjie = self._cangjie_encode(char)
                if cangjie is None:
                    output.append(char)
                    continue
                
                code_tokens = [f"[cj_{c}]" for c in cangjie]
                code_tokens.append("[cj_.]")
                
                output.append("".join(code_tokens))
            else:
                output.append(char)
        
        return "".join(output)


class MTLTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        model_dir = Path(vocab_file_path).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str, language_id: str = None):
        text_tokens = self.encode(text, language_id=language_id)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str, language_id: str = None):
        # Language-specific text processing
        if language_id == 'zh':
            txt = self.cangjie_converter.convert_chinese_text(txt)
        elif language_id == 'ja':
            txt = hiragana_normalize(txt)
        elif language_id == 'he':
            txt = add_hebrew_diacritics(txt)
        elif language_id == 'ko':
            txt = korean_normalize(txt)
        
        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"
        
        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt
