from pathlib import Path
import logging
import json

from unicodedata import category
import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)


def hp_vocab_consistency_check(tokenizer, hp):
    """
    Consistency test for start/end of text tokens between hparam & vocab.
    """
    voc = tokenizer.get_vocab()
    sot_idx = voc.get(SOT, len(voc))
    eot_idx = voc.get(EOT, 0)
    if hp.start_text_token != sot_idx:
        msg = (f"Inconsistent <SOT> token indices: {sot_idx} in vocab "
               f"vs. {hp.start_text_token} in hparam")
        logger.warning(msg)

    if hp.stop_text_token != eot_idx:
        msg = (f"Inconsistent <EOT> token indices: {eot_idx} in vocab "
               f"vs. {hp.stop_text_token} in hparam")
        logger.warning(msg)


def prepend_language_label(txt, lang_id):
    ipa_tag = ""
    if "IPA" in lang_id:
        ipa_tag = "[IPA]"

    if "cmn" == lang_id:
        lang_id = "zh"

    # We currently don't handle language variants, accents, or dialects.
    if lang_id:
        lang_id = lang_id.split("-")[0]
        if len(lang_id) > 0:
            lang_label = f"[{lang_id}]"
    else:  # None or empty string
        # allow empty `lang_id` for code switching in the future
        lang_id = ""

    txt = lang_label + ipa_tag + txt
    return txt


def debug_tokenizer(code, input_text=None, unk_id=1):
    """
    Print text in whie and token ID in yellow while highlighting UNKs.
    """
    from colorama import Fore, Style
    ids = code.ids
    toks = code.tokens
    DEFAULT_STYLE = f"{Fore.WHITE}{Style.DIM}｜{Fore.RESET}{Style.NORMAL}"
    if input_text is None:
        toks = [(t if t != SPACE else " ") for t in toks]
        toks = [t if "M" not in category(t[0]) else f" {t}" for t in toks]
        print(
            DEFAULT_STYLE
                .join(f"{t}{Fore.YELLOW}{i}{Fore.RESET}"
                    for t, i in zip(toks, ids))
        )
    else:
        offsets = code.offsets
        output = []
        for i, (s, e) in zip(ids, offsets):
            t = input_text[s:e]
            print_id = f"{Fore.YELLOW}{i}{Fore.RESET}{Style.NORMAL}"
            if i == unk_id:
                o = f"{Style.BRIGHT}{t}{print_id}"
            elif t == SPACE:
                t = " "
                o = f"{Style.DIM}{t}{print_id}"
            else:
                o = f"{Style.DIM}{t}{print_id}"
            output.append(o)
        print(DEFAULT_STYLE.join(output))


# TODO: remove: just use Tokenizer & pass path
class VBpeConfig:
    symbol_type = "checkpoints/grapheme_en_common.json"

    # TODO: remove; get them from vocab
    start_text_token = 255
    stop_text_token = 0
    max_text_tokens = 2048


class VoiceBpeTokenizer:
    chinese_lang_ids = "cmn zh zh-tw zh-cn".split()
    def __init__(self, hp=VBpeConfig()):
        """
        Args:
        hp: the following attributes are used:
            - symbol_type: for tokenizer file path mapping
            - stop/start_text_token: for consistency check
            - text_preproc: (str) instructions for text normalization
        """
        # self.hp = default_hp if hp is None else hp
        self.hp = hp

        self.languages = []
        self.use_phonemes = False

        symbol_type = self.hp.symbol_type

        if Path(symbol_type).exists():
            # Fall back to experimental/arbitrary vocab
            vocab_file_path = symbol_type
        else:
            raise NotImplementedError(f"{symbol_type} not recognised")

        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        hp_vocab_consistency_check(self.tokenizer, self.hp)

    def is_multilingual(self):
        return len(self.languages) > 0


    def clean_text(self, raw_text, lang_id="en-us", override=None):
        """TODO: we're not using `clean_text` anymore"""
        return raw_text

    def encode(
        self,
        txt,
        clean_symbols=True,
        lang_id="en-us",
        prepend_lang_id=True,
        verbose=False
    ):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        if verbose or any(i == 1 for i in ids):
            debug_tokenizer(code, txt, unk_id=self.tokenizer.token_to_id(UNK))
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
