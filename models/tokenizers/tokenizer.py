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


class VoiceBpeTokenizer:
    def __init__(self, vocab_file_path="checkpoints/grapheme_en_common.json"):
        """
        Args:
        hp: the following attributes are used:
            - symbol_type: for tokenizer file path mapping
            - stop/start_text_token: for consistency check
            - text_preproc: (str) instructions for text normalization
        """
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def clean_text(self, raw_text):
        """TODO: we no longer `clean_text`"""
        return raw_text

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        if verbose and any(i == 1 for i in ids):
            debug_tokenizer(code, txt, unk_id=self.tokenizer.token_to_id(UNK))
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
