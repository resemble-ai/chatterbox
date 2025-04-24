from .tokenizer import VoiceBpeTokenizer

# from bisect import bisect_left

# import numpy as np
# import torch
# import torch.nn.functional as F


# class ExcessiveResourceRequest(Exception):
#     arg_names = ("explanation", "error_type")
#     fingerprint = ("error_type",)
#     user_message_template = "Your request could not be handled by our servers. {explanation}"
#     status_code = 400
#     ws_status_code = 1008


# def text_to_tokens(tokenizer: VoiceBpeTokenizer, text: str, phonemize=False, lang_id="en-us", prepend_lang_id=True):
#     """
#     :return: text tokens as an int32 tensor of shape (1, n_tokens)
#     """
#     text_tokens = tokenizer.encode(text)
#     text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)

#     # TODO: this EOT pad is needed for API stream endpoint, are there other codepaths that duplicate this EOT?
#     text_tokens = F.pad(text_tokens, (0, 1), value=tokenizer.hp.stop_text_token)

#     if text_tokens.shape[-1] > tokenizer.hp.max_text_tokens:
#         char_limit = bisect_left(np.arange(len(text)), tokenizer.hp.max_text_tokens, key=lambda i: len(tokenizer.encode(text[:i])) + 1)
#         raise ExcessiveResourceRequest(
#             f"The sentence exceeds the maximum sentence length allowed by {len(text) - char_limit}. Break down the "
#             f"sentence before \"{text[char_limit:char_limit + 30]}(...)\".",
#             "max_tokens_exceeded",
#             internal_message=\
#             f"The sentence \"{text[:30]}(...)\" yields {text_tokens.shape[-1]} tokens when the limit is {tokenizer.hp.max_text_tokens} tokens."
#         )

#     return text_tokens
