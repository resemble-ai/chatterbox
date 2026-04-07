import torch
from transformers.generation.logits_process import MinPLogitsWarper


class FastMinPLogitsWarper(MinPLogitsWarper):
    def __init__(self, *args, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.min_p = torch.tensor(self.min_p, device=device)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        top_probs, top_indices = probs.max(dim=-1, keepdim=True)
        tokens_to_remove = probs < self.min_p * top_probs

        # Keep at least the top token (min_tokens_to_keep=1).
        # Uses scatter_ instead of argsort+gather+slice — argsort allocates
        # temporary memory which is not permitted during CUDA graph capture.
        tokens_to_remove.scatter_(1, top_indices, 0)

        return scores.masked_fill(tokens_to_remove, self.filter_value)
