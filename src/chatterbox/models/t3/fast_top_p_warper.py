import torch
from transformers.generation.logits_process import TopPLogitsWarper


class FastTopPLogitsWarper(TopPLogitsWarper):
    def __init__(self, *args, device="cuda", skip_when_1=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_p = torch.tensor(self.top_p, device=device)
        self.zero_tensor = torch.tensor(0, device=device)
        self.skip_when_1 = skip_when_1

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.skip_when_1 and self.top_p == 1.0:
            return scores

        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -1:] = self.zero_tensor

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed
