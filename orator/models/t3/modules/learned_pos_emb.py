from typing import Union

import torch
from torch import nn, Tensor


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        """
        Returns positional embeddings for index 0 up to the length of x
        """
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, idx: 'Union[int, Tensor]'):
        """
        Args:
            idx: scalar int or an integer tensor of shape (T,) or (B, T)
        Returns:
            positional embeddings for given indices, shape (B, T, dim), ie (1, 1, dim) for int input
        """
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx = torch.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx)  # (B, T, dim)
