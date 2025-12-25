from typing import Union

import torch
from torch import nn, Tensor


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, idx: 'Union[int, Tensor]'):
        device = self.emb.weight.device
        idx = idx.to(device) if torch.is_tensor(idx) else torch.tensor(idx, device=device)
        idx = torch.atleast_2d(idx)
        assert idx.ndim == 2
        return self.emb(idx)
