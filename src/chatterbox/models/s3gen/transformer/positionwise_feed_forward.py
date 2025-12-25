import torch

class PositionwiseFeedForward(torch.nn.Module):

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class MoEFFNLayer(torch.nn.Module):
    def __init__(
            self,
            n_expert: int,
            n_expert_per_token: int,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super(MoEFFNLayer, self).__init__()
        self.gate = torch.nn.Linear(idim, n_expert, bias=False)
        self.experts = torch.nn.ModuleList(
            PositionwiseFeedForward(idim, hidden_units, dropout_rate,
                                    activation) for _ in range(n_expert))
        self.n_expert_per_token = n_expert_per_token

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        B, L, D = xs.size(
        )
        xs = xs.view(-1, D)
        router = self.gate(xs)
        logits, indices = torch.topk(
            router, self.n_expert_per_token
        )
        weights = torch.nn.functional.softmax(
            logits, dim=1,
            dtype=torch.float).to(dtype=xs.dtype)
        output = torch.zeros_like(xs)
        for i, expert in enumerate(self.experts):
            mask = indices == i
            batch_idx, ith_expert = torch.where(mask)
            output[batch_idx] += weights[batch_idx, ith_expert, None] * expert(
                xs[batch_idx])
        return output.view(B, L, D)
