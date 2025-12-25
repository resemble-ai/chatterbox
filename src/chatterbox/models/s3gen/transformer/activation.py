import torch
from torch import nn, sin, pow
from torch.nn import Parameter


class Swish(torch.nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class Snake(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):

        super(Snake, self).__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):

        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x
