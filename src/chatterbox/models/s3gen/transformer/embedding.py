import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        self.pe = self.pe.to(x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int,
                          apply_dropout: bool = True) -> torch.Tensor:
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset:offset + size]
        else:
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + \
                torch.arange(0, size).to(offset.device)
            flag = index > 0
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class WhisperPositionalEncoding(PositionalEncoding):

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 1500):
        super().__init__(d_model, dropout_rate, max_len)
        self.xscale = 1.0
        log_timescale_increment = np.log(10000) / (d_model // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment *
                                   torch.arange(d_model // 2))
        scaled_time = torch.arange(max_len)[:, np.newaxis] * \
            inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        delattr(self, "pe")
        self.register_buffer("pe", pe.unsqueeze(0))


class LearnablePositionalEncoding(PositionalEncoding):

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 448):
        super().__init__(d_model, dropout_rate, max_len)
        self.pe = torch.nn.Parameter(torch.empty(1, max_len, d_model))
        self.xscale = 1.0


class NoPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                x: torch.Tensor,
                offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
        return self.dropout(x), pos_emb

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return torch.zeros(1, size, self.d_model)


class EspnetRelPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1: self.pe.size(1) // 2 + size,
        ]
        return pos_emb