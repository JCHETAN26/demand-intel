"""Sequence-to-sequence LSTM forecaster F4 (PROJECT_SPEC.md §5.3, plan Phase 14).

Encoder LSTM over the past window (observed sales + known inputs + static embeddings);
its final state initializes a decoder LSTM that runs over the forecast days with the
known-future inputs. Two heads per decoder step: a mean (softplus) and monotone quantiles
(a softplus base plus cumulative softplus increments, so quantiles never cross).
"""

import torch
from torch import nn

from demand_intel.forecasting.evaluation.metrics import QUANTILES
from demand_intel.models.deep_learning.data import ENC, N_KNOWN_CONT, SequenceArrays


def embedding_dim(cardinality: int) -> int:
    return int(min(16, max(2, round(cardinality**0.25 * 2))))


class InputEmbeddings(nn.Module):
    """Static categorical embeddings plus per-day calendar/event embeddings."""

    def __init__(self, arr: SequenceArrays) -> None:
        super().__init__()
        self.static = nn.ModuleList(
            nn.Embedding(c, embedding_dim(c)) for c in arr.static_cardinality
        )
        self.wday = nn.Embedding(7, 3)
        self.month = nn.Embedding(12, 3)
        self.event = nn.Embedding(arr.n_events, 4)
        self.static_dim = sum(embedding_dim(c) for c in arr.static_cardinality)
        self.time_dim = 3 + 3 + 4 + N_KNOWN_CONT

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        static = torch.cat([emb(b["static"][:, j]) for j, emb in enumerate(self.static)], dim=-1)
        time = torch.cat(
            [self.wday(b["wday"]), self.month(b["month"]), self.event(b["event"]), b["known"]],
            dim=-1,
        )
        return static, time  # (B, S), (B, ENC + H, T)


class QuantileHead(nn.Module):
    def __init__(self, hidden: int, n_q: int = len(QUANTILES)) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden, 1)
        self.q = nn.Linear(hidden, n_q)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = nn.functional.softplus(self.mean(h)).squeeze(-1)
        raw = self.q(h)
        base = nn.functional.softplus(raw[..., :1])
        steps = nn.functional.softplus(raw[..., 1:])
        q = torch.cat([base, base + torch.cumsum(steps, dim=-1)], dim=-1)
        return mean, q


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self, arr: SequenceArrays, hidden: int = 64, layers: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.inputs = InputEmbeddings(arr)
        s, t = self.inputs.static_dim, self.inputs.time_dim
        self.encoder = nn.LSTM(
            2 + t + s, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.decoder = nn.LSTM(
            t + s, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0
        )
        self.drop = nn.Dropout(dropout)
        self.head = QuantileHead(hidden)

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        static, time = self.inputs(b)
        enc_t, dec_t = time[:, :ENC], time[:, ENC:]
        s_enc = static[:, None, :].expand(-1, enc_t.shape[1], -1)
        s_dec = static[:, None, :].expand(-1, dec_t.shape[1], -1)
        _, state = self.encoder(torch.cat([b["past"], enc_t, s_enc], dim=-1))
        out, _ = self.decoder(torch.cat([dec_t, s_dec], dim=-1), state)
        return self.head(self.drop(out))
