"""Temporal Fusion Transformer F5 (Lim et al., 2021), compact implementation (plan Phase 15).

Components, as in the paper:
- Gated residual networks (GRN) with GLU gating and layer norm
- Variable selection networks (VSN) for static, past and known-future inputs; their
  softmax weights are returned for interpretation
- Static covariate encoders producing four contexts: variable selection, LSTM initial
  hidden and cell state, and static enrichment
- LSTM encoder-decoder for local processing, gated skip connection
- Interpretable multi-head attention (shared value projection, averaged heads) with a
  causal mask over encoder + decoder positions
- Position-wise GRN and a quantile head (same monotone head as the LSTM)
"""

import torch
from torch import nn

from demand_intel.models.deep_learning.data import ENC, N_KNOWN_CONT, SequenceArrays
from demand_intel.models.deep_learning.lstm import QuantileHead


class GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lin(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(
        self, d_in: int, d_hidden: int, d_out: int, d_context: int = 0, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.gate = GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.ctx is not None and context is not None:
            h = h + self.ctx(context)
        h = self.fc2(nn.functional.elu(h))
        return self.norm(self.skip(x) + self.gate(self.drop(h)))


class VSN(nn.Module):
    """Variable selection over V inputs already projected to d_model each."""

    def __init__(self, n_vars: int, d: int, d_context: int = 0, dropout: float = 0.1) -> None:
        super().__init__()
        self.weights = GRN(n_vars * d, d, n_vars, d_context, dropout)
        self.var_grns = nn.ModuleList(GRN(d, d, d, dropout=dropout) for _ in range(n_vars))

    def forward(
        self, xs: list[torch.Tensor], context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.cat(xs, dim=-1)
        w = torch.softmax(self.weights(flat, context), dim=-1)  # (..., V)
        processed = torch.stack([g(x) for g, x in zip(self.var_grns, xs, strict=True)], dim=-1)
        return (processed * w.unsqueeze(-2)).sum(dim=-1), w


class InterpretableAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads, self.dk = heads, d // heads
        self.q = nn.ModuleList(nn.Linear(d, self.dk) for _ in range(heads))
        self.k = nn.ModuleList(nn.Linear(d, self.dk) for _ in range(heads))
        self.v = nn.Linear(d, self.dk)  # shared across heads (interpretability)
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.v(x)
        outs, attns = [], []
        for qh, kh in zip(self.q, self.k, strict=True):
            scores = qh(x) @ kh(x).transpose(1, 2) / self.dk**0.5
            a = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
            attns.append(a)
            outs.append(self.drop(a) @ v)
        return self.out(torch.stack(outs).mean(0)), torch.stack(attns).mean(0)


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self, arr: SequenceArrays, d: int = 32, heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.static_emb = nn.ModuleList(nn.Embedding(c, d) for c in arr.static_cardinality)
        self.wday, self.month, self.event = (
            nn.Embedding(7, d),
            nn.Embedding(12, d),
            nn.Embedding(arr.n_events, d),
        )
        self.known_cont = nn.ModuleList(nn.Linear(1, d) for _ in range(N_KNOWN_CONT))
        self.past_cont = nn.ModuleList(nn.Linear(1, d) for _ in range(2))
        n_static, n_known = len(arr.static_cardinality), 3 + N_KNOWN_CONT
        self.static_vsn = VSN(n_static, d, dropout=dropout)
        self.ctx_select, self.ctx_h, self.ctx_c, self.ctx_enrich = (
            GRN(d, d, d, dropout=dropout) for _ in range(4)
        )
        self.enc_vsn = VSN(2 + n_known, d, d, dropout)
        self.dec_vsn = VSN(n_known, d, d, dropout)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = GLU(d, d)
        self.post_lstm_norm = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, d, dropout)
        self.attn = InterpretableAttention(d, heads, dropout)
        self.post_attn = GLU(d, d)
        self.post_attn_norm = nn.LayerNorm(d)
        self.pos_ff = GRN(d, d, d, dropout=dropout)
        self.final = GLU(d, d)
        self.final_norm = nn.LayerNorm(d)
        self.head = QuantileHead(d)
        self.last_weights: dict[str, torch.Tensor] = {}

    def _known(self, b: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        k = b["known"]
        return [self.wday(b["wday"]), self.month(b["month"]), self.event(b["event"])] + [
            lin(k[..., j : j + 1]) for j, lin in enumerate(self.known_cont)
        ]

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        static_vars = [emb(b["static"][:, j]) for j, emb in enumerate(self.static_emb)]
        s, w_static = self.static_vsn(static_vars)
        c_sel, c_h, c_c, c_enr = (
            self.ctx_select(s),
            self.ctx_h(s),
            self.ctx_c(s),
            self.ctx_enrich(s),
        )

        known = self._known(b)
        past = [lin(b["past"][..., j : j + 1]) for j, lin in enumerate(self.past_cont)]
        enc_vars = past + [x[:, :ENC] for x in known]
        dec_vars = [x[:, ENC:] for x in known]
        t_enc, t_dec = enc_vars[0].shape[1], dec_vars[0].shape[1]
        x_enc, w_enc = self.enc_vsn(enc_vars, c_sel[:, None].expand(-1, t_enc, -1))
        x_dec, w_dec = self.dec_vsn(dec_vars, c_sel[:, None].expand(-1, t_dec, -1))

        state = (c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous())
        h_enc, state = self.encoder(x_enc, state)
        h_dec, _ = self.decoder(x_dec, state)
        x = torch.cat([x_enc, x_dec], dim=1)
        h = self.post_lstm_norm(x + self.post_lstm(torch.cat([h_enc, h_dec], dim=1)))

        enriched = self.enrich(h, c_enr[:, None].expand(-1, h.shape[1], -1))
        n = h.shape[1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=h.device), diagonal=1)
        a, attn = self.attn(enriched, mask)
        a = self.post_attn_norm(enriched + self.post_attn(a))
        out = self.final_norm(h + self.final(self.pos_ff(a)))[:, ENC:]
        self.last_weights = {
            "static": w_static.detach(),
            "encoder": w_enc.detach().mean(1),
            "decoder": w_dec.detach().mean(1),
            "attention": attn.detach()[:, ENC:].mean(1),
        }
        return self.head(out)
