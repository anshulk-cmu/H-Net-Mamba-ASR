"""External LM shallow fusion for the +LM columns (beam side only, plan §6.3).

Shallow fusion adds an external language model's opinion to the beam score:
    score(h) = (1-ctc_weight)·AED + ctc_weight·CTC-prefix + lm_weight·logP_LM(h)
The LM is trained separately on the official 810M-word LibriSpeech LM corpus, over the SAME
BPE vocabulary as the ASR decoder (so fusion is just adding log-probs over identical tokens).
This module provides a reference decoder-only `TransformerLM` (trained in the external-LM
step) and `CausalLMScorer`, a thin adapter turning ANY causal LM (`forward(ids)->logits
[B,T,V]`) into the next-token scorer `joint_beam_search` consumes via `lm=`/`lm_weight=`.
An n-gram (ARPA/KenLM) scorer implementing the same `next_logprobs` interface is added with
the external-LM step; +LM lives on the beam side only (no "greedy +LM" cell).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def _causal_mask(size: int, device) -> torch.Tensor:
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerLM(nn.Module):
    """Decoder-only causal Transformer LM over the BPE vocab (trained on the LM corpus, #6).

    forward(ids [B,T]) -> logits [B,T,V] (next-token at each position). `loss` = next-token
    label-smoothed CE with bos/eos wrapping (same target contract as the AED head).
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 16, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1, lsm_weight: float = 0.1,
                 bos_id: int = 1, eos_id: int = 2, pad_id: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.lsm_weight = float(lsm_weight)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = _SinusoidalPositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, vocab_size)
        logger.debug("TransformerLM(vocab=%d, d_model=%d, layers=%d)", vocab_size, d_model, n_layers)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B,T] -> next-token logits [B,T,V] (causal self-attention)."""
        x = self.pos(self.embed(ids) * math.sqrt(self.d_model))
        h = self.blocks(x, mask=_causal_mask(ids.size(1), ids.device))
        return self.out(h)

    def loss(self, tokens: torch.Tensor, token_lengths: torch.Tensor,
             return_acc: bool = False):
        """Next-token CE. tokens [B,U] BARE ids -> scalar (per-token mean over [w.., eos]).

        return_acc=True additionally returns next-token prediction accuracy (argmax
        over the same masked positions) — the standard LM training health metric.
        """
        B, U = tokens.shape
        ar = torch.arange(B, device=tokens.device)
        ys_in = tokens.new_full((B, U + 1), self.pad_id)
        ys_in[:, 0] = self.bos_id
        ys_in[:, 1:] = tokens                              # [bos, w..]
        ys_out = tokens.new_full((B, U + 1), self.pad_id)
        ys_out[:, :U] = tokens
        ys_out[ar, token_lengths] = self.eos_id            # [w.., eos]
        logp = F.log_softmax(self.forward(ys_in).float(), dim=-1)
        V = logp.size(-1)
        with torch.no_grad():
            true = torch.full_like(logp, self.lsm_weight / (V - 1))
            true.scatter_(2, ys_out.unsqueeze(2), 1.0 - self.lsm_weight)
        mask = torch.arange(U + 1, device=tokens.device)[None, :] < (token_lengths + 1)[:, None]
        nll = -(true * logp).sum(-1)
        loss = (nll * mask).sum() / mask.sum().clamp(min=1)
        if not return_acc:
            return loss
        with torch.no_grad():
            acc = ((logp.argmax(-1) == ys_out) & mask).sum() / mask.sum().clamp(min=1)
        return loss, acc


class CausalLMScorer:
    """Adapts a causal LM into the beam's next-token scorer (shallow fusion).

    `next_logprobs(prefixes, device)` prepends bos to each bare-id prefix, runs the LM, and
    returns the last-step log-softmax [n, V]. Handles ragged prefixes (pads, gathers the true
    last position). Recompute-per-step (no kv-cache) — fine for offline decode.
    """

    def __init__(self, lm, bos_id: int = 1, pad_id: int = 3):
        self.lm = lm
        self.bos_id = bos_id
        self.pad_id = pad_id

    @torch.no_grad()
    def next_logprobs(self, prefixes: list[list[int]], device) -> torch.Tensor:
        lens = [len(p) + 1 for p in prefixes]              # +1 for bos
        maxL = max(lens)
        ys = torch.full((len(prefixes), maxL), self.pad_id, dtype=torch.long, device=device)
        for i, p in enumerate(prefixes):
            ys[i, 0] = self.bos_id
            if p:
                ys[i, 1 : 1 + len(p)] = torch.tensor(p, device=device)
        logits = self.lm(ys)                               # [n, maxL, V]
        last = logits[torch.arange(len(prefixes), device=device),
                      torch.tensor(lens, device=device) - 1]   # [n, V] (causal -> padding after is ignored)
        return torch.log_softmax(last.float(), dim=-1)
