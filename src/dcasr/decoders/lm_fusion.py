"""External LM integration for the +LM columns (beam side only, plan §6.3).

The LM is trained separately on the official 810M-word LibriSpeech LM corpus, over the SAME
BPE vocabulary as the ASR decoder (so integration is just adding log-probs over identical
tokens). It enters decoding two ways, both provided by `CausalLMScorer` (a thin adapter over
ANY causal LM `forward(ids)->logits [B,T,V]`):

  - CTC prefix beam: FIRST-PASS shallow fusion via `next_logprobs` — CTC is nearly
    language-blind per frame, so the LM adds real information and fuses cleanly (the
    ctc_beam_lm cell).
  - AED / joint beam: SECOND-PASS rescoring via `sequence_logprob` — the acoustic beam runs
    LM-free, then `joint.lm_rescore` re-ranks the completed n-best by the LM's full-sequence
    log-prob. An autoregressive AED already carries a strong internal LM and is very
    low-entropy, so first-pass shallow fusion double-counts the language prior and truncates
    hypotheses; rescoring re-orders a fixed complete-hyp set and cannot (aed_beam_lm /
    joint_beam_lm). See runlog 2026-07-23.

This module provides the reference decoder-only `TransformerLM` (trained in the external-LM
step) and `CausalLMScorer`. An n-gram (ARPA/KenLM) scorer implementing the same interface is
added with the external-LM step; +LM lives on the beam side only (no "greedy +LM" cell).
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
    """Adapts a causal LM into the two decode-time LM interfaces.

    `next_logprobs(prefixes, device)` -> [n, V]: the next-token log-softmax after bos+prefix,
    for FIRST-PASS shallow fusion in the CTC prefix beam. Handles ragged prefixes (pads,
    gathers the true last position).

    `sequence_logprob(sequences, device)` -> [n]: the full-sequence log-prob of each COMPLETE
    hypothesis (incl. terminal eos), for SECOND-PASS n-best rescoring of the AED/joint beam.

    Recompute-per-step (no kv-cache) — fine for offline decode.
    """

    def __init__(self, lm, bos_id: int = 1, eos_id: int = 2, pad_id: int = 3):
        self.lm = lm
        self.bos_id = bos_id
        self.eos_id = eos_id
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

    @torch.no_grad()
    def sequence_logprob(self, sequences: list[list[int]], device) -> torch.Tensor:
        """Full-sequence LM log-prob of each COMPLETE hypothesis, for second-pass rescoring.

        For a hyp [w_1..w_L] returns Σ_i logP_LM(w_i | bos, w_<i) + logP_LM(eos | bos, w_1..w_L)
        — the LM likelihood of the whole sentence INCLUDING its terminal eos, matching the LM
        training contract (bos-wrapped input, eos at the true length). One batched teacher-forced
        pass; ragged hyps are right-padded and the pad targets masked out. Returns [n]. Summing
        `next_logprobs` step-by-step over the same tokens (with the final eos) gives the identical
        value — the two interfaces are the same log-linear LM term, applied per-step vs. post-hoc.
        """
        n = len(sequences)
        lens = [len(s) + 1 for s in sequences]             # scored positions: L tokens + eos
        maxL = max(lens)
        ys_in = torch.full((n, maxL), self.pad_id, dtype=torch.long, device=device)
        ys_out = torch.full((n, maxL), self.pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(sequences):
            ys_in[i, 0] = self.bos_id
            L = len(s)
            if L:
                st = torch.tensor(s, dtype=torch.long, device=device)
                ys_in[i, 1 : 1 + L] = st                   # [bos, w_1..w_L]
                ys_out[i, :L] = st                         # target: w_1..w_L, then eos at L
            ys_out[i, L] = self.eos_id
        logp = torch.log_softmax(self.lm(ys_in).float(), dim=-1)          # [n, maxL, V]
        tgt = logp.gather(-1, ys_out.unsqueeze(-1)).squeeze(-1)           # [n, maxL] target logp
        mask = (torch.arange(maxL, device=device)[None, :]
                < torch.tensor(lens, device=device)[:, None])            # keep positions 0..L
        return (tgt * mask).sum(-1)                                       # [n]
