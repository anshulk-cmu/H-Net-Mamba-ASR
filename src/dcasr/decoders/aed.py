"""AED (attention) decoder head for DC-ASR: autoregressive Transformer decoder,
label-smoothed cross-entropy, greedy decode (plan §2.3 / §4.4).

The complement to the CTC head: instead of scoring frames independently, it emits the
transcript one token at a time, cross-attending to the encoder's fine-rate output
(`memory` [B, L0, d_model]) and self-attending causally over the tokens produced so far,
so it can learn language-like fluency. It is a *trained* head — one hybrid model carries
both CTC and AED heads (HybridLoss weights their two scalar losses), and "joint CTC+AED"
is a decode-time score combination, not a third head.

Targets arrive as BARE token ids in [0, V) (same batch format the CTC head consumes); the
head wraps them internally: decoder input = [bos, y_1..y_n], target = [y_1..y_n, eos]
(tokenizer contract bos=1/eos=2/pad=3). Padding is right-side, so with a causal self-attn
mask real positions never attend to pad, and pad target positions are ignored in the loss
— no decoder key-padding mask needed (avoids fully-masked-row NaNs). Label-synchronous beam
search and +LM fusion come with the decode stage (decode.py, lm_fusion.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def _pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """[B, max_len] bool, True at PADDED positions (key_padding_mask convention)."""
    return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]


def _causal_mask(size: int, device) -> torch.Tensor:
    """[size, size] additive mask, -inf above the diagonal (position t sees only <= t)."""
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal PE"
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))            # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x [B, T, d_model]
        return self.dropout(x + self.pe[:, : x.size(1)])


class AEDHead(nn.Module):
    """Autoregressive Transformer-decoder head over the tokenizer's V-token vocabulary.

    forward -> per-step logits [B, T, V]; loss -> label-smoothed CE scalar; greedy_decode
    -> one bare-id list per utterance (bos stripped, cut at eos). `d_memory` projects the
    encoder output to `d_model` when they differ (default: assume memory is already d_model).
    """

    def __init__(self, vocab_size: int, d_model: int, *, n_layers: int = 6, n_heads: int = 4,
                 d_ff: int = 2048, dropout: float = 0.1, lsm_weight: float = 0.1,
                 bos_id: int = 1, eos_id: int = 2, pad_id: int = 3,
                 d_memory: int | None = None, max_decode_len: int = 512):
        # 512 > the longest LibriSpeech reference (230 tokens @ spm_bpe_500):
        # a 200 cap provably truncated 3 real dev/test utterances
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.lsm_weight = float(lsm_weight)
        self.max_decode_len = int(max_decode_len)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = _SinusoidalPositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.mem_proj = (nn.Linear(d_memory, d_model)
                         if d_memory is not None and d_memory != d_model else nn.Identity())
        logger.debug("AEDHead(vocab=%d, d_model=%d, layers=%d, heads=%d, d_ff=%d, lsm=%.3g)",
                     vocab_size, d_model, n_layers, n_heads, d_ff, lsm_weight)

    # ---- target prep --------------------------------------------------------
    def _add_sos_eos(self, ys: torch.Tensor, ys_lens: torch.Tensor):
        """bare [B,U] (pad-padded) -> (ys_in=[bos,y..], ys_out=[y..,eos], ys_in_lens=len+1)."""
        B, U = ys.shape
        ar = torch.arange(B, device=ys.device)
        ys_in = ys.new_full((B, U + 1), self.pad_id)
        ys_in[:, 0] = self.bos_id
        ys_in[:, 1:] = ys                                      # right-padded -> pad stays pad
        ys_out = ys.new_full((B, U + 1), self.pad_id)
        ys_out[:, :U] = ys
        ys_out[ar, ys_lens] = self.eos_id                     # eos at each true length
        return ys_in, ys_out, ys_lens + 1

    # ---- training logits ----------------------------------------------------
    def forward(self, memory: torch.Tensor, memory_lengths: torch.Tensor,
                ys_in: torch.Tensor) -> torch.Tensor:
        """memory [B,S,d_mem], ys_in [B,T] -> logits [B, T, vocab_size]."""
        mem = self.mem_proj(memory)
        tgt = self.pos(self.embed(ys_in) * math.sqrt(self.d_model))
        dec = self.decoder(tgt, mem,
                           tgt_mask=_causal_mask(ys_in.size(1), ys_in.device),
                           memory_key_padding_mask=_pad_mask(memory_lengths, mem.size(1)))
        return self.out(dec)

    def _label_smoothing_loss(self, logits: torch.Tensor, target: torch.Tensor,
                              target_lengths: torch.Tensor) -> torch.Tensor:
        """Label-smoothed CE over [B,T,V] vs [B,T] target; positions >= length ignored."""
        V = logits.size(-1)
        logp = F.log_softmax(logits.float(), dim=-1)          # fp32 for stability
        with torch.no_grad():
            true = torch.full_like(logp, self.lsm_weight / (V - 1))
            true.scatter_(2, target.unsqueeze(2), 1.0 - self.lsm_weight)
        mask = _pad_mask(target_lengths, target.size(1)).logical_not()   # True = counted
        nll = -(true * logp).sum(-1)                          # [B, T] soft-target CE
        return (nll * mask).sum() / mask.sum().clamp(min=1)   # per-token mean

    def loss(self, memory: torch.Tensor, memory_lengths: torch.Tensor,
             targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """AED loss. memory [B,S,d]; targets [B,U] BARE ids; -> scalar (per-token mean CE)."""
        ys_in, ys_out, ys_out_lens = self._add_sos_eos(targets, target_lengths)
        logits = self.forward(memory, memory_lengths, ys_in)
        return self._label_smoothing_loss(logits, ys_out, ys_out_lens)

    # ---- decode -------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, memory: torch.Tensor, memory_lengths: torch.Tensor,
                      max_len: int | None = None) -> list[list[int]]:
        """Autoregressive greedy decode. Returns one bare-id list per utterance
        (bos stripped, truncated at the first eos). Run in eval() to disable dropout."""
        B, dev = memory.size(0), memory.device
        mem = self.mem_proj(memory)
        mem_pad = _pad_mask(memory_lengths, mem.size(1))
        cap = self.max_decode_len if max_len is None else int(max_len)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(cap):
            tgt = self.pos(self.embed(ys) * math.sqrt(self.d_model))
            dec = self.decoder(tgt, mem, tgt_mask=_causal_mask(ys.size(1), dev),
                               memory_key_padding_mask=mem_pad)
            step = self.out(dec[:, -1])                       # [B, V]
            step[:, self.bos_id] = float("-inf")             # bos/pad are non-emittable
            step[:, self.pad_id] = float("-inf")
            nxt = step.argmax(-1)
            nxt = nxt.masked_fill(finished, self.eos_id)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            finished |= nxt == self.eos_id
            if bool(finished.all()):
                break
        out: list[list[int]] = []
        for i in range(B):
            seq = ys[i, 1:].tolist()                          # drop bos
            if self.eos_id in seq:
                seq = seq[: seq.index(self.eos_id)]
            out.append(seq)
        return out
