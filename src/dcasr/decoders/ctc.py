"""CTC head for DC-ASR: linear projection to V+1 classes, CTC loss, greedy decode.

The head reads the encoder's fine-rate output [B, L0, d_model] and scores each frame
over the tokenizer's V subword pieces plus one CTC blank appended at id V
(`blank_id = vocab_size`, the tokenizer contract), so it has V+1 outputs. Targets are
bare token ids in [0, V) — no bos/eos (those are AED-only). Greedy decode = argmax per
frame, collapse consecutive repeats, drop blank; its per-frame spikes (`frame_argmax`)
also feed the interpretability cross-check vs H-Net boundaries. Prefix-beam / +LM come
with the decode stage (decode.py, lm_fusion.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def ctc_greedy_collapse(frame_ids: list[int], blank_id: int) -> list[int]:
    """CTC greedy rule: collapse consecutive duplicates, then drop blanks.

    A blank between two identical labels keeps them distinct (…a ␣ a… -> a a).
    Returns token ids in [0, vocab_size) (blank removed).
    """
    out: list[int] = []
    prev = None
    for s in frame_ids:
        if s != prev:
            if s != blank_id:
                out.append(s)
            prev = s
    return out


class CTCHead(nn.Module):
    """Linear d_model -> vocab_size+1 CTC head (blank appended at id vocab_size)."""

    def __init__(self, d_model: int, vocab_size: int, blank_id: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.blank_id = vocab_size if blank_id is None else blank_id
        self.num_classes = vocab_size + 1
        self.proj = nn.Linear(d_model, self.num_classes)
        logger.debug("CTCHead(d_model=%d, vocab=%d, blank=%d, classes=%d)",
                     d_model, vocab_size, self.blank_id, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features [B, L, d_model] -> logits [B, L, vocab_size+1]."""
        return self.proj(features)

    def log_probs(self, features: torch.Tensor) -> torch.Tensor:
        """Log-softmax over classes (fp32 for numerically stable CTC), [B, L, V+1]."""
        return F.log_softmax(self.forward(features).float(), dim=-1)

    def loss(self, features: torch.Tensor, feat_lengths: torch.Tensor,
             targets: torch.Tensor, target_lengths: torch.Tensor,
             reduction: str = "mean") -> torch.Tensor:
        """CTC loss. features [B,L,d]; feat_lengths [B]; targets [B,U] (pad beyond
        target_lengths is ignored) or 1-D concatenated; target_lengths [B]. Scalar."""
        lp = self.log_probs(features).transpose(0, 1)          # [L, B, V+1] time-first
        return F.ctc_loss(lp, targets, feat_lengths, target_lengths,
                          blank=self.blank_id, reduction=reduction, zero_infinity=True)

    @torch.no_grad()
    def frame_argmax(self, features: torch.Tensor) -> torch.Tensor:
        """Per-frame top class incl. blank, [B, L] — the raw CTC spikes (interp)."""
        return self.forward(features).argmax(dim=-1)

    @torch.no_grad()
    def greedy_decode(self, features: torch.Tensor,
                      feat_lengths: torch.Tensor) -> list[list[int]]:
        """Greedy CTC decode over the valid frames of each utterance.

        Returns one token-id list per utterance (ids in [0, vocab_size), blank removed).
        """
        preds = self.frame_argmax(features)                    # [B, L]
        return [ctc_greedy_collapse(preds[i, :n].tolist(), self.blank_id)
                for i, n in enumerate(feat_lengths.tolist())]
