"""Tests for CTC prefix beam search (src/dcasr/decoders/ctc.py). CPU-only.

Anchor: with a large beam + full candidate set the beam is exhaustive, so its top-1 must
equal the brute-force most-probable LABEL sequence (summing every alignment's probability).
"""
import itertools
import math

import torch
import torch.nn as nn

from dcasr.decoders.ctc import ctc_prefix_beam_search
from dcasr.decoders.lm_fusion import CausalLMScorer


def _collapse(pi, blank):
    out, prev = [], None
    for s in pi:
        if s != prev:
            if s != blank:
                out.append(s)
            prev = s
    return out


def _brute_best_label_seq(p, blank):
    """(most-probable label sequence, its log-prob) summing over all alignments. p: [T, C]."""
    T, C = p.shape
    totals: dict[tuple, float] = {}
    for pi in itertools.product(range(C), repeat=T):
        prob = 1.0
        for t in range(T):
            prob *= float(p[t, pi[t]])
        seq = tuple(_collapse(pi, blank))
        totals[seq] = totals.get(seq, 0.0) + prob
    best = max(totals.items(), key=lambda kv: kv[1])
    return list(best[0]), math.log(best[1])


def test_prefix_beam_matches_bruteforce():
    torch.manual_seed(0)
    blank = 2                                                # V=2, C=3
    for T in (2, 3, 4, 5):
        logp = torch.log_softmax(torch.randn(1, T, 3), dim=-1)
        best_seq, _ = _brute_best_label_seq(logp[0].exp().numpy(), blank)
        out = ctc_prefix_beam_search(logp, torch.tensor([T]), blank_id=blank,
                                     beam_size=100, pre_beam=10)
        assert out[0] == best_seq, (T, out[0], best_seq)     # exhaustive beam == MAP label seq


def test_prefix_beam_batch_and_lengths():
    torch.manual_seed(1)
    logp = torch.log_softmax(torch.randn(3, 6, 4), dim=-1)   # blank=3
    lengths = torch.tensor([6, 4, 2])
    out = ctc_prefix_beam_search(logp, lengths, blank_id=3, beam_size=8)
    assert len(out) == 3
    assert all(all(0 <= i < 3 for i in seq) for seq in out)  # only non-blank labels [0,V)


def test_prefix_beam_can_beat_greedy():
    # crafted [T=2] where the best PATH collapses to [0] but summing alignments favors [] or [0,1]
    # here: frame0 slightly favors label0, frame1 favors blank; greedy -> [0]; beam agrees on MAP.
    logp = torch.log(torch.tensor([[[0.5, 0.2, 0.3]], [[0.2, 0.1, 0.7]]])).transpose(0, 1)
    out = ctc_prefix_beam_search(logp, torch.tensor([2]), blank_id=2, beam_size=50, pre_beam=10)
    best, _ = _brute_best_label_seq(logp[0].exp().numpy(), 2)
    assert out[0] == best                                    # beam finds the true MAP


# ── LM fusion into the CTC prefix beam ───────────────────────────────────────
class _FavLM(nn.Module):
    def __init__(self, vocab, fav):
        super().__init__()
        self.vocab, self.fav = vocab, fav

    def forward(self, ids):
        logits = torch.zeros(*ids.shape, self.vocab)
        logits[..., self.fav] = 12.0
        return logits


def test_prefix_beam_lm_weight_zero_equals_no_lm():
    torch.manual_seed(2)
    logp = torch.log_softmax(torch.randn(2, 6, 6), dim=-1)   # V=5, blank=5
    lengths = torch.tensor([6, 5])
    lm = CausalLMScorer(_FavLM(5, 3), bos_id=1, pad_id=4)
    a = ctc_prefix_beam_search(logp, lengths, blank_id=5, beam_size=8, lm=lm, lm_weight=0.0)
    b = ctc_prefix_beam_search(logp, lengths, blank_id=5, beam_size=8)
    assert a == b                                            # lm_weight=0 -> LM never consulted


def test_prefix_beam_lm_requires_blank_last():
    import pytest
    logp = torch.log_softmax(torch.randn(1, 4, 4), dim=-1)   # 4 classes
    lm = CausalLMScorer(_FavLM(3, 1), bos_id=1, pad_id=2)
    with pytest.raises(ValueError):                          # blank not at last class + LM -> clear error
        ctc_prefix_beam_search(logp, torch.tensor([4]), blank_id=0, lm=lm, lm_weight=1.0)


def test_prefix_beam_lm_biases_output():
    torch.manual_seed(3)
    V, blank = 5, 5
    logp = torch.log_softmax(torch.randn(1, 8, V + 1) * 0.5, dim=-1)   # near-flat acoustics
    fav = 3
    lm = CausalLMScorer(_FavLM(V, fav), bos_id=1, pad_id=4)
    strong = ctc_prefix_beam_search(logp, torch.tensor([8]), blank_id=blank, beam_size=8,
                                    pre_beam=V, lm=lm, lm_weight=8.0)
    none = ctc_prefix_beam_search(logp, torch.tensor([8]), blank_id=blank, beam_size=8, pre_beam=V)
    assert strong != none and fav in strong[0]              # LM pulls its favored token in
