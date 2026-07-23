"""Unit tests for second-pass n-best LM rescoring (the aed_beam_lm / joint_beam_lm cells).
CPU-only. Two pieces under test:

  - CausalLMScorer.sequence_logprob: the full-sequence LM log-prob of a COMPLETE hypothesis
    (incl. terminal eos). Checked against a hand-built oracle AND against chaining next_logprobs
    step by step (the two interfaces must be the identical log-linear LM term, to machine
    precision — that is what makes rescoring the post-hoc form of shallow fusion).
  - joint.lm_rescore + joint_beam_search_nbest: the n-best is complete, sorted, and consistent
    with single-best; the re-ranking is the brute-force argmax of S_ac + lambda*logP_LM.
"""
import math

import torch

from dcasr.decoders.aed import AEDHead
from dcasr.decoders.ctc import CTCHead
from dcasr.decoders.joint import (_Hyp, joint_beam_search, joint_beam_search_nbest,
                                  lm_rescore)
from dcasr.decoders.lm_fusion import CausalLMScorer, TransformerLM

CPU = torch.device("cpu")
V, D = 12, 16


def _lm(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    # inference-only fixtures: no autograd graph (decode always runs under no_grad)
    return TransformerLM(V, D, bos_id=1, eos_id=2, pad_id=3, **kw).eval().requires_grad_(False)


def _aed(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    return AEDHead(V, D, bos_id=1, eos_id=2, pad_id=3, **kw).eval().requires_grad_(False)


# ── sequence_logprob: independent oracle ─────────────────────────────────────
@torch.no_grad()
def _manual_seq_logprob(lm, seq, bos=1, eos=2):
    """Oracle: teacher-force [bos, w..] and sum the log-softmax of targets [w.., eos]."""
    ys_in = torch.tensor([[bos] + list(seq)])
    logp = torch.log_softmax(lm(ys_in)[0].float(), dim=-1)     # [L+1, V]
    tgt = list(seq) + [eos]
    return sum(float(logp[t, tgt[t]]) for t in range(len(tgt)))


def test_sequence_logprob_matches_manual_oracle():
    lm = _lm()
    sc = CausalLMScorer(lm, bos_id=1, eos_id=2, pad_id=3)
    seqs = [[5, 7, 9], [4], [8, 6, 4, 5]]
    got = sc.sequence_logprob(seqs, CPU)
    for i, s in enumerate(seqs):
        assert abs(float(got[i]) - _manual_seq_logprob(lm, s)) < 1e-5, s


def test_sequence_logprob_empty_hyp_is_p_eos_given_bos():
    lm = _lm()
    sc = CausalLMScorer(lm, bos_id=1, eos_id=2, pad_id=3)
    got = float(sc.sequence_logprob([[]], CPU)[0])
    ref = float(torch.log_softmax(lm(torch.tensor([[1]]))[0, -1].float(), -1)[2])  # logP(eos|bos)
    assert abs(got - ref) < 1e-6


def test_sequence_logprob_equals_chained_next_logprobs():
    """The rescoring term == first-pass shallow-fusion term accumulated over the same tokens
    (including the final eos). Ties the two interfaces to machine precision."""
    lm = _lm(1)
    sc = CausalLMScorer(lm, bos_id=1, eos_id=2, pad_id=3)
    for seq in ([5, 7, 9], [4, 4, 6], []):
        chained = 0.0
        for k in range(len(seq)):
            chained += float(sc.next_logprobs([seq[:k]], CPU)[0, seq[k]])
        chained += float(sc.next_logprobs([seq], CPU)[0, 2])            # + logP(eos | full)
        assert abs(float(sc.sequence_logprob([seq], CPU)[0]) - chained) < 1e-5, seq


def test_sequence_logprob_ragged_batch_matches_per_row():
    lm = _lm(2)
    sc = CausalLMScorer(lm, bos_id=1, eos_id=2, pad_id=3)
    seqs = [[5, 7, 9, 4], [6], [8, 8]]
    batched = sc.sequence_logprob(seqs, CPU)                            # padded + masked together
    for i, s in enumerate(seqs):
        solo = sc.sequence_logprob([s], CPU)[0]
        assert abs(float(batched[i]) - float(solo)) < 1e-5, s          # padding must not leak


# ── lm_rescore: brute-force argmax ───────────────────────────────────────────
class _StubSeqLM:
    """Returns a prescribed full-sequence log-prob per hypothesis (decouples the argmax test
    from any real LM's internals)."""

    def __init__(self, table):
        self.table = table                                             # tuple(tokens) -> float

    def sequence_logprob(self, seqs, device):
        return torch.tensor([self.table[tuple(s)] for s in seqs])


def _nbest(specs):
    """specs: list of (tokens, aed, ctc) -> sorted (desc by acoustic score, ctc_weight applied
    at rescore time) list of _Hyp. score field is set to the ctc_weight=0 acoustic score here;
    lm_rescore recomputes from aed/ctc anyway."""
    return [_Hyp(tokens=list(t), aed=a, ctc=c, score=a) for (t, a, c) in specs]


def test_lm_rescore_argmax_matches_bruteforce():
    specs = [([5, 7], -3.0, -4.0), ([5, 8], -2.5, -6.0), ([9], -4.0, -3.5)]
    table = {(5, 7): -1.0, (5, 8): -5.0, (9,): -0.5}
    lm = _StubSeqLM(table)
    for ctc_w in (0.0, 0.3):
        for lam in (0.0, 0.2, 0.5, 1.0, 3.0):
            got = lm_rescore(_nbest(specs), lm, lam, ctc_weight=ctc_w, device=CPU)
            # independent oracle: argmax over S_ac + lam*logP_LM
            best, best_s = None, -1e18
            for (t, a, c) in specs:
                s = (1 - ctc_w) * a + ctc_w * c + lam * table[tuple(t)]
                if s > best_s:
                    best_s, best = s, list(t)
            assert got == best, (ctc_w, lam, got, best)


def test_lm_rescore_lambda_zero_returns_acoustic_best():
    """lambda=0 -> the LM never changes the choice: aed_beam_lm == aed_beam. Defensibility."""
    specs = [([5, 7], -1.0, -2.0), ([9], -1.5, -1.0), ([3, 4], -2.0, -0.5)]
    lm = _StubSeqLM({(5, 7): 99.0, (9,): -99.0, (3, 4): -99.0})       # would flip if consulted
    nb = _nbest(specs)
    for ctc_w in (0.0, 0.3):
        got = lm_rescore(nb, lm, 0.0, ctc_weight=ctc_w, device=CPU)
        best = max(specs, key=lambda s: (1 - ctc_w) * s[1] + ctc_w * s[2])[0]
        assert got == list(best), (ctc_w, got, best)


def test_lm_rescore_can_flip_the_winner():
    specs = [([5, 7], -1.0, -1.0), ([9], -1.2, -1.2)]                 # [5,7] wins acoustically
    lm = _StubSeqLM({(5, 7): -5.0, (9,): 0.0})                        # LM strongly prefers [9]
    assert lm_rescore(_nbest(specs), lm, 0.0, ctc_weight=0.0, device=CPU) == [5, 7]
    assert lm_rescore(_nbest(specs), lm, 10.0, ctc_weight=0.0, device=CPU) == [9]


def test_lm_rescore_empty_nbest_returns_empty():
    assert lm_rescore([], _StubSeqLM({}), 0.5, ctc_weight=0.0, device=CPU) == []


# ── joint_beam_search_nbest: shape / order / consistency ─────────────────────
def _mem(B=2, S=8):
    torch.manual_seed(3)
    return torch.randn(B, S, D), torch.full((B,), S)


def test_nbest_is_sorted_complete_and_bounded():
    aed = _aed()
    mem, ml = _mem()
    nb = joint_beam_search_nbest(None, aed, mem, ml, beam_size=5, ctc_weight=0.0, nbest=5)
    assert len(nb) == 2
    for hyps in nb:
        assert 1 <= len(hyps) <= 5
        scores = [h.score for h in hyps]
        assert scores == sorted(scores, reverse=True)               # best-first
        for h in hyps:
            assert all(i not in (1, 2, 3) for i in h.tokens)        # no bos/eos/pad leak


def test_nbest_top1_equals_single_best_wrapper():
    """joint_beam_search (the no-LM cell) must equal the n-best top-1, bit for bit."""
    for ctc_w, head in ((0.0, None), (0.3, CTCHead(D, V))):
        aed = _aed(7)
        mem, ml = _mem(2, 8)
        single = joint_beam_search(head, aed, mem, ml, beam_size=4, ctc_weight=ctc_w)
        nb = joint_beam_search_nbest(head, aed, mem, ml, beam_size=4, ctc_weight=ctc_w, nbest=4)
        assert single == [hyps[0].tokens for hyps in nb]


def test_rescore_lambda0_reproduces_no_lm_beam():
    """End-to-end: the aed_beam_lm path at lambda=0 reproduces the aed_beam cell exactly."""
    aed = _aed(5)
    lm = CausalLMScorer(_lm(9), bos_id=1, eos_id=2, pad_id=3)
    mem, ml = _mem(3, 8)
    no_lm = joint_beam_search(None, aed, mem, ml, beam_size=4, ctc_weight=0.0)
    rescored = []
    nb = joint_beam_search_nbest(None, aed, mem, ml, beam_size=4, ctc_weight=0.0, nbest=4)
    for hyps in nb:
        rescored.append(lm_rescore(hyps, lm, 0.0, ctc_weight=0.0, device=CPU))
    assert rescored == no_lm
