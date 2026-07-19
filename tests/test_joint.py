"""Unit tests for joint CTC+AED beam search (src/dcasr/decoders/joint.py). CPU-only.

The CTC prefix scorer is validated against a brute-force enumerator over ALL CTC alignments
(the ground-truth definition of a prefix probability); the beam is checked by AED-greedy
equivalence at beam=1 and by recovering an overfit target.
"""
import itertools
import math

import torch

from dcasr.decoders.aed import AEDHead
from dcasr.decoders.ctc import CTCHead
from dcasr.decoders.joint import CTCPrefixScorer, joint_beam_search


# ── brute-force CTC prefix probability (ground truth) ────────────────────────
def _ctc_collapse(pi, blank):
    out, prev = [], None
    for s in pi:
        if s != prev:
            if s != blank:
                out.append(s)
            prev = s
    return out


def _brute_prefix_logprob(p, g, blank):
    """log Σ P(π) over all frame-label sequences π whose CTC collapse starts with prefix g."""
    T, K = p.shape
    total = 0.0
    for pi in itertools.product(range(K), repeat=T):
        prob = 1.0
        for t in range(T):
            prob *= float(p[t, pi[t]])
        if _ctc_collapse(pi, blank)[: len(g)] == list(g):
            total += prob
    return math.log(total) if total > 0 else -1e10


def _chain_score(scorer, g):
    """Chain the incremental scorer through prefix g; return the prefix score of full g."""
    state = scorer.initial_state()
    sc = None
    for k in range(len(g)):
        scores, states = scorer.score(list(g[:k]), torch.tensor([g[k]]), state)
        sc, state = float(scores[0]), states[0]
    return sc


def test_ctc_prefix_scorer_matches_bruteforce():
    torch.manual_seed(0)
    V, blank, eos = 3, 3, 99                                # eos out of [0,V): eos path untouched here
    for T in (3, 4, 5):
        logp = torch.log_softmax(torch.randn(T, V + 1), dim=-1)
        p = logp.exp()
        scorer = CTCPrefixScorer(logp, blank, eos)
        for g in [[0], [1], [2], [0, 1], [1, 2], [0, 0], [2, 1, 0]]:
            if len(g) >= T:
                continue                                   # need out_len < T
            got, ref = _chain_score(scorer, g), _brute_prefix_logprob(p, g, blank)
            assert abs(got - ref) < 1e-4, (T, g, got, ref)


def test_initial_state_is_all_blank_cumulative():
    logp = torch.log_softmax(torch.randn(4, 4), dim=-1)
    s = CTCPrefixScorer(logp, blank_id=3, eos_id=99).initial_state()
    assert torch.all(s[:, 0] < -1e9)                       # empty prefix can't end non-blank
    assert abs(float(s[0, 1]) - float(logp[0, 3])) < 1e-6
    assert abs(float(s[2, 1]) - float(logp[:3, 3].sum())) < 1e-6   # cumulative blank


# ── beam search ──────────────────────────────────────────────────────────────
def _aed(V=12, d=16, **kw):
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    return AEDHead(V, d, bos_id=1, eos_id=2, pad_id=3, **kw).eval()


def _mem(B=2, S=8, d=16):
    return torch.randn(B, S, d), torch.full((B,), S)


def test_aed_beam_runs_valid():
    torch.manual_seed(0)
    mem, ml = _mem()
    out = joint_beam_search(None, _aed(), mem, ml, beam_size=4, ctc_weight=0.0)
    assert len(out) == 2
    assert all(all(0 <= i < 12 for i in s) and 2 not in s for s in out)   # bare ids, no eos


def test_joint_ctc_aed_beam_runs_valid():
    torch.manual_seed(0)
    mem, ml = _mem(2, 8, 16)
    out = joint_beam_search(CTCHead(16, 12), _aed(V=12, d=16), mem, ml,
                            beam_size=4, ctc_weight=0.3)
    assert len(out) == 2 and all(all(0 <= i < 12 for i in s) and 2 not in s for s in out)


def test_beam1_ctc0_matches_aed_greedy():
    torch.manual_seed(1)
    aed = _aed(V=12, d=16, max_decode_len=8)
    mem, ml = _mem(2, 8, 16)
    beam = joint_beam_search(None, aed, mem, ml, beam_size=1, ctc_weight=0.0, max_len_ratio=1.0)
    greedy = aed.greedy_decode(mem, ml, max_len=7)          # max_steps = min(8, 8-1, 8) = 7
    assert beam == greedy


def test_no_special_tokens_leak_in_beam():
    aed = _aed(V=12, d=16)                                  # bos=1, eos=2, pad=3
    for s in range(8):
        torch.manual_seed(s)
        mem, ml = _mem(2, 8, 16)
        for seq in joint_beam_search(None, aed, mem, ml, beam_size=4, ctc_weight=0.0):
            assert 1 not in seq and 2 not in seq and 3 not in seq   # no bos/eos/pad emitted


def test_ctc_prefix_scorer_eos_outside_ctc_vocab():
    logp = torch.log_softmax(torch.randn(4, 3), dim=-1)    # V=2, blank=2, only 3 CTC classes
    sc = CTCPrefixScorer(logp, blank_id=2, eos_id=99)      # eos id outside the CTC class dim
    st = sc.initial_state()
    scores, _ = sc.score([0], torch.tensor([99]), st)      # must NOT IndexError on the gather
    assert torch.isfinite(scores).all()
    assert abs(float(scores[0]) - float(torch.logaddexp(st[:, 0], st[:, 1])[-1])) < 1e-5


def test_beam_recovers_overfit_target():
    torch.manual_seed(0)
    V, d = 12, 16
    aed = AEDHead(V, d, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, lsm_weight=0.0,
                  bos_id=1, eos_id=2, pad_id=3)
    mem = torch.randn(1, 8, d)
    ml = torch.tensor([8])
    tgt, tl = torch.tensor([[5, 7, 9, 4]]), torch.tensor([4])
    opt = torch.optim.Adam(aed.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        aed.loss(mem, ml, tgt, tl).backward()
        opt.step()
    aed.eval()
    assert joint_beam_search(None, aed, mem, ml, beam_size=4, ctc_weight=0.0)[0] == [5, 7, 9, 4]


def test_ctc_weight_one_full_prebeam_finite():
    """ctc_weight=1.0 + pre_beam=V used to make 0·(−inf)=NaN scores; specials never expand."""
    torch.manual_seed(0)
    mem, ml = _mem(2, 8, 16)
    out = joint_beam_search(CTCHead(16, 12), _aed(V=12, d=16), mem, ml,
                            beam_size=4, ctc_weight=1.0, pre_beam=12)
    assert len(out) == 2
    assert all(all(0 <= i < 12 and i not in (1, 3) for i in s) for s in out)


def test_positive_length_bonus_no_premature_stop():
    """With a dominant per-token bonus the beam must search to the cap (no unsound early stop)."""
    torch.manual_seed(0)
    mem, ml = _mem(1, 8, 16)
    aed = _aed(V=12, d=16)
    short = joint_beam_search(None, aed, mem, ml, beam_size=4, ctc_weight=0.0, length_bonus=0.0)
    long_ = joint_beam_search(None, aed, mem, ml, beam_size=4, ctc_weight=0.0, length_bonus=5.0)
    assert len(long_[0]) >= len(short[0])
    assert all(0 <= i < 12 and i not in (1, 2, 3) for i in long_[0])
