"""Unit tests for internal-LM (ILM) estimation and density-ratio fusion. CPU-only.

The ILM estimate is checked against an INDEPENDENT reimplementation of the decoder stack
built directly from the layer submodules (it never touches `no_acoustic_context`, so the
two share no code path), and against exact algebraic identities in the beam:
subtracting a scorer with the same weight that adds it must cancel to machine precision.
"""
import math

import torch

from dcasr.decoders.aed import AEDHead, no_acoustic_context
from dcasr.decoders.joint import joint_beam_search
from dcasr.decoders.lm_fusion import ILMScorer
from dcasr.tasks.decode_task import ilm_weight_for

V, D = 12, 16


def _aed(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    return AEDHead(V, D, bos_id=1, eos_id=2, pad_id=3, **kw).eval()


def _manual_ilm_logits(head, ys_in):
    """Independent oracle: the decoder stack with NO cross-attention, rebuilt by hand.

    Mirrors the intended zero-out ILM (embed -> pos -> per layer: pre-LN self-attention
    residual then pre-LN FFN residual, cross-attention residual omitted -> final norm ->
    output projection) without invoking no_acoustic_context() or the disable flag.
    """
    x = head.pos(head.embed(ys_in) * math.sqrt(head.d_model))
    T = ys_in.size(1)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    for layer in head.decoder.layers:
        h = layer.norm1(x)
        x = x + layer.drop(layer.self_attn(h, h, h, attn_mask=mask))
        h = layer.norm3(x)
        x = x + layer.drop(layer.linear2(layer.drop(layer.act(layer.linear1(h)))))
    return head.out(head.decoder.norm(x))


class _FixedScorer:
    """Deterministic prefix-dependent next-token scorer (stands in for any LM)."""

    def __init__(self, seed=0):
        self.seed = seed

    def next_logprobs(self, prefixes, device):
        rows = []
        for p in prefixes:
            g = torch.Generator().manual_seed(self.seed + 7 * sum(p) + 131 * len(p))
            rows.append(torch.log_softmax(torch.randn(V, generator=g), dim=-1))
        return torch.stack(rows).to(device)


# ── the ILM estimate itself ──────────────────────────────────────────────────
def test_ilm_scorer_matches_independent_reimplementation():
    """ILMScorer == hand-rebuilt cross-attention-free decoder, to machine precision."""
    head = _aed()
    prefixes = [[4, 5, 6], [7], [], [8, 9]]
    got = ILMScorer(head, bos_id=1, pad_id=3).next_logprobs(prefixes, torch.device("cpu"))

    for i, p in enumerate(prefixes):
        ys = torch.tensor([[1] + p])                        # bos + prefix
        ref = torch.log_softmax(_manual_ilm_logits(head, ys)[0, -1].float(), dim=-1)
        assert torch.allclose(got[i], ref, atol=1e-6), (p, (got[i] - ref).abs().max())


def test_ilm_is_invariant_to_the_acoustic_input():
    """Severance check: inside the block, wildly different memory gives identical logits."""
    head = _aed()
    ys = torch.tensor([[1, 4, 5]])
    ml = torch.tensor([6])
    with no_acoustic_context(head):
        a = head(torch.zeros(1, 6, D), ml, ys)
        b = head(torch.randn(1, 6, D) * 100.0, ml, ys)
    assert torch.equal(a, b), (a - b).abs().max()


def test_cross_attention_is_restored_after_the_block_and_matters_inside_it():
    head = _aed()
    ys = torch.tensor([[1, 4, 5]])
    mem, ml = torch.randn(1, 6, D), torch.tensor([6])
    before = head(mem, ml, ys)
    with no_acoustic_context(head):
        inside = head(mem, ml, ys)
    after = head(mem, ml, ys)
    assert torch.equal(before, after)                        # state fully restored
    assert not torch.allclose(before, inside)                # and it really was disabled
    assert all(not l.disable_cross_attn for l in head.decoder.layers)


def test_context_manager_restores_state_even_on_exception():
    head = _aed()
    try:
        with no_acoustic_context(head):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert all(not l.disable_cross_attn for l in head.decoder.layers)


# ── density-ratio algebra in the beam ────────────────────────────────────────
def test_adding_and_subtracting_the_same_scorer_cancels_exactly():
    """+lam*S - lam*S = 0 for every hypothesis, so the beam must match the no-LM beam.

    lm and ilm accumulate on identical expansions (eos included), so the cancellation is
    exact for partial AND ended hypotheses. This pins the sign and the accumulation site.
    """
    torch.manual_seed(0)
    head = _aed()
    mem, ml = torch.randn(2, 8, D), torch.full((2,), 8)
    s = _FixedScorer()
    base = joint_beam_search(None, head, mem, ml, beam_size=4, ctc_weight=0.0)
    same = joint_beam_search(None, head, mem, ml, beam_size=4, ctc_weight=0.0,
                             lm=s, lm_weight=0.7, ilm=s, ilm_weight=0.7)
    assert same == base


def test_zero_ilm_weight_reduces_to_plain_shallow_fusion():
    torch.manual_seed(0)
    head = _aed()
    mem, ml = torch.randn(2, 8, D), torch.full((2,), 8)
    s = _FixedScorer()
    ref = joint_beam_search(None, head, mem, ml, beam_size=4, ctc_weight=0.0,
                            lm=s, lm_weight=0.4)
    got = joint_beam_search(None, head, mem, ml, beam_size=4, ctc_weight=0.0,
                            lm=s, lm_weight=0.4, ilm=_FixedScorer(9), ilm_weight=0.0)
    assert got == ref


def test_ilm_subtraction_lengthens_hypotheses():
    """The whole point of the fix: subtracting the internal LM removes the per-token
    surplus cost that truncates hypotheses, so the beam stops quitting early."""
    head = _aed()
    ilm = ILMScorer(head, bos_id=1, pad_id=3)
    short = long = 0
    for seed in range(5):
        torch.manual_seed(seed)
        mem, ml = torch.randn(3, 10, D), torch.full((3,), 10)
        short += sum(len(h) for h in joint_beam_search(
            None, head, mem, ml, beam_size=4, ctc_weight=0.0))
        long += sum(len(h) for h in joint_beam_search(
            None, head, mem, ml, beam_size=4, ctc_weight=0.0, ilm=ilm, ilm_weight=3.0))
    assert long > short, (short, long)


def test_ctc_prefix_beam_is_untouched_by_ilm():
    """CTC read-outs carry no internal LM and must keep fusing without any correction."""
    from dcasr.decoders.ctc import ctc_prefix_beam_search
    import inspect
    assert "ilm" not in inspect.signature(ctc_prefix_beam_search).parameters


# ── config resolution ────────────────────────────────────────────────────────
def test_ilm_weight_applies_to_lm_cells_only():
    cfg = {"ilm_weight": 0.2}
    assert ilm_weight_for({"name": "aed_beam_lm", "lm": True}, cfg) == 0.2
    assert ilm_weight_for({"name": "aed_beam", "lm": False}, cfg) == 0.0
    assert ilm_weight_for({"name": "joint_beam", "lm": False}, cfg) == 0.0


def test_ilm_weight_defaults_to_zero_when_unset():
    assert ilm_weight_for({"name": "aed_beam_lm", "lm": True}, {}) == 0.0
