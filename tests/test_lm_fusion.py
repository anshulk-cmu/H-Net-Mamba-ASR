"""Unit tests for the external LM and its decode-time adapter (src/dcasr/decoders/lm_fusion.py).
CPU-only. Covers the reference TransformerLM (causal + loss + overfit) and the CausalLMScorer
`next_logprobs` adapter (incl. ragged prefixes) used for CTC first-pass shallow fusion. The
second-pass `sequence_logprob` interface and n-best rescoring are covered in test_rescore.py.
"""
import torch

from dcasr.decoders.lm_fusion import CausalLMScorer, TransformerLM

CPU = torch.device("cpu")


def _lm(V=12, d=16, **kw):
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    return TransformerLM(V, d, bos_id=1, eos_id=2, pad_id=3, **kw)


# ── TransformerLM ─────────────────────────────────────────────────────────────
def test_lm_forward_shape_and_causal():
    lm = _lm().eval()
    ids = torch.randint(0, 12, (2, 6))
    o1 = lm(ids)
    assert o1.shape == (2, 6, 12)
    ids2 = ids.clone()
    ids2[:, 4] = (ids[:, 4] + 1) % 12
    assert torch.allclose(o1[:, :4], lm(ids2)[:, :4], atol=1e-5)   # causal: earlier unaffected


def test_lm_loss_grads_and_padding_idx():
    lm = _lm()
    toks = torch.randint(4, 12, (2, 5))
    toks[1, 4] = 3                                            # row1 len 4 -> pad the tail
    loss = lm.loss(toks, torch.tensor([5, 4]))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lm.out.weight.grad).all()
    assert torch.count_nonzero(lm.embed.weight.grad[3]) == 0   # padding_idx row: zero grad


def test_lm_overfit_predicts_next():
    torch.manual_seed(0)
    lm = TransformerLM(12, 16, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, lsm_weight=0.0,
                       bos_id=1, eos_id=2, pad_id=3)
    toks, tl = torch.tensor([[5, 7, 9, 4]]), torch.tensor([4])
    opt = torch.optim.Adam(lm.parameters(), lr=1e-2)
    loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = lm.loss(toks, tl)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    sc = CausalLMScorer(lm.eval(), bos_id=1, pad_id=3)
    assert int(sc.next_logprobs([[5, 7]], CPU)[0].argmax()) == 9   # learned 5,7 -> 9


# ── CausalLMScorer ────────────────────────────────────────────────────────────
def test_scorer_matches_manual_forward():
    torch.manual_seed(0)
    lm = _lm().eval()
    sc = CausalLMScorer(lm, bos_id=1, pad_id=3)
    got = sc.next_logprobs([[5, 7, 9], [4, 6, 8]], CPU)
    manual = torch.log_softmax(lm(torch.tensor([[1, 5, 7, 9], [1, 4, 6, 8]]))[:, -1].float(), -1)
    assert torch.allclose(got, manual, atol=1e-5)
    assert torch.allclose(torch.logsumexp(got, -1), torch.zeros(2), atol=1e-5)   # normalized


def test_scorer_ragged_prefixes():
    torch.manual_seed(0)
    lm = _lm().eval()
    sc = CausalLMScorer(lm, bos_id=1, pad_id=3)
    got = sc.next_logprobs([[5, 7, 9], [4]], CPU)              # different lengths -> pad + gather
    r0 = torch.log_softmax(lm(torch.tensor([[1, 5, 7, 9]]))[:, -1].float(), -1)[0]
    r1 = torch.log_softmax(lm(torch.tensor([[1, 4]]))[:, -1].float(), -1)[0]
    assert torch.allclose(got[0], r0, atol=1e-5) and torch.allclose(got[1], r1, atol=1e-5)


def test_scorer_empty_prefix():
    lm = _lm().eval()
    got = CausalLMScorer(lm, bos_id=1, pad_id=3).next_logprobs([[]], CPU)
    manual = torch.log_softmax(lm(torch.tensor([[1]]))[:, -1].float(), -1)
    assert torch.allclose(got, manual, atol=1e-5)              # empty prefix == just bos
