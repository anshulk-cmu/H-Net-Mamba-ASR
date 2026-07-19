"""Unit tests for the AED attention decoder head (src/dcasr/decoders/aed.py).

CPU-only (pure PyTorch Transformer decoder). dropout defaults to 0 so forward is
deterministic; the overfit test proves the head learns and greedy decode recovers the
target end-to-end (mirrors the CTC head's overfit check).
"""
import torch
import torch.nn.functional as F

from dcasr.decoders.aed import AEDHead


def _head(V=20, d=16, **kw):
    kw.setdefault("n_layers", 2)
    kw.setdefault("n_heads", 4)
    kw.setdefault("d_ff", 32)
    kw.setdefault("dropout", 0.0)
    return AEDHead(V, d, **kw)


def _mem(B=3, S=7, d=16, lens=None):
    m = torch.randn(B, S, d)
    lens = torch.full((B,), S) if lens is None else torch.tensor(lens)
    return m, lens


def _targets(lens, V, Umax=None, pad_id=3):
    Umax = Umax or max(lens)
    t = torch.full((len(lens), Umax), pad_id, dtype=torch.long)
    for i, L in enumerate(lens):
        t[i, :L] = torch.randint(4, V, (L,))
    return t, torch.tensor(lens)


# ── target prep + shape ──────────────────────────────────────────────────────
def test_add_sos_eos():
    h = _head()
    ys = torch.tensor([[5, 6, 7], [8, 9, 3]])              # row1 len 2 (pad_id=3 at end)
    yi, yo, li = h._add_sos_eos(ys, torch.tensor([3, 2]))
    assert yi[0].tolist() == [1, 5, 6, 7] and yo[0].tolist() == [5, 6, 7, 2]   # bos..  ..eos
    assert yi[1].tolist() == [1, 8, 9, 3] and yo[1].tolist() == [8, 9, 2, 3]   # eos at len, pad after
    assert li.tolist() == [4, 3]                          # ys_in/out valid length = len+1


def test_forward_shape():
    h = _head(V=20, d=16)
    m, ml = _mem(3, 7, 16)
    assert h.forward(m, ml, torch.randint(0, 20, (3, 5))).shape == (3, 5, 20)


def test_embedding_padding_idx_zero():
    h = _head()
    assert torch.count_nonzero(h.embed.weight[h.pad_id]) == 0


# ── loss ─────────────────────────────────────────────────────────────────────
def test_loss_scalar_and_grads():
    h = _head(V=20, d=16)
    m, ml = _mem(3, 7, 16)
    m.requires_grad_(True)
    t, tl = _targets([4, 3, 5], 20)
    loss = h.loss(m, ml, t, tl)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(m.grad).all()                   # grad reaches the encoder memory
    assert torch.isfinite(h.out.weight.grad).all() and torch.isfinite(h.embed.weight.grad).all()


def test_loss_padding_invariant():
    h = _head(V=20, d=16).eval()
    m, ml = _mem(2, 7, 16)
    lens = torch.tensor([3, 3])
    base = torch.tensor([[5, 6, 7], [8, 9, 10]])
    wide = torch.full((2, 6), 3)                          # same targets, more pad columns
    wide[:, :3] = base
    assert torch.allclose(h.loss(m, ml, base, lens), h.loss(m, ml, wide, lens), atol=1e-6)


def test_lsm_zero_equals_cross_entropy():
    h = _head(V=20, d=16, lsm_weight=0.0).eval()
    m, ml = _mem(2, 7, 16)
    t, tl = _targets([3, 4], 20)
    ours = h.loss(m, ml, t, tl)
    yi, yo, yol = h._add_sos_eos(t, tl)
    logits = h.forward(m, ml, yi)
    B, T, V = logits.shape
    posmask = torch.arange(T)[None, :] < yol[:, None]
    ce = F.cross_entropy(logits.reshape(-1, V).float(), yo.reshape(-1),
                         reduction="none").reshape(B, T)
    ref = (ce * posmask).sum() / posmask.sum()
    assert torch.allclose(ours, ref, atol=1e-5)           # lsm=0 reduces to plain CE


# ── masking correctness ──────────────────────────────────────────────────────
def test_causal_self_attention():
    h = _head(V=20, d=16).eval()
    m, ml = _mem(1, 7, 16)
    ys = torch.randint(0, 20, (1, 6))
    o1 = h.forward(m, ml, ys)
    ys2 = ys.clone()
    ys2[0, 4] = (int(ys[0, 4]) + 1) % 20                  # change a LATER token
    o2 = h.forward(m, ml, ys2)
    assert torch.allclose(o1[:, :4], o2[:, :4], atol=1e-5)   # earlier positions unaffected
    assert not torch.allclose(o1[:, 4:], o2[:, 4:], atol=1e-4)   # position >=4 changes


def test_memory_padding_ignored():
    h = _head(V=20, d=16).eval()
    m = torch.randn(2, 7, 16)
    ml = torch.tensor([7, 4])                             # row 1 has 3 padded frames
    ys = torch.randint(0, 20, (2, 5))
    o1 = h.forward(m, ml, ys)
    m2 = m.clone()
    m2[1, 4:] = torch.randn(3, 16) * 100                 # perturb padded frames only
    assert torch.allclose(o1, h.forward(m2, ml, ys), atol=1e-5)


# ── decode ───────────────────────────────────────────────────────────────────
def test_greedy_decode_contract():
    h = _head(V=20, d=16, max_decode_len=10).eval()
    m, ml = _mem(3, 7, 16)
    out = h.greedy_decode(m, ml)
    assert isinstance(out, list) and len(out) == 3
    for seq in out:
        assert all(0 <= i < 20 for i in seq)              # bare ids in [0, V)
        assert h.eos_id not in seq and len(seq) <= 10     # eos stripped, capped


def test_greedy_no_special_leak():
    torch.manual_seed(0)
    h = _head(V=12, d=16).eval()
    m, ml = _mem(3, 8, 16)
    for seq in h.greedy_decode(m, ml, max_len=15):
        assert 1 not in seq and 3 not in seq                 # bos/pad never emitted


def test_train_mode_dropout_finite():
    h = AEDHead(20, 16, n_layers=2, n_heads=4, d_ff=32, dropout=0.3).train()
    m, ml = _mem(2, 7, 16)
    t, tl = _targets([3, 4], 20)
    loss = h.loss(m, ml, t, tl)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(h.out.weight.grad).all()        # dropout path runs + backprops


# ── end-to-end: overfit a single example ─────────────────────────────────────
def test_overfit_single_example():
    torch.manual_seed(0)
    V, d = 20, 16
    h = _head(V=V, d=d, dropout=0.0, lsm_weight=0.0)   # no smoothing floor -> loss can reach ~0
    m = torch.randn(1, 6, d)
    ml = torch.tensor([6])
    tgt, tl = torch.tensor([[5, 7, 9, 11]]), torch.tensor([4])
    opt = torch.optim.Adam(h.parameters(), lr=1e-2)
    loss = None
    for _ in range(300):
        opt.zero_grad()
        loss = h.loss(m, ml, tgt, tl)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1                               # memorized the example
    h.eval()
    assert h.greedy_decode(m, ml, max_len=10)[0] == [5, 7, 9, 11]   # greedy recovers it exactly
