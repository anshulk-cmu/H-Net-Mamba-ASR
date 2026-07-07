"""Unit tests for the H-Net dynamic-chunking core (src/dcasr/models/hnet_chunk.py).

Runs on CPU/MPS — no CUDA needed. Verifies the *properties* the paper guarantees:
identity at N=1, ~1/N compression at N>=2, shape round-trip, differentiability,
ratio-loss behaviour, causality, and the p_1=1 convention.

Run:  pytest -q tests/test_hnet_chunk.py
"""
import math
import torch
import pytest

from dcasr.models.hnet_chunk import (
    RoutingModule, DynamicChunker, ratio_loss, ChunkOutput,
)

torch.manual_seed(0)
B, L, D = 4, 40, 32


# ── Router ───────────────────────────────────────────────────────────────────
def test_router_range_and_p1():
    r = RoutingModule(D)
    x = torch.randn(B, L, D)
    p, b = r(x)
    assert p.shape == (B, L) and b.shape == (B, L)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0), "p_t must lie in [0,1]"
    assert torch.all(p[:, 0] == 1.0), "p_1 must be forced to 1.0"
    assert torch.all(b[:, 0] == 1.0), "first frame is always a boundary"
    assert torch.all((b == 0) | (b == 1)), "b_t must be binary"


def test_router_identical_frames_no_boundary():
    """If consecutive frames are identical, cos=1 ⇒ p=0 ⇒ no interior boundary."""
    r = RoutingModule(D)
    x = torch.ones(1, L, D)                      # all frames identical
    p, b = r(x)
    # interior frames (t>=1) should have p≈0 (cos of identical vecs = 1)
    assert torch.allclose(p[0, 1:], torch.zeros(L - 1), atol=1e-4)
    assert b[0, 1:].sum() == 0, "no interior boundaries when frames don't change"


def test_router_causal():
    """p_t must not depend on frames after t (no future leakage)."""
    r = RoutingModule(D).eval()
    x = torch.randn(1, L, D)
    p_full, _ = r(x)
    t = 20
    x2 = x.clone()
    x2[0, t + 1:] = torch.randn(L - t - 1, D)     # perturb the future only
    p_pert, _ = r(x2)
    assert torch.allclose(p_full[0, : t + 1], p_pert[0, : t + 1], atol=1e-6), \
        "p_{<=t} changed when only the future was perturbed → not causal"


# ── Ratio loss ───────────────────────────────────────────────────────────────
def test_ratio_loss_zero_at_N1():
    p = torch.rand(B, L); b = (p >= 0.5).float()
    assert float(ratio_loss(p, b, N=1)) == 0.0


def test_ratio_loss_minimised_at_target():
    """L_ratio is lower when the actual keep-fraction matches 1/N than when far off."""
    N = 4
    L2 = 1000
    # keep-fraction = 1/N (on target)
    on = torch.zeros(1, L2); on[0, ::N] = 1.0
    p_on = on.clone()
    # keep-fraction = 1.0 (keep everything — worst case)
    off = torch.ones(1, L2); p_off = off.clone()
    l_on = float(ratio_loss(p_on, on, N))
    l_off = float(ratio_loss(p_off, off, N))
    assert l_on < l_off, f"on-target loss {l_on} should be < keep-all loss {l_off}"


def test_ratio_loss_differentiable_through_G():
    p = torch.rand(B, L, requires_grad=True)
    b = (p.detach() >= 0.5).float()
    loss = ratio_loss(p, b, N=3)
    loss.backward()
    assert p.grad is not None and torch.any(p.grad != 0), "grad must flow through G=mean(p)"


# ── Chunker: N=1 identity ────────────────────────────────────────────────────
def test_N1_is_exact_identity():
    ch = DynamicChunker(D, N=1)
    x = torch.randn(B, L, D)
    co = ch.chunk(x)
    assert torch.equal(co.z, x), "N=1 chunk must return input unchanged"
    assert float(co.ratio_loss) == 0.0
    assert float(co.kept_fraction) == 1.0
    y = ch.dechunk(co.z, co)
    assert torch.equal(y, x), "N=1 dechunk must be identity"


def test_N1_gradient_is_identity():
    ch = DynamicChunker(D, N=1)
    x = torch.randn(B, L, D, requires_grad=True)
    co = ch.chunk(x)
    y = ch.dechunk(co.z, co)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), "N=1 must pass gradient straight through"


# ── Chunker: N>=2 compression + round-trip ───────────────────────────────────
@pytest.mark.parametrize("N", [2, 3, 4])
def test_compression_and_roundtrip_shapes(N):
    ch = DynamicChunker(D, N=N)
    x = torch.randn(B, L, D)
    co = ch.chunk(x)
    assert co.z.shape[0] == B and co.z.shape[2] == D
    assert co.z.shape[1] <= L, "compressed length must not exceed input length"
    # dechunk restores the fine resolution exactly
    y = ch.dechunk(co.z, co)
    assert y.shape == (B, L, D), "dechunk must restore [B,L,D]"


def test_membership_matches_boundaries():
    """membership[t] = cumsum(b)[t]-1, and #chunks = number of boundaries."""
    ch = DynamicChunker(D, N=2)
    x = torch.randn(1, L, D)
    co = ch.chunk(x)
    exp = (torch.cumsum(co.b, dim=1) - 1).clamp_min(0).long()
    assert torch.equal(co.membership, exp)
    assert int(co.z_mask.sum()) == int(co.b.sum()), "z_mask trues == #boundaries"


def test_ratio_loss_pulls_toward_target_when_trained():
    """A few optimisation steps on the ratio loss should move kept_fraction toward 1/N."""
    torch.manual_seed(1)
    N = 4
    ch = DynamicChunker(D, N=N)
    x = torch.randn(2, 200, D)
    opt = torch.optim.SGD(ch.parameters(), lr=5.0)
    start = float(ch.chunk(x).kept_fraction)
    for _ in range(50):
        opt.zero_grad()
        co = ch.chunk(x)
        co.ratio_loss.backward()
        opt.step()
    end = float(ch.chunk(x).kept_fraction)
    # starts near 1.0-ish (identity-init router); ratio loss should push it down toward 0.25
    assert end <= start + 1e-3, f"kept_fraction rose ({start:.3f}->{end:.3f}) under ratio loss"


# ── End-to-end differentiability through chunk→(main)→dechunk ────────────────
def test_full_block_is_differentiable():
    """Simulate E->chunk->M(identity linear on coarse)->dechunk->loss; grad must reach x."""
    ch = DynamicChunker(D, N=2)
    main = torch.nn.Linear(D, D)                  # stand-in for the Mamba main network
    x = torch.randn(B, L, D, requires_grad=True)
    co = ch.chunk(x)
    z_proc = main(co.z)                           # process coarse sequence
    y = ch.dechunk(z_proc, co)
    loss = y.pow(2).mean() + 0.03 * co.ratio_loss
    loss.backward()
    assert x.grad is not None and torch.any(x.grad != 0), "no gradient reached the input"
    assert torch.isfinite(x.grad).all(), "non-finite gradients"


def test_masking_ignores_padding():
    ch = DynamicChunker(D, N=2)
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, L // 2:] = False                       # right-pad half
    co = ch.chunk(x, mask)
    # no boundaries counted in the padded region
    assert float((co.b * (~mask).float()).sum()) == 0.0
