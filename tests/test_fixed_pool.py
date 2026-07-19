"""Unit tests for the fixed-stride pooling chunker (src/dcasr/models/fixed_pool.py).

Pure-PyTorch (no Mamba/CUDA) → runs on CPU. Verifies the H2-control properties: identity
at N=1 (== DynamicChunker N=1), exact masked mean pooling to rate 1/N, the ChunkOutput
contract, fixed-boundary/membership invariants, differentiability, masking, bf16 stability,
and that a non-integer stride (Type B at non-square N) raises.

Run:  pytest -q tests/test_fixed_pool.py
"""
import pytest
import torch

from dcasr.models.fixed_pool import FixedPoolChunker
from dcasr.models.hnet_chunk import ChunkOutput, DynamicChunker

torch.manual_seed(0)
B, L, D = 4, 40, 8


def _ref_masked_pool(x, mask, stride):
    """Ground-truth per-row/per-window masked mean + z_mask + membership + kept_fraction."""
    B, L, Dd = x.shape
    lengths = mask.sum(1).tolist() if mask is not None else [L] * B
    nwin = [max(1, (n + stride - 1) // stride) for n in lengths]
    M = max(nwin)
    z = torch.zeros(B, M, Dd, dtype=x.dtype)
    zm = torch.zeros(B, M, dtype=torch.bool)
    for i in range(B):
        n = lengths[i]
        for w in range(nwin[i]):
            lo, hi = w * stride, min((w + 1) * stride, n)
            if hi > lo:
                z[i, w] = x[i, lo:hi].float().mean(0).to(x.dtype)
                zm[i, w] = True
    kept = sum(nwin) / max(1, sum(lengths))
    return z, zm, nwin, kept


def _rand_mask(B, L, stride):
    """Right-padded lengths incl. a non-stride-divisible tail, a full row, and a short row."""
    lengths = torch.tensor([L, L - 1, L - stride - 1, max(1, stride - 1)][:B])
    m = torch.arange(L)[None, :] < lengths[:, None]
    return m, lengths


# ── construction / stride guard ──────────────────────────────────────────────
def test_integer_strides_ok():
    for n in (1, 2, 3, 4, 2.0, 9):
        ch = FixedPoolChunker(D, n)
        assert ch.stride == int(round(float(n)))


def test_non_integer_stride_raises():
    import math
    with pytest.raises(ValueError):
        FixedPoolChunker(D, math.sqrt(2))          # Type B at non-square N (√2)
    with pytest.raises(ValueError):
        FixedPoolChunker(D, math.sqrt(3))
    with pytest.raises(ValueError):
        FixedPoolChunker(D, 0)                      # stride < 1


# ── N=1 identity (must coincide with DynamicChunker N=1) ─────────────────────
def test_N1_is_exact_identity():
    ch = FixedPoolChunker(D, 1)
    x = torch.randn(B, L, D)
    co = ch.chunk(x)
    assert torch.equal(co.z, x)
    assert float(co.ratio_loss) == 0.0 and float(co.kept_fraction) == 1.0
    assert torch.equal(ch.dechunk(co.z, co), x)


def test_N1_gradient_is_identity():
    ch = FixedPoolChunker(D, 1)
    x = torch.randn(B, L, D, requires_grad=True)
    co = ch.chunk(x)
    ch.dechunk(co.z, co).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_N1_matches_dynamic_N1_fieldwise():
    """At N=1 fixed pooling and dynamic chunking are the same identity control (pure Mamba)."""
    x = torch.randn(B, L, D)
    mask = _rand_mask(B, L, 2)[0]
    fco = FixedPoolChunker(D, 1).chunk(x, mask)
    dco = DynamicChunker(D, 1).chunk(x, mask)
    assert torch.equal(fco.z, dco.z)
    assert torch.equal(fco.z_mask, dco.z_mask)
    assert torch.equal(fco.membership, dco.membership)
    assert torch.equal(fco.p, dco.p) and torch.equal(fco.b, dco.b)
    assert float(fco.kept_fraction) == float(dco.kept_fraction)


# ── masked mean pooling correctness vs an independent reference ──────────────
@pytest.mark.parametrize("stride", [2, 3, 4])
def test_masked_mean_matches_reference(stride):
    x = torch.randn(B, L, D)
    mask, _ = _rand_mask(B, L, stride)
    co = FixedPoolChunker(D, stride).chunk(x, mask)
    z_ref, zm_ref, nwin, kept = _ref_masked_pool(x, mask, stride)
    assert co.z.shape == z_ref.shape
    assert torch.equal(co.z_mask, zm_ref)
    # only compare real windows (padded slots are 0 in both by construction)
    assert torch.allclose(co.z[co.z_mask], z_ref[zm_ref], atol=1e-5)
    assert torch.allclose(co.z[~co.z_mask], torch.zeros_like(co.z[~co.z_mask]))
    assert abs(float(co.kept_fraction) - kept) < 1e-6


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_no_mask_full_pooling(stride):
    x = torch.randn(B, L, D)
    co = FixedPoolChunker(D, stride).chunk(x)                    # mask=None → all valid
    z_ref, zm_ref, nwin, _ = _ref_masked_pool(x, None, stride)
    import math
    assert co.z.shape[1] == math.ceil(L / stride)
    assert bool(co.z_mask.all())                                 # every window real
    assert torch.allclose(co.z, z_ref, atol=1e-5)


# ── ChunkOutput contract + fixed-boundary / membership invariants ────────────
def test_contract_fields_match_dynamic():
    x = torch.randn(B, L, D)
    fco = FixedPoolChunker(D, 2).chunk(x)
    dco = DynamicChunker(D, 2).chunk(x)
    assert isinstance(fco, ChunkOutput)
    for f in ("z", "z_mask", "p", "b", "membership"):           # [B, ...] tensors
        a, b = getattr(fco, f), getattr(dco, f)
        assert a.shape[0] == b.shape[0] == B and a.dim() == b.dim()
        assert a.dtype == b.dtype, f"{f}: {a.dtype} != {b.dtype}"
    for f in ("ratio_loss", "kept_fraction"):                   # 0-dim scalars
        a, b = getattr(fco, f), getattr(dco, f)
        assert a.dim() == b.dim() == 0 and a.dtype == b.dtype


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_boundary_and_membership_invariants(stride):
    x = torch.randn(B, L, D)
    mask, _ = _rand_mask(B, L, stride)
    co = FixedPoolChunker(D, stride).chunk(x, mask)
    pos = torch.arange(L)
    exp_memb = (pos // stride)[None, :].expand(B, L)
    assert torch.equal(co.membership, exp_memb)
    # boundaries land on fixed window starts within the valid region
    exp_b = ((pos % stride == 0).float()[None, :] * mask.float())
    assert torch.equal(co.b.float(), exp_b)
    assert torch.equal(co.p, co.b)
    # #boundaries per row == #real chunks per row (drop-in with the H-Net invariant)
    assert torch.equal(co.b.float().sum(1).long(), co.z_mask.sum(1))


def test_ratio_loss_is_zero():
    co = FixedPoolChunker(D, 3).chunk(torch.randn(B, L, D))
    assert float(co.ratio_loss) == 0.0


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_kept_fraction_near_inverse_stride(stride):
    co = FixedPoolChunker(D, stride).chunk(torch.randn(2, 600, D))
    assert abs(float(co.kept_fraction) - 1.0 / stride) < 0.02


# ── compression + round-trip shapes ──────────────────────────────────────────
@pytest.mark.parametrize("stride", [2, 3, 4])
def test_compression_and_roundtrip_shapes(stride):
    ch = FixedPoolChunker(D, stride)
    x = torch.randn(B, L, D)
    co = ch.chunk(x)
    assert co.z.shape[0] == B and co.z.shape[2] == D and co.z.shape[1] <= L
    assert ch.dechunk(co.z, co).shape == (B, L, D)


def test_dechunk_broadcasts_window_vector():
    """dechunk must copy each processed window vector to every fine frame in that window."""
    stride = 3
    ch = FixedPoolChunker(D, stride)
    x = torch.randn(B, L, D)
    co = ch.chunk(x)
    M = co.z.shape[1]
    z_proc = torch.arange(M, dtype=torch.float32)[None, :, None].expand(B, M, D).contiguous()
    up = ch.dechunk(z_proc, co)                                  # [B,L,D]
    exp = (torch.arange(L) // stride).clamp(max=M - 1).float()[None, :, None].expand(B, L, D)
    assert torch.equal(up, exp)


# ── differentiability (mean-pool → main → broadcast) ─────────────────────────
def test_full_block_is_differentiable():
    ch = FixedPoolChunker(D, 2)
    main = torch.nn.Linear(D, D)
    x = torch.randn(B, L, D, requires_grad=True)
    co = ch.chunk(x)
    y = ch.dechunk(main(co.z), co)
    y.pow(2).mean().backward()
    assert x.grad is not None and torch.any(x.grad != 0)
    assert torch.isfinite(x.grad).all()


def test_meanpool_broadcast_gradcheck_fp64():
    """Composite pool→broadcast is exactly differentiable (autograd gradcheck)."""
    stride = 2
    ch = FixedPoolChunker(3, stride)

    def fn(x):
        co = ch.chunk(x)
        return ch.dechunk(co.z, co)

    x = torch.randn(2, 6, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x,), atol=1e-6)


def test_grad_of_meanpool_distributes_uniformly():
    """d(sum z)/dx = 1/count for each frame in a full window; a single-frame tail → 1."""
    stride = 2
    ch = FixedPoolChunker(4, stride)
    x = torch.randn(1, 5, 4, requires_grad=True)                 # windows: (0,1)(2,3)(4,)
    ch.chunk(x).z.sum().backward()
    exp = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])[None, :, None].expand(1, 5, 4)
    assert torch.allclose(x.grad, exp, atol=1e-6)


# ── masking + edge cases ─────────────────────────────────────────────────────
def test_masking_ignores_padding():
    ch = FixedPoolChunker(D, 2)
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, L // 2:] = False
    co = ch.chunk(x, mask)
    # perturbing the padded region must not change any real (masked) chunk vector
    x2 = x.clone()
    x2[:, L // 2:] = torch.randn(B, L - L // 2, D)
    co2 = ch.chunk(x2, mask)
    assert torch.allclose(co.z[co.z_mask], co2.z[co2.z_mask], atol=1e-6)
    assert float((co.b * (~mask).float()).sum()) == 0.0          # no boundaries in padding


def test_short_and_single_frame_windows():
    ch = FixedPoolChunker(D, 4)
    x = torch.randn(1, 3, D)                                     # L < stride → one window
    co = ch.chunk(x)
    assert co.z.shape[1] == 1 and bool(co.z_mask.all())
    assert torch.allclose(co.z[0, 0], x[0].mean(0), atol=1e-5)
    x1 = torch.randn(1, 1, D)                                    # single frame
    co1 = FixedPoolChunker(D, 4).chunk(x1)
    assert co1.z.shape[1] == 1 and torch.allclose(co1.z[0, 0], x1[0, 0], atol=1e-5)


def test_bf16_pooling_matches_fp32():
    stride = 2
    x = torch.randn(2, 400, D)
    co32 = FixedPoolChunker(D, stride).chunk(x)
    co16 = FixedPoolChunker(D, stride).chunk(x.to(torch.bfloat16))
    assert co16.z.dtype == torch.bfloat16
    assert torch.isfinite(co16.z).all()
    assert torch.allclose(co16.z.float(), co32.z, atol=3e-2)
    assert torch.equal(co16.z_mask, co32.z_mask)
