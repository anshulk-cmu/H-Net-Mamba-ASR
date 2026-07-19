"""Unit tests for the DC-ASR encoder (src/dcasr/models/encoder.py).

CUDA-only (Mamba-2 kernels). Small dims for speed; n_mels=80 to match the real contract.
"""
import pytest
import torch

from dcasr.models.encoder import (
    ConvSubsampling4, DCASREncoder, _subsampled_length, build_chunker)
from dcasr.models.fixed_pool import FixedPoolChunker
from dcasr.models.hnet_chunk import DynamicChunker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-2 kernels are CUDA-only")

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


def _enc(arch="A", N=1, chunker="dynamic"):
    return DCASREncoder(n_mels=80, d_outer=64, d_main=128, n_enc=2, n_main=2,
                        n_dec=2, n_mid=2, arch_type=arch, N=N, chunker=chunker)


def _batch(T=(100, 80)):
    lengths = torch.tensor(list(T))
    return torch.randn(len(T), max(T), 80), lengths


def test_conv_subsample_length_matches_actual():
    sub = ConvSubsampling4(80, 32)
    for T in (50, 100, 137, 200):
        x, _ = sub(torch.randn(1, T, 80), torch.tensor([T]))
        assert x.shape[1] == _subsampled_length(torch.tensor([T])).item()
    assert x.shape[2] == 32                                     # projects to d_model


def test_typeA_shape_and_lengths():
    enc = _enc("A", N=1)
    feats, lengths = _batch((100, 80))
    out = enc(feats, lengths)
    exp = _subsampled_length(lengths)
    assert out.features.shape == (2, int(exp.max()), 64)
    assert torch.equal(out.lengths, exp)


def test_N1_is_passthrough():
    out = _enc("A", N=1)(*_batch())
    assert out.ratio_loss.item() == 0.0
    assert abs(out.kept_fractions[0].item() - 1.0) < 1e-6


def test_N2_compresses_but_output_is_fine_rate():
    enc = _enc("A", N=2)
    feats, lengths = _batch((120, 96))
    out = enc(feats, lengths)
    exp = _subsampled_length(lengths)
    assert out.features.shape[1] == int(exp.max())             # dechunked back to fine rate
    assert torch.isfinite(out.ratio_loss).item() and out.ratio_loss.item() > 0
    assert 0.0 < out.kept_fractions[0].item() <= 1.0


def test_interpretability_hooks_typeA():
    out = _enc("A", N=2)(*_batch())
    assert len(out.boundaries) == 1 and len(out.chunk_embeddings) == 1
    p, b = out.boundaries[0]
    assert p.shape[0] == 2 and p.dim() == 2                     # [B, L0]
    assert out.chunk_embeddings[0].dim() == 3                   # [B, M, d]


def test_gradients_flow():
    enc = _enc("A", N=2)
    feats, lengths = _batch()
    out = enc(feats, lengths)
    (out.features.sum() + out.ratio_loss).backward()
    g = [p.grad for p in enc.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert sum(x.abs().sum() for x in g) > 0


def test_typeB_two_stages():
    enc = _enc("B", N=4)
    feats, lengths = _batch((140, 110))
    out = enc(feats, lengths)
    exp = _subsampled_length(lengths)
    assert out.features.shape == (2, int(exp.max()), 64)       # fine-rate output
    assert len(out.boundaries) == 2 and len(out.kept_fractions) == 2
    assert torch.isfinite(out.ratio_loss).item() and out.ratio_loss.item() > 0


def test_typeB_N1_reduces_to_passthrough():
    out = _enc("B", N=1)(*_batch())
    assert out.ratio_loss.item() == 0.0
    assert all(abs(k.item() - 1.0) < 1e-6 for k in out.kept_fractions)


def test_invalid_arch_raises():
    with pytest.raises(ValueError):
        DCASREncoder(arch_type="C")


# ── fixed-stride pooling chunker (H2 control) ────────────────────────────────
def test_build_chunker_registry():
    assert isinstance(build_chunker("dynamic", 64, 2), DynamicChunker)
    assert isinstance(build_chunker("fixed", 64, 2), FixedPoolChunker)
    with pytest.raises(ValueError):
        build_chunker("nope", 64, 2)


def test_default_chunker_is_dynamic():
    assert isinstance(_enc("A", N=2).chunk, DynamicChunker)


def test_fixed_typeA_selected_and_compresses():
    enc = _enc("A", N=2, chunker="fixed")
    assert isinstance(enc.chunk, FixedPoolChunker)
    feats, lengths = _batch((120, 96))
    out = enc(feats, lengths)
    exp = _subsampled_length(lengths)
    assert out.features.shape[1] == int(exp.max())              # dechunked back to fine rate
    assert out.ratio_loss.item() == 0.0                          # fixed pooling has no ratio loss
    assert abs(out.kept_fractions[0].item() - 0.5) < 0.05        # rate ≈ 1/N


def test_fixed_N1_is_passthrough():
    out = _enc("A", N=1, chunker="fixed")(*_batch())
    assert out.ratio_loss.item() == 0.0
    assert abs(out.kept_fractions[0].item() - 1.0) < 1e-6


def test_fixed_interpretability_hooks():
    out = _enc("A", N=2, chunker="fixed")(*_batch())
    assert len(out.boundaries) == 1 and len(out.chunk_embeddings) == 1
    p, b = out.boundaries[0]
    assert p.shape[0] == 2 and p.dim() == 2
    assert out.chunk_embeddings[0].dim() == 3


def test_fixed_gradients_flow():
    enc = _enc("A", N=2, chunker="fixed")
    out = enc(*_batch())
    out.features.sum().backward()
    g = [p.grad for p in enc.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert sum(x.abs().sum() for x in g) > 0


def test_fixed_typeB_square_N_ok():
    enc = _enc("B", N=4, chunker="fixed")                        # √4 = 2 (integer stride)
    assert isinstance(enc.chunk1, FixedPoolChunker) and enc.chunk1.stride == 2
    feats, lengths = _batch((140, 110))
    out = enc(feats, lengths)
    exp = _subsampled_length(lengths)
    assert out.features.shape == (2, int(exp.max()), 64)
    assert len(out.boundaries) == 2 and out.ratio_loss.item() == 0.0


def test_fixed_typeB_nonsquare_N_raises():
    with pytest.raises(ValueError):                              # √2 not an integer stride
        DCASREncoder(n_mels=80, d_outer=64, d_main=128, n_enc=2, n_main=2, n_dec=2,
                     n_mid=2, arch_type="B", N=2, chunker="fixed")
