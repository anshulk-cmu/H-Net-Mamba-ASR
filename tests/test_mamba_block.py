"""Unit tests for the Mamba-2 backbone (src/dcasr/models/mamba_block.py).

CUDA-only: the mamba_ssm kernels require a GPU.
"""
import pytest
import torch

from dcasr.models.mamba_block import MambaBlock, MambaStack, reverse_sequences

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Mamba-2 kernels are CUDA-only")

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


def test_block_shape_preserving():
    blk = MambaBlock(128)
    assert blk(torch.randn(2, 40, 128)).shape == (2, 40, 128)


def test_stack_shape_preserving():
    st = MambaStack(3, 128, bidirectional=True)
    assert st(torch.randn(2, 30, 128)).shape == (2, 30, 128)


def test_gradients_flow():
    st = MambaStack(2, 128)
    x = torch.randn(2, 20, 128, requires_grad=True)
    st(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    p = next(st.parameters())
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_causal_block_ignores_future():
    blk = MambaBlock(128, bidirectional=False).eval()
    x = torch.randn(1, 20, 128)
    y1 = blk(x)
    x2 = x.clone()
    x2[:, 10:] += torch.randn(1, 10, 128)
    y2 = blk(x2)
    assert torch.allclose(y1[:, :10], y2[:, :10], atol=1e-4)      # past unaffected by future


def test_bidirectional_sees_future():
    blk = MambaBlock(128, bidirectional=True).eval()
    x = torch.randn(1, 20, 128)
    y1 = blk(x)
    x2 = x.clone()
    x2[:, 10:] += torch.randn(1, 10, 128)
    y2 = blk(x2)
    assert not torch.allclose(y1[:, :10], y2[:, :10], atol=1e-4)  # future changes past output


def test_reverse_sequences_length_aware():
    x = torch.randn(2, 10, 4)
    lengths = torch.tensor([10, 6])
    r = reverse_sequences(x, lengths)
    assert torch.allclose(reverse_sequences(r, lengths), x)       # double reverse = identity
    assert torch.allclose(r[1, :6], x[1, :6].flip(0))             # valid span reversed
    assert torch.allclose(r[1, 6:], x[1, 6:])                     # padding untouched


def test_reverse_no_lengths_is_flip():
    x = torch.randn(2, 7, 4)
    assert torch.allclose(reverse_sequences(x), torch.flip(x, dims=[1]))


def test_length_aware_bidirectional_runs():
    blk = MambaBlock(128, bidirectional=True)
    x = torch.randn(3, 25, 128)
    y = blk(x, torch.tensor([25, 18, 10]))
    assert y.shape == (3, 25, 128) and torch.isfinite(y).all()


def test_dmodel_headdim_constraint_raises():
    with pytest.raises(AssertionError):
        MambaBlock(80, headdim=64)                               # 160 not divisible by 64
