"""Unit tests for the hybrid training loss (src/dcasr/training/loss.py).

CPU-only (no GPU gate): pure weighted-combination arithmetic over scalar losses.
"""
import pytest
import torch

from dcasr.training.loss import HybridLoss, LossOutput


def _t(x):
    return torch.tensor(float(x))


def test_weighted_sum():
    hl = HybridLoss(0.3, 0.7, 0.1)
    out = hl(ctc_loss=_t(2.0), aed_loss=_t(3.0), ratio_loss=_t(4.0))
    assert out.total.item() == pytest.approx(0.3 * 2 + 0.7 * 3 + 0.1 * 4)   # 3.1
    # raw components are stored UNWEIGHTED (for logging)
    assert (out.ctc.item(), out.aed.item(), out.ratio.item()) == (2.0, 3.0, 4.0)


def test_ctc_only_equals_ctc():
    hl = HybridLoss(1.0, 0.0, 0.0)                       # the go/no-go config
    out = hl(ctc_loss=_t(5.5))
    assert out.total.item() == pytest.approx(5.5)
    assert out.aed.item() == 0.0 and out.ratio.item() == 0.0


def test_zero_weight_drops_term():
    hl = HybridLoss(1.0, 0.0, 0.0)
    # aed present but zero-weighted -> cannot change the total
    out = hl(ctc_loss=_t(2.0), aed_loss=_t(1000.0), ratio_loss=_t(1000.0))
    assert out.total.item() == pytest.approx(2.0)
    assert out.aed.item() == 1000.0                     # raw value still reported


def test_from_config():
    hl = HybridLoss.from_config({"ctc_weight": 0.3, "aed_weight": 0.7, "ratio_weight": 0.05})
    assert (hl.ctc_weight, hl.aed_weight, hl.ratio_weight) == (0.3, 0.7, 0.05)
    d = HybridLoss.from_config({})                       # defaults
    assert (d.ctc_weight, d.aed_weight, d.ratio_weight) == (1.0, 0.0, 0.0)


def test_missing_component_with_positive_weight_raises():
    with pytest.raises(ValueError):
        HybridLoss(0.3, 0.7, 0.0)(ctc_loss=_t(1.0))     # aed weighted but missing
    with pytest.raises(ValueError):
        HybridLoss(1.0, 0.0, 0.1)(ctc_loss=_t(1.0))     # ratio weighted but missing
    with pytest.raises(ValueError):
        HybridLoss(1.0, 0.0, 0.0)(aed_loss=_t(1.0))     # ctc weighted but missing


def test_all_none_raises():
    with pytest.raises(ValueError):
        HybridLoss(1.0, 0.0, 0.0)()


def test_missing_optional_is_zero_not_error():
    hl = HybridLoss(1.0, 0.0, 0.0)                       # aed/ratio weight 0 -> may be omitted
    out = hl(ctc_loss=_t(3.0))
    assert out.total.item() == pytest.approx(3.0)
    assert out.total.dtype == out.ctc.dtype              # zero matches ref device/dtype


def test_gradients_flow_and_scalar():
    hl = HybridLoss(0.3, 0.7, 0.1)
    ctc = torch.tensor(2.0, requires_grad=True)
    aed = torch.tensor(3.0, requires_grad=True)
    ratio = torch.tensor(4.0, requires_grad=True)
    out = hl(ctc_loss=ctc, aed_loss=aed, ratio_loss=ratio)
    assert out.total.ndim == 0                           # a scalar to backprop
    out.total.backward()
    assert ctc.grad.item() == pytest.approx(0.3)         # d total / d ctc == w_ctc
    assert aed.grad.item() == pytest.approx(0.7)
    assert ratio.grad.item() == pytest.approx(0.1)


def test_items_for_logging():
    out = HybridLoss(1.0, 0.0, 0.0)(ctc_loss=_t(2.0))
    d = out.items()
    assert set(d) == {"loss/total", "loss/ctc", "loss/aed", "loss/ratio"}
    assert d["loss/total"].item() == pytest.approx(2.0)
    assert isinstance(out, LossOutput)
