"""Unit tests for the optimizer/scheduler factories (src/dcasr/optim.py). CPU-only."""
import pytest
import torch

from dcasr.optim import WarmupLR, build_optimizer, build_scheduler


def _expected(step, base, warmup):
    return base * warmup ** 0.5 * min(step ** -0.5, step * warmup ** -1.5)


def test_warmuplr_matches_espnet_formula():
    base, warmup = 1e-3, 100
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=base)
    sch = WarmupLR(opt, warmup_steps=warmup)
    for i in range(400):
        got = opt.param_groups[0]["lr"]                 # lr for step_num = i+1
        assert got == pytest.approx(_expected(i + 1, base, warmup), rel=1e-9)
        opt.step()                                      # real order: optimizer before scheduler
        sch.step()


def test_warmuplr_peak_at_warmup_and_shape():
    base, warmup = 1e-3, 50
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=base)
    sch = WarmupLR(opt, warmup_steps=warmup)
    lrs = []
    for _ in range(300):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()                                      # real order: optimizer before scheduler
        sch.step()
    assert lrs[warmup - 1] == pytest.approx(base, rel=1e-9)   # peak == base_lr at step==warmup
    assert lrs[0] < lrs[warmup - 1] > lrs[-1]                 # ramps up then decays
    assert all(a < b for a, b in zip(lrs[:warmup - 1], lrs[1:warmup]))   # strictly up in warmup
    assert lrs[warmup] > lrs[warmup + 50]                     # decays after


def test_build_optimizer_adamw_kwargs():
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = build_optimizer(p, "adamw", {"lr": 2e-3, "weight_decay": 1e-6, "betas": [0.9, 0.98]})
    assert isinstance(opt, torch.optim.AdamW)
    g = opt.param_groups[0]
    assert g["lr"] == 2e-3 and g["weight_decay"] == 1e-6 and g["betas"] == (0.9, 0.98)


def test_build_optimizer_unknown_raises():
    with pytest.raises(ValueError):
        build_optimizer([torch.nn.Parameter(torch.zeros(1))], "nope", {})


def test_build_scheduler_warmuplr_and_none():
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = build_optimizer(p, "adamw", {"lr": 1e-3})
    assert isinstance(build_scheduler(opt, "warmuplr", {"warmup_steps": 10}), WarmupLR)
    assert build_scheduler(opt, None) is None
    assert build_scheduler(opt, "none") is None


def test_build_scheduler_unknown_raises():
    opt = build_optimizer([torch.nn.Parameter(torch.zeros(1))], "adamw", {"lr": 1e-3})
    with pytest.raises(ValueError):
        build_scheduler(opt, "nope", {})
