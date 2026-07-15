"""Unit tests for the ASR task builders (src/dcasr/tasks/asr_task.py).

Head/loss/optim factories are CPU-only; full encoder/model construction is GPU-gated
(the Mamba-2 kernels need CUDA), exercised on synthetic tensors here + real audio in smoke.
"""
import pytest
import torch

from dcasr.decoders.ctc import CTCHead
from dcasr.tasks import asr_task
from dcasr.training.loss import HybridLoss


def _cfg(**over):
    cfg = {
        "encoder": "dcasr", "head": "ctc", "vocab_size": 500,
        "frontend_conf": {"n_mels": 80},
        "encoder_conf": {"arch_type": "A", "d_outer": 384, "d_main": 512, "n_enc": 4,
                         "n_main": 12, "n_dec": 4, "bidirectional": True,
                         "hnet": {"compression_N": 1, "ema_smoothing": True}},
        "model_conf": {"ctc_weight": 1.0, "aed_weight": 0.0, "hnet_ratio_beta": 0.0},
    }
    cfg.update(over)
    return cfg


# ── CPU: head / loss / optim factories ───────────────────────────────────────
def test_build_head_ctc():
    head = asr_task.build_head(_cfg(), 500)
    assert isinstance(head, CTCHead)
    assert head.blank_id == 500 and head.num_classes == 501
    assert head.proj.in_features == 384                         # d_outer


def test_build_loss_from_model_conf():
    loss = asr_task.build_loss(_cfg(model_conf={"ctc_weight": 0.3, "aed_weight": 0.7,
                                                "hnet_ratio_beta": 0.05}))
    assert isinstance(loss, HybridLoss)
    assert (loss.ctc_weight, loss.aed_weight, loss.ratio_weight) == (0.3, 0.7, 0.05)


def test_build_loss_defaults():
    loss = asr_task.build_loss(_cfg(model_conf={}))
    assert (loss.ctc_weight, loss.aed_weight, loss.ratio_weight) == (1.0, 0.0, 0.0)


def test_optim_builders_reexported():
    p = [torch.nn.Parameter(torch.zeros(1))]
    opt = asr_task.build_optimizer(p, "adamw", {"lr": 1e-3})
    assert isinstance(opt, torch.optim.AdamW)
    assert asr_task.build_scheduler(opt, "warmuplr", {"warmup_steps": 5}) is not None


def test_unknown_encoder_and_head_raise():
    with pytest.raises(ValueError):
        asr_task.build_encoder(_cfg(encoder="nope"))
    with pytest.raises(ValueError):
        asr_task.build_head(_cfg(head="nope"), 500)


# ── GPU: full encoder/model construction + forward ───────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_build_model_forward_and_backward():
    torch.manual_seed(0)
    model = asr_task.build_model(_cfg(), 500).cuda()
    B, T, V = 2, 400, 500
    feats = torch.randn(B, T, 80, device="cuda")
    feat_lens = torch.full((B,), T, dtype=torch.long, device="cuda")
    targets = torch.randint(0, V, (B, 6), device="cuda")
    tgt_lens = torch.full((B,), 6, dtype=torch.long, device="cuda")
    loss, stats = model(feats, feat_lens, targets, tgt_lens)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss/total" in stats and "loss/ctc" in stats and "kept_fraction" in stats
    loss.backward()
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert gnorm > 0
    hyps = model.greedy_decode(feats, feat_lens)
    assert len(hyps) == B and all(0 <= i < 500 for h in hyps for i in h)
