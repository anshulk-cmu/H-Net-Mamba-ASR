"""Unit tests for the ASR task builders (src/dcasr/tasks/asr_task.py).

Head/loss/optim factories are CPU-only; full encoder/model construction is GPU-gated
(the Mamba-2 kernels need CUDA), exercised on synthetic tensors here + real audio in smoke.
"""
import pytest
import torch

from dcasr.decoders.aed import AEDHead
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


def test_build_aed_head():
    head = asr_task.build_aed_head(_cfg(aed_conf={"n_layers": 3, "n_heads": 4, "d_ff": 128},
                                        model_conf={"lsm_weight": 0.1}), 500)
    assert isinstance(head, AEDHead)
    assert head.vocab_size == 500 and head.d_model == 384        # d_outer
    assert len(head.decoder.layers) == 3 and head.lsm_weight == 0.1


def test_build_model_requires_a_head():
    with pytest.raises(ValueError):                              # early guard, before build_encoder
        asr_task.build_model(_cfg(model_conf={"ctc_weight": 0.0, "aed_weight": 0.0}), 500)


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
    assert model.ctc_head is not None and model.aed_head is None    # CTC-only build
    hyps = model.greedy_decode(feats, feat_lens)
    assert len(hyps) == B and all(0 <= i < 500 for h in hyps for i in h)


def _gpu_batch(B=2, T=400, V=500):
    return (torch.randn(B, T, 80, device="cuda"),
            torch.full((B,), T, dtype=torch.long, device="cuda"),
            torch.randint(4, V, (B, 6), device="cuda"),
            torch.full((B,), 6, dtype=torch.long, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_build_model_hybrid_two_heads():
    torch.manual_seed(0)
    cfg = _cfg(model_conf={"ctc_weight": 0.3, "aed_weight": 0.7, "hnet_ratio_beta": 0.0},
               aed_conf={"n_layers": 2, "n_heads": 4, "d_ff": 256})
    model = asr_task.build_model(cfg, 500).cuda()
    assert model.ctc_head is not None and model.aed_head is not None
    feats, feat_lens, targets, tgt_lens = _gpu_batch()
    loss, stats = model(feats, feat_lens, targets, tgt_lens)
    assert torch.isfinite(loss) and stats["loss/ctc"] > 0 and stats["loss/aed"] > 0
    # total == 0.3*ctc + 0.7*aed (raw components in stats)
    assert torch.allclose(stats["loss/total"], 0.3 * stats["loss/ctc"] + 0.7 * stats["loss/aed"],
                          atol=1e-4)
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.ctc_head.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.aed_head.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in model.encoder.parameters()) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_build_model_aed_only():
    cfg = _cfg(model_conf={"ctc_weight": 0.0, "aed_weight": 1.0},
               aed_conf={"n_layers": 2, "n_heads": 4, "d_ff": 256})
    model = asr_task.build_model(cfg, 500).cuda()
    assert model.ctc_head is None and model.aed_head is not None
    feats, feat_lens, targets, tgt_lens = _gpu_batch()
    loss, stats = model(feats, feat_lens, targets, tgt_lens)
    assert torch.isfinite(loss) and stats["loss/aed"] > 0
    hyps = model.greedy_decode(feats, feat_lens)                # falls back to AED greedy
    assert len(hyps) == 2 and all(0 <= i < 500 for h in hyps for i in h)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_build_encoder_fixed_chunker_from_config():
    """The hnet.chunker key routes build_encoder to the fixed-pool control (H2)."""
    from dcasr.models.fixed_pool import FixedPoolChunker
    from dcasr.models.hnet_chunk import DynamicChunker
    ec = {"arch_type": "A", "d_outer": 384, "d_main": 512, "n_enc": 2, "n_main": 2,
          "n_dec": 2, "bidirectional": True}
    fixed = asr_task.build_encoder(_cfg(encoder_conf={
        **ec, "hnet": {"compression_N": 2, "chunker": "fixed"}}))
    assert isinstance(fixed.chunk, FixedPoolChunker) and fixed.chunk.stride == 2
    dyn = asr_task.build_encoder(_cfg(encoder_conf={
        **ec, "hnet": {"compression_N": 2}}))                   # default → dynamic
    assert isinstance(dyn.chunk, DynamicChunker)
    feats = torch.randn(2, 400, 80, device="cuda")
    lens = torch.full((2,), 400, dtype=torch.long, device="cuda")
    out = fixed.cuda()(feats, lens)
    assert out.ratio_loss.item() == 0.0 and torch.isfinite(out.features).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_type_b_per_stage_kept_fractions_logged():
    torch.manual_seed(0)
    cfg = _cfg(encoder_conf={"arch_type": "B", "d_outer": 64, "d_main": 128, "n_enc": 2,
                             "n_main": 2, "n_dec": 2, "n_mid": 2, "bidirectional": True,
                             "hnet": {"compression_N": 4}})
    model = asr_task.build_model(cfg, 500).cuda()
    feats, feat_lens, targets, tgt_lens = _gpu_batch()
    _, stats = model(feats, feat_lens, targets, tgt_lens)
    assert "kept_fraction" in stats and "kept_fraction_1" in stats   # BOTH stages observable


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 kernels need CUDA")
def test_ctc_infeasible_counter():
    torch.manual_seed(0)
    model = asr_task.build_model(_cfg(), 500).cuda()
    feats, feat_lens, targets, tgt_lens = _gpu_batch(B=2, T=400)     # enc ~99 frames >> 6 tokens
    _, stats = model(feats, feat_lens, targets, tgt_lens)
    assert int(stats["ctc_infeasible"]) == 0
    # T=100 -> enc_len 24; row0 has 60 (distinct) tokens > 24 -> infeasible; row1 has 5 -> fine
    feats2 = torch.randn(2, 100, 80, device="cuda")
    lens2 = torch.full((2,), 100, dtype=torch.long, device="cuda")
    tgt2 = torch.arange(4, 64, device="cuda").unsqueeze(0).repeat(2, 1)
    tl2 = torch.tensor([60, 5], dtype=torch.long, device="cuda")
    _, stats2 = model(feats2, lens2, tgt2, tl2)
    assert int(stats2["ctc_infeasible"]) == 1
