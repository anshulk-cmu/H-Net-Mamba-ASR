"""Unit tests for analytic efficiency accounting (src/dcasr/eval/efficiency.py).
CPU: closed forms vs independent arithmetic + pure-torch instantiation; scaling laws;
config adapter; the real script end-to-end. GPU-gated: analytic == instantiated
param counts for the REAL Mamba/encoder/model across the grid's architecture matrix."""
import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from dcasr.eval.efficiency import (aed_flops_per_token, chunker_params,
                                   conv_subsample_flops, conv_subsample_params,
                                   ctc_head_flops, efficiency_report, ema_flops,
                                   encoder_flops, encoder_params, format_efficiency,
                                   head_params, mamba2_flops_per_token, mamba2_params,
                                   mamba_stack_flops, mamba_stack_params, router_flops,
                                   subsampled_frames)

SMALL = {"arch_type": "A", "d_outer": 384, "d_main": 512, "n_enc": 4, "n_main": 12,
         "n_dec": 4, "bidirectional": True, "hnet": {"compression_N": 1,
                                                     "chunker": "dynamic",
                                                     "ema_smoothing": True}}


def _enc(**over):
    e = {k: v for k, v in SMALL.items()}
    e["hnet"] = dict(SMALL["hnet"])
    for k, v in over.items():
        (e["hnet"] if k in e["hnet"] else e).__setitem__(k, v)
    return e


# ── closed forms vs independent arithmetic ───────────────────────────────────
def test_mamba2_params_independent_arithmetic():
    for d in (128, 384, 512):
        d_inner, nheads = 2 * d, 2 * d // 64
        d_in_proj = 2 * d_inner + 2 * 128 + nheads
        conv_dim = d_inner + 2 * 128
        expect = (d * d_in_proj + conv_dim * 4 + conv_dim + 3 * nheads
                  + d_inner + d_inner * d)
        assert mamba2_params(d) == expect


def test_mamba2_flops_independent_arithmetic():
    d = 256
    d_inner, nheads = 512, 8
    d_in_proj = 2 * d_inner + 256 + nheads
    conv_dim = d_inner + 256
    macs = d * d_in_proj + conv_dim * 4 + 2 * d_inner * 128 + d_inner * d
    assert mamba2_flops_per_token(d) == 2.0 * macs


def test_conv_subsample_params_match_instantiation():
    for n_mels, d in ((80, 384), (80, 512), (40, 64)):
        f = ((n_mels - 1) // 2 - 1) // 2
        real = nn.ModuleList([nn.Conv2d(1, d, 3, 2), nn.Conv2d(d, d, 3, 2),
                              nn.Linear(d * f, d)])
        assert conv_subsample_params(n_mels, d) == sum(p.numel() for p in real.parameters())


def test_ctc_and_aed_head_params_match_instantiation():
    from dcasr.decoders.aed import AEDHead
    from dcasr.decoders.ctc import CTCHead
    cfg = {"encoder_conf": {"d_outer": 96},
           "model_conf": {"ctc_weight": 0.3, "aed_weight": 0.7},
           "aed_conf": {"n_layers": 2, "n_heads": 4, "d_ff": 128}}
    p = head_params(cfg, vocab_size=50)
    assert p["ctc_head"] == sum(q.numel() for q in CTCHead(96, 50).parameters())
    assert p["ctc_head"] == (96 + 1) * (50 + 1)
    real_aed = AEDHead(50, 96, n_layers=2, n_heads=4, d_ff=128)
    assert p["aed_head"] == sum(q.numel() for q in real_aed.parameters())
    # independent manual formula for the QK-norm pre-LN decoder internals
    d, ff, n, V, h = 96, 128, 2, 50, 4
    dh = d // h
    # per attention: q/k/v/out projections (4 Linear d->d) + QK-norm gains (q_g,k_g)
    attn = 4 * (d * d + d) + 2 * dh
    # per layer: self+cross attn, 3 LayerNorms, FFN (2 Linear)
    layer = 2 * attn + 3 * 2 * d + (ff * d + ff) + (d * ff + d)
    # head: embed + n layers + final stack LayerNorm + output projection
    assert p["aed_head"] == V * d + n * layer + 2 * d + (V * d + V)


def test_head_gating_matches_build_model_rules():
    cfg = {"encoder_conf": {"d_outer": 64}, "model_conf": {"ctc_weight": 1.0}}
    p = head_params(cfg, 30)
    assert p["ctc_head"] > 0 and p["aed_head"] == 0
    cfg["model_conf"] = {"ctc_weight": 0.0, "aed_weight": 1.0}
    p = head_params(cfg, 30)
    assert p["ctc_head"] == 0 and p["aed_head"] > 0


def test_chunker_params_rules():
    assert chunker_params("dynamic", 384, 2) == 2 * 384 * 384
    assert chunker_params("dynamic", 384, 1) == 0            # N=1 identity: no router
    assert chunker_params("fixed", 384, 4) == 0              # fixed pool: parameter-free


def test_encoder_params_type_a_vs_b_structure():
    a = encoder_params(_enc(compression_N=2))
    assert set(a["breakdown"]) == {"subsample", "enc_stack", "dec_stack", "chunker",
                                   "projections", "main_stack"}
    assert a["breakdown"]["chunker"] == 2 * 384 * 384
    b = encoder_params(_enc(arch_type="B", compression_N=4))
    assert {"mid_stack", "mid_dec_stack"} <= set(b["breakdown"])
    assert b["breakdown"]["chunker"] == 2 * 384 * 384 + 2 * 512 * 512   # both stages
    assert b["total"] > a["total"]                            # mid stacks add params
    n1 = encoder_params(_enc())
    assert n1["breakdown"]["chunker"] == 0


# ── flops formulas + scaling laws ────────────────────────────────────────────
def test_conv_subsample_flops_hand_computed():
    n_frames, n_mels, d = 11, 9, 4
    t1, f1 = 5, 4
    t2, f2 = 2, 1
    expect = 2.0 * (t1 * f1 * d * 9 + t2 * f2 * d * 9 * d + t2 * d * f2 * d)
    assert conv_subsample_flops(n_frames, n_mels, d) == expect
    assert subsampled_frames(11) == 2 and subsampled_frames(3) == 0


def test_stack_router_ema_ctc_formulas():
    assert mamba_stack_flops(3, 64, 10, True) == 3 * 2 * mamba2_flops_per_token(64) * 10
    assert mamba_stack_flops(3, 64, 10, False) == 3 * mamba2_flops_per_token(64) * 10
    assert router_flops(64, 10) == 2.0 * 2 * 64 * 64 * 10
    assert ema_flops(10, 64) == 2.0 * 100 * 64
    assert ema_flops(20, 64) == 4 * ema_flops(10, 64)         # quadratic in length
    assert ctc_head_flops(384, 500, 100) == 2.0 * 100 * 384 * 501


def test_encoder_flops_identity_vs_compressed():
    n1 = encoder_flops(_enc(), 1000)
    assert n1["breakdown"]["router"] == 0 and n1["breakdown"]["ema"] == 0
    assert n1["kept_fractions"] == [1.0]
    n4 = encoder_flops(_enc(compression_N=4), 1000)
    assert n4["kept_fractions"] == [0.25]
    assert n4["breakdown"]["main_stack"] == pytest.approx(
        0.25 * n1["breakdown"]["main_stack"])
    assert n4["breakdown"]["router"] > 0 and n4["breakdown"]["ema"] > 0
    assert n4["total"] < n1["total"]                          # compression saves compute
    fixed = encoder_flops(_enc(compression_N=4, chunker="fixed"), 1000)
    assert fixed["breakdown"]["router"] == 0 and fixed["breakdown"]["ema"] == 0
    no_ema = encoder_flops(_enc(compression_N=4, ema_smoothing=False), 1000)
    assert no_ema["breakdown"]["ema"] == 0


def test_encoder_flops_kept_override_and_guard():
    r = encoder_flops(_enc(compression_N=2), 1000, kept_fractions=[0.6])
    assert r["kept_fractions"] == [0.6]
    assert r["compressed_frames"][0] == pytest.approx(0.6 * r["frames_25hz"])
    with pytest.raises(ValueError, match="kept fraction"):
        encoder_flops(_enc(compression_N=2), 1000, kept_fractions=[0.5, 0.5])


def test_encoder_flops_type_b():
    b = encoder_flops(_enc(arch_type="B", compression_N=4), 1000)
    l0 = b["frames_25hz"]
    assert b["kept_fractions"] == [0.5, 0.5]
    assert b["compressed_frames"] == [pytest.approx(0.5 * l0), pytest.approx(0.25 * l0)]
    assert {"mid_stack", "mid_dec_stack"} <= set(b["breakdown"])
    assert b["breakdown"]["router"] == router_flops(384, l0) + router_flops(512, 0.5 * l0)
    assert b["breakdown"]["ema"] == ema_flops(0.5 * l0, 512) + ema_flops(l0, 384)
    bf = encoder_flops(_enc(arch_type="B", compression_N=4, chunker="fixed"), 1000)
    assert bf["breakdown"]["router"] == 0 and bf["breakdown"]["ema"] == 0


def test_stack_flops_linear_in_length():
    one = mamba_stack_flops(4, 384, 250, True)
    assert mamba_stack_flops(4, 384, 500, True) == 2 * one


# ── report + script ──────────────────────────────────────────────────────────
def _cfg(vocab_note="unused", **model_conf):
    return {"experiment": {"name": "eff_test", "seed": 1},
            "frontend_conf": {"n_mels": 80},
            "encoder_conf": _enc(),
            "model_conf": model_conf or {"ctc_weight": 1.0},
            "aed_conf": {"n_layers": 2, "n_heads": 4, "d_ff": 128}}


def test_efficiency_report_structure_and_sums():
    rep = efficiency_report(_cfg(), 100, audio_seconds=8.0)
    assert rep["params"]["total"] == (rep["params"]["encoder"] + rep["params"]["ctc_head"]
                                      + rep["params"]["aed_head"])
    f = rep["flops"]
    assert f["gflops_total"] == pytest.approx(sum(f["breakdown_gflops"].values()))
    assert f["gflops_per_second"] == pytest.approx(f["gflops_total"] / 8.0)
    assert f["input_frames"] == 800 and "ctc_head" in f["breakdown_gflops"]
    assert "aed_secondary" not in f and rep["assumptions"]
    hybrid = efficiency_report(_cfg(ctc_weight=0.3, aed_weight=0.7), 100)
    assert hybrid["params"]["aed_head"] > 0 and "aed_secondary" in hybrid["flops"]
    assert hybrid["flops"]["aed_secondary"]["per_token"] > 0
    table = format_efficiency(rep)
    assert "main_stack" in table and "GFLOPs/s" in table


def test_aed_flops_per_token_formula():
    d, ff, n, V, mem, ctx = 96, 128, 2, 50, 200.0, 16.0
    per_layer = (4 * d * d + 2 * ctx * d + 2 * d * d + 2 * mem * d + 2 * d * ff)
    out = aed_flops_per_token(V, d, n, ff, memory_len=mem, ctx_len=ctx)
    assert out["per_token"] == 2.0 * (n * per_layer + d * V)
    assert out["memory_kv_per_utt"] == 2.0 * n * 2 * mem * d * d


WORDS = "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG NEAR RIVERS".split()


def test_script_run_end_to_end(tmp_path):
    from omegaconf import OmegaConf
    from dcasr.data.tokenizer import Tokenizer
    rng = random.Random(0)
    lines = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
             for _ in range(200)]
    tok = Tokenizer.train(lines, tmp_path / "sp", vocab_size=60, hard_vocab_limit=False)

    spec = importlib.util.spec_from_file_location(
        "efficiency_script", Path(__file__).resolve().parents[1] / "scripts" / "efficiency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] /
                         "configs" / "typeA_small_N1_ctc.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
        ["bpemodel=sp.model", "experiment.name=eff_smoke",
         "efficiency.audio_seconds=6.0"]))
    rep = mod.run(cfg, repo_root=tmp_path)
    out = tmp_path / "experiments" / "eff_smoke" / "efficiency"
    saved = json.loads((out / "efficiency.json").read_text())
    assert saved["params"]["total"] == rep["params"]["total"]
    assert saved["flops"]["audio_seconds"] == 6.0
    assert saved["arch"]["vocab"] == tok.vocab_size
    assert saved["params"]["encoder"] == encoder_params(
        {"arch_type": "A", "d_outer": 384, "d_main": 512, "n_enc": 4, "n_main": 12,
         "n_dec": 4, "hnet": {"compression_N": 1}})["total"]
    assert "provenance" in saved and (out / "report.txt").exists()


# ── GPU-gated: analytic == instantiated for the REAL modules ─────────────────
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (mamba_ssm)")


@gpu
def test_mamba_params_match_real_modules():
    from dcasr.models.mamba_block import MambaStack
    from mamba_ssm import Mamba2
    for d in (128, 384, 512):
        real = sum(p.numel() for p in
                   Mamba2(d_model=d, d_state=128, d_conv=4, expand=2, headdim=64).parameters())
        assert mamba2_params(d) == real
    for n, d, bidir in ((3, 128, True), (2, 384, False)):
        real = sum(p.numel() for p in MambaStack(n, d, bidir).parameters())
        assert mamba_stack_params(n, d, bidir) == real


@gpu
@pytest.mark.parametrize("over", [
    {},                                                       # A dynamic N=1 (go/no-go)
    {"compression_N": 2},
    {"compression_N": 4, "chunker": "fixed"},
    {"arch_type": "B", "compression_N": 4},
    {"arch_type": "B", "compression_N": 4, "chunker": "fixed"},
    {"d_outer": 512, "d_main": 768, "n_enc": 6, "n_main": 18, "n_dec": 6},  # Large
])
def test_encoder_params_match_real_encoder(over):
    from dcasr.models.encoder import DCASREncoder
    e = _enc(**over)
    h = e["hnet"]
    real = DCASREncoder(n_mels=80, d_outer=e["d_outer"], d_main=e["d_main"],
                        n_enc=e["n_enc"], n_main=e["n_main"], n_dec=e["n_dec"],
                        arch_type=e["arch_type"], N=h["compression_N"],
                        bidirectional=e["bidirectional"], chunker=h["chunker"])
    assert encoder_params(e)["total"] == sum(p.numel() for p in real.parameters())


@gpu
def test_model_params_match_real_build_model():
    from dcasr.tasks.asr_task import build_model
    cfg = {"frontend_conf": {"n_mels": 80}, "encoder_conf": _enc(),
           "encoder": "dcasr", "head": "ctc", "model_conf": {"ctc_weight": 1.0},
           "aed_conf": {"n_layers": 6, "n_heads": 4, "d_ff": 2048}}
    rep = efficiency_report(cfg, 500)
    real = sum(p.numel() for p in build_model(cfg, 500).parameters())
    assert rep["params"]["total"] == real                     # the known ~62M Small
    cfg["model_conf"] = {"ctc_weight": 0.3, "aed_weight": 0.7}
    rep = efficiency_report(cfg, 500)
    real = sum(p.numel() for p in build_model(cfg, 500).parameters())
    assert rep["params"]["total"] == real                     # the known ~67.8M hybrid


# ── fixes from the adversarial verification (wf_68be3655) ────────────────────
def test_chunker_name_case_insensitive():
    """The build seam lowercases chunker names; the formulas must agree."""
    assert chunker_params("Dynamic", 384, 2) == 2 * 384 * 384
    assert chunker_params("DYNAMIC", 384, 2) == 2 * 384 * 384
    assert chunker_params("Fixed", 384, 4) == 0
    r = encoder_flops(_enc(compression_N=2, chunker="DYNAMIC"), 1000)
    assert r["breakdown"]["router"] > 0 and r["breakdown"]["ema"] > 0
    assert (encoder_params(_enc(compression_N=2, chunker="Dynamic"))["total"]
            == encoder_params(_enc(compression_N=2))["total"])


def test_unbuildable_configs_rejected():
    with pytest.raises(ValueError, match="perfect-square"):     # real FixedPool refuses
        encoder_params(_enc(arch_type="B", compression_N=2, chunker="fixed"))
    with pytest.raises(ValueError, match="headdim"):            # no such Mamba2
        mamba2_params(100)
    with pytest.raises(ValueError, match="headdim"):
        mamba2_flops_per_token(100)
    with pytest.raises(ValueError, match="ctc_weight > 0 or aed_weight"):
        head_params({"encoder_conf": {"d_outer": 64},
                     "model_conf": {"ctc_weight": 0.0, "aed_weight": 0.0}}, 30)


def test_compression_n_cast_matches_build_seam():
    """asr_task casts N with int(); float 2.5 must price the model that would train."""
    r = encoder_flops(_enc(compression_N=2.5), 1000)
    assert r["kept_fractions"] == [0.5]                         # int(2.5) == 2
    assert encoder_params(_enc(compression_N=2.5)) == encoder_params(_enc(compression_N=2))


def test_audio_seconds_and_kept_guards():
    with pytest.raises(ValueError, match="audio_seconds"):
        efficiency_report(_cfg(), 100, audio_seconds=0.0)
    with pytest.raises(ValueError, match="audio_seconds"):
        efficiency_report(_cfg(), 100, audio_seconds=-5.0)
    enc = _enc(compression_N=2)
    with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
        encoder_flops(enc, 1000, kept_fractions=[1.5])
    with pytest.raises(ValueError, match=r"outside \(0, 1\]"):
        encoder_flops(enc, 1000, kept_fractions=[-0.3])
    with pytest.raises(ValueError, match="must be a list"):
        encoder_flops(enc, 1000, kept_fractions=0.52)
    with pytest.raises(ValueError, match="kept fraction"):      # [] is not "use default"
        encoder_flops(enc, 1000, kept_fractions=[])
    assert encoder_flops(enc, 1000, kept_fractions=[1.0])["kept_fractions"] == [1.0]


def test_report_table_head_params_column():
    rep = efficiency_report(_cfg(), 100, audio_seconds=8.0)
    line = next(l for l in format_efficiency(rep).splitlines() if l.startswith("ctc_head"))
    shown = float(line.split()[1])                              # params(M) column
    assert shown == pytest.approx(rep["params"]["ctc_head"] / 1e6, abs=0.005)
    assert shown > 0
