"""Unit tests for the acoustic frontend (src/dcasr/data/features.py)."""
import math

import pytest
import torch

from dcasr.data.features import (
    CMVNAccumulator, GlobalCMVN, LogMelFrontend, SpecAugment,
    HOP_LENGTH, LOG_FLOOR, N_MELS, WIN_LENGTH,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DC-ASR targets CUDA (Babel GPU nodes)"
)

torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _cuda_default_device():
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


# ── LogMelFrontend: the [B, T, 80] @ 100 Hz contract ─────────────────────────
def test_shape_and_frame_count():
    fe = LogMelFrontend()
    wave = torch.randn(3, 16000)
    feats, lens = fe(wave)
    assert feats.shape == (3, 98, N_MELS)          # 1 + (16000-400)//160
    assert torch.all(lens == 98)
    feats10, lens10 = fe(torch.randn(1, 160000))   # 10 s -> ~100 frames/s
    assert lens10.item() == 998 and feats10.shape[1] == 998


def test_frame_count_formula_edges():
    fe = LogMelFrontend()
    n = torch.tensor([16000, WIN_LENGTH, WIN_LENGTH - 1, 560])
    assert fe.frame_count(n).tolist() == [98, 1, 0, 2]


def test_1d_input_treated_as_batch_of_one():
    fe = LogMelFrontend()
    feats, lens = fe(torch.randn(16000))
    assert feats.shape == (1, 98, N_MELS) and lens.tolist() == [98]


def test_invalid_inputs_raise():
    fe = LogMelFrontend()
    with pytest.raises(ValueError):
        fe(torch.randn(2, 3, 16000))
    with pytest.raises(ValueError):
        fe(torch.randn(1, WIN_LENGTH - 1))


def test_empty_batch():
    fe = LogMelFrontend()
    feats, lens = fe(torch.randn(0, 16000))
    assert feats.shape == (0, 98, N_MELS) and lens.shape == (0,)


def test_fp64_and_half_waveforms_promote_to_fp32():
    fe = LogMelFrontend()
    wave = torch.randn(2, 16000)
    ref, _ = fe(wave)
    feats64, _ = fe(wave.double())
    assert feats64.dtype == torch.float32
    assert torch.allclose(feats64, ref, atol=1e-5)
    for dt in (torch.float16, torch.bfloat16):
        feats, _ = fe(wave.to(dt))
        assert feats.dtype == torch.float32
        loud = ref > -5.0        # half-precision audio only matches on energetic bins
        assert torch.allclose(feats[loud], ref[loud], atol=0.2)


def test_cpu_lengths_returned_on_wave_device():
    fe = LogMelFrontend()
    wave = torch.randn(2, 16000)
    _, lens = fe(wave, torch.tensor([16000, 9000], device="cpu"))
    assert lens.device == wave.device


def test_padding_never_leaks_into_valid_frames():
    fe = LogMelFrontend()
    wave = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 9000])
    wave[1, 9000:] = 0.0
    feats_a, lens = fe(wave, lengths)
    garbage = wave.clone()
    garbage[1, 9000:] = 1e4
    feats_b, _ = fe(garbage, lengths)
    n_valid = int(lens[1])
    assert torch.equal(feats_a[1, :n_valid], feats_b[1, :n_valid])
    assert torch.equal(feats_a[0], feats_b[0])


def test_batched_matches_per_utterance():
    fe = LogMelFrontend()
    w0, w1 = torch.randn(16000), torch.randn(9000)
    batch = torch.zeros(2, 16000)
    batch[0], batch[1, :9000] = w0, w1
    feats, lens = fe(batch, torch.tensor([16000, 9000]))
    solo0, _ = fe(w0)
    solo1, _ = fe(w1)
    assert torch.allclose(feats[0, :int(lens[0])], solo0[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(feats[1, :int(lens[1])], solo1[0], atol=1e-5, rtol=1e-5)


def test_deterministic():
    fe = LogMelFrontend()
    wave = torch.randn(2, 12000)
    a, _ = fe(wave)
    b, _ = fe(wave)
    assert torch.equal(a, b)


def test_tone_and_silence_sanity():
    fe = LogMelFrontend()
    t = torch.arange(16000, dtype=torch.float32) / 16000.0
    tone = torch.sin(2 * math.pi * 1000.0 * t).unsqueeze(0)
    feats, _ = fe(tone)
    peak = int(feats[0].mean(dim=0).argmax())
    assert 20 < peak < 40                          # 1 kHz lands mid-low mel range
    silence, _ = fe(torch.zeros(1, 16000))
    assert torch.allclose(silence, torch.full_like(silence, math.log(LOG_FLOOR)))


# ── Global CMVN ───────────────────────────────────────────────────────────────
def _random_feats():
    feats = torch.randn(4, 300, N_MELS) * 3.0 - 7.0
    lengths = torch.tensor([300, 250, 100, 17])
    return feats, lengths


def test_cmvn_normalises_to_zero_mean_unit_var():
    feats, lengths = _random_feats()
    acc = CMVNAccumulator()
    acc.update(feats, lengths)
    stats = acc.finalize()
    cm = GlobalCMVN(stats["mean"], stats["std"]).to(feats.device)
    normed = cm(feats)
    valid = torch.arange(300) < lengths[:, None]
    sel = normed[valid]
    assert torch.allclose(sel.mean(dim=0), torch.zeros(N_MELS), atol=1e-4)
    assert torch.allclose(sel.std(dim=0, unbiased=False), torch.ones(N_MELS), atol=1e-3)


def test_cmvn_ignores_padding():
    feats, lengths = _random_feats()
    acc_a = CMVNAccumulator()
    acc_a.update(feats, lengths)
    poisoned = feats.clone()
    valid = torch.arange(300) < lengths[:, None]
    poisoned[~valid] = 1e6
    acc_b = CMVNAccumulator()
    acc_b.update(poisoned, lengths)
    assert torch.equal(acc_a.sum, acc_b.sum) and torch.equal(acc_a.sumsq, acc_b.sumsq)
    assert acc_a.count == acc_b.count == int(lengths.sum())


def test_cmvn_save_load_roundtrip(tmp_path):
    feats, lengths = _random_feats()
    acc = CMVNAccumulator()
    acc.update(feats, lengths)
    stats = acc.save(tmp_path / "cmvn.pt")
    cm = GlobalCMVN.load(tmp_path / "cmvn.pt").to(feats.device)
    ref = GlobalCMVN(stats["mean"], stats["std"]).to(feats.device)
    assert torch.allclose(cm(feats), ref(feats))


def test_cmvn_constant_features_no_nan():
    acc = CMVNAccumulator()
    acc.update(torch.full((2, 50, N_MELS), 3.14))
    stats = acc.finalize()
    x = torch.full((1, 5, N_MELS), 3.14)
    cm = GlobalCMVN(stats["mean"], stats["std"]).to(x.device)
    assert torch.isfinite(cm(x)).all()


def test_cmvn_empty_accumulator_raises():
    with pytest.raises(RuntimeError):
        CMVNAccumulator().finalize()


# ── SpecAugment ───────────────────────────────────────────────────────────────
def test_specaugment_eval_is_identity():
    sa = SpecAugment().eval()
    x = torch.randn(2, 100, N_MELS)
    assert torch.equal(sa(x), x)


def test_specaugment_masks_are_zero_and_rest_untouched():
    sa = SpecAugment().train()
    x = torch.randn(4, 200, N_MELS) + 5.0          # keep values away from 0
    out = sa(x)
    changed = out != x
    assert changed.any()
    assert torch.all(out[changed] == 0.0)
    assert torch.equal(out[~changed], x[~changed])


def test_specaugment_time_masks_stay_inside_lengths():
    sa = SpecAugment(freq_masks=0, time_masks=2, time_width=100).train()
    x = torch.randn(4, 200, N_MELS) + 5.0
    lengths = torch.tensor([200, 150, 60, 10])
    for _ in range(20):
        out = sa(x, lengths)
        changed_t = (out != x).any(dim=2)          # [B, T]
        for i, n in enumerate(lengths.tolist()):
            assert not changed_t[i, n:].any()


def test_specaugment_freq_mask_budget():
    sa = SpecAugment(freq_masks=2, freq_width=27, time_masks=0).train()
    x = torch.randn(2, 100, N_MELS) + 5.0
    for _ in range(20):
        out = sa(x)
        masked_bins = (out != x).any(dim=1).sum(dim=1)   # [B]
        assert torch.all(masked_bins <= 2 * 27)


def test_specaugment_reproducible_and_varies_across_batch():
    sa = SpecAugment().train()
    x = torch.randn(8, 200, N_MELS) + 5.0
    torch.manual_seed(7)
    a = sa(x)
    torch.manual_seed(7)
    b = sa(x)
    assert torch.equal(a, b)
    rows = [(a[i] != x[i]) for i in range(8)]
    assert any(not torch.equal(rows[0], r) for r in rows[1:])


def test_specaugment_gradients_flow():
    sa = SpecAugment().train()
    x = torch.randn(2, 100, N_MELS, requires_grad=True)
    sa(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.max() == 1.0

def test_specaugment_generator_determinism():
    """SpecAugment masks are a deterministic function of the generator seed (resume-exact)."""
    sa = SpecAugment()
    sa.train()
    x = torch.randn(1, 200, 80, device="cpu")            # CPU: mirrors the DataLoader-worker path
    y1 = sa(x.clone(), generator=torch.Generator().manual_seed(123))
    y2 = sa(x.clone(), generator=torch.Generator().manual_seed(123))
    y3 = sa(x.clone(), generator=torch.Generator().manual_seed(999))
    assert torch.equal(y1, y2)                            # same seed -> identical masks
    assert not torch.equal(y1, y3)                        # different seed -> different masks
    assert (y1 == 0).any()                                # something was actually masked
    sa.eval()
    assert torch.equal(sa(x.clone()), x)                 # eval mode: no-op
