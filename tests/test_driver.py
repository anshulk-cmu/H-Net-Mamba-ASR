"""Unit tests for the interp driver (src/dcasr/interp/driver.py). CPU-only:
perturbation math + waveform effects, mandate guards (disjointness, durations,
coverage), checkpoint enumeration, robustness scoring on hand cases, emergence
orchestration with fake encoders/loaders."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from dcasr.data.features import LogMelFrontend
from dcasr.data.librispeech import feat_frames
from dcasr.interp.boundary_align import match_boundaries
from dcasr.interp.driver import (NoisePerturbation, Perturbation,
                                 PerturbedDataset, RecordingLoader,
                                 SilencePerturbation, SpeedPerturbation,
                                 assert_disjoint, boundary_report,
                                 durations_from_entries, emergence_report,
                                 flatten_metrics, list_epoch_checkpoints,
                                 matched_deltas, perturbations_from_config,
                                 probe_report, robustness_report,
                                 score_perturbation, utt_seed)
from dcasr.interp.probes import chunk_spans


# ── perturbations ────────────────────────────────────────────────────────────
def test_utt_seed_stable_and_distinct():
    assert utt_seed(1, "u1") == utt_seed(1, "u1")        # crc32: hash()-free
    assert utt_seed(1, "u1") != utt_seed(1, "u2")
    assert utt_seed(1, "u1") != utt_seed(2, "u1")
    assert 0 <= utt_seed(7, "1272-128104-0000") < 2 ** 31


def test_noise_snr_and_determinism():
    g = torch.Generator().manual_seed(0)
    wave = torch.randn(1, 16000, generator=g) * 0.1
    pert = NoisePerturbation(10.0)
    out = pert.apply_wave(wave, "u1", seed=1)
    noise = out - wave
    snr_db = 10 * torch.log10(wave.pow(2).mean() / noise.pow(2).mean())
    assert abs(float(snr_db) - 10.0) < 0.5
    assert torch.equal(out, pert.apply_wave(wave, "u1", seed=1))     # per-utt seeded
    assert not torch.equal(out, pert.apply_wave(wave, "u2", seed=1))
    assert not torch.equal(out, pert.apply_wave(wave, "u1", seed=2))
    silent = torch.zeros(1, 100)
    assert torch.equal(pert.apply_wave(silent, "u1"), silent)        # SNR undefined
    assert pert.transform_times([0.5], 2.0) == [0.5]                 # timings untouched
    assert pert.transform_duration(2.0) == 2.0 and pert.transform_samples(5) == 5


def test_speed_perturbation_time_math():
    pert = SpeedPerturbation(2.0)
    assert pert.transform_times([0.5, 1.0], 2.0) == [0.25, 0.5]
    assert pert.transform_duration(2.0) == 1.0
    assert pert.transform_samples(16000) == 8000
    wave = torch.randn(1, 16000, generator=torch.Generator().manual_seed(0))
    out = pert.apply_wave(wave, "u1")
    assert abs(out.shape[-1] - 8000) <= 100              # resampler rounding only
    assert SpeedPerturbation(1.0).apply_wave(wave, "u1").shape == wave.shape
    with pytest.raises(ValueError, match="positive"):
        SpeedPerturbation(0.0)


def test_silence_perturbation_wave_and_times():
    pert = SilencePerturbation(0.1, at_frac=0.5)         # 1600 samples inserted
    wave = torch.arange(1.0, 1601.0).view(1, 1600)       # nonzero everywhere
    out = pert.apply_wave(wave, "u1", sample_rate=16000)
    assert out.shape == (1, 3200)
    assert torch.equal(out[0, :800], wave[0, :800])
    assert torch.all(out[0, 800:2400] == 0)
    assert torch.equal(out[0, 2400:], wave[0, 800:])
    # duration 0.1 s -> T0 = 0.05: earlier times fixed, later shifted by 0.1
    assert pert.transform_times([0.04, 0.05, 0.09], 0.1) == (
        pytest.approx([0.04, 0.15, 0.19]))
    assert pert.transform_duration(0.1) == pytest.approx(0.2)
    assert pert.transform_samples(1600) == 3200
    assert pert.window(0.1) == (pytest.approx(0.05), pytest.approx(0.15))
    # a unit spanning T0 stretches: start fixed, end shifted
    rec = {"id": "u", "words": [["w", 0.02, 0.08]], "phones": [["W", 0.02, 0.08]]}
    t = pert.transform_record(rec, 0.1)
    assert t["words"] == [["w", 0.02, pytest.approx(0.18)]] and t["id"] == "u"
    with pytest.raises(ValueError, match="at_frac"):
        SilencePerturbation(0.5, at_frac=1.5)
    with pytest.raises(ValueError, match="duration_s"):
        SilencePerturbation(0.0)


def test_perturbations_from_config():
    default = perturbations_from_config({})
    assert [p.kind for p in default] == ["noise"] * 4 + ["speed"] * 2 + ["silence"]
    names = [p.name for p in default]
    assert len(set(names)) == 7                          # all distinguishable
    custom = perturbations_from_config(
        {"noise_snr_db": [10], "speed_factors": [], "silence": None})
    assert [p.name for p in custom] == ["noise_snr10"]
    # empty mapping means "defaults", NOT "disabled" (only null/false disable)
    empty = perturbations_from_config(
        {"noise_snr_db": [], "speed_factors": [], "silence": {}})
    assert [p.name for p in empty] == ["silence_0.5s_at0.5"]
    off = perturbations_from_config(
        {"noise_snr_db": [], "speed_factors": [], "silence": False})
    assert off == []


# ── plumbing ─────────────────────────────────────────────────────────────────
def test_durations_from_entries():
    d = durations_from_entries([{"id": "a", "frames": 16000},
                                {"id": "b", "frames": 8000}])
    assert d == {"a": 1.0, "b": 0.5}


def test_assert_disjoint():
    assert_disjoint({"a", "b"}, {"c"})                   # no raise
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint({"a", "b"}, {"b", "c"})


def test_recording_loader_accumulates():
    batches = [{"ids": ["a", "b"]}, {"ids": ["c"]}]
    rl = RecordingLoader(batches)
    assert list(rl) == batches and rl.seen == {"a", "b", "c"}
    list(rl)                                             # re-iteration accumulates
    assert rl.seen == {"a", "b", "c"}


def test_list_epoch_checkpoints(tmp_path):
    for name in ("epoch0000.pt", "epoch0002.pt", "epoch0010.pt", "latest.pt",
                 "valid.loss.ave.pt", "epochXX.pt"):
        (tmp_path / name).touch()
    out = list_epoch_checkpoints(tmp_path)
    assert [e for e, _ in out] == [0, 2, 10]             # numeric, non-epochs skipped
    assert all(p.name.startswith("epoch") for _, p in out)
    with pytest.raises(FileNotFoundError, match="keep_all_checkpoints"):
        list_epoch_checkpoints(tmp_path / "empty")


def test_matched_deltas_signs_and_oracle():
    assert matched_deltas([0.10, 0.35], [0.11, 0.34]) == (
        [pytest.approx(-0.01), pytest.approx(0.01)])
    rng = np.random.default_rng(0)
    for _ in range(200):                                 # oracle: len == greedy hits
        pred = sorted(rng.uniform(0, 3, rng.integers(0, 12)))
        true = sorted(rng.uniform(0, 3, rng.integers(0, 12)))
        hits, _, _ = match_boundaries(pred, true, 0.05)
        deltas = matched_deltas(pred, true, 0.05)
        assert len(deltas) == hits
        assert all(abs(d) <= 0.05 + 1e-9 for d in deltas)


# ── boundary report (fake encoder) ───────────────────────────────────────────
class _BEnc:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, feats, lens):
        B = feats.shape[0]
        bounds = [(torch.ones(B, r.shape[0]), r.unsqueeze(0).expand(B, -1))
                  for r in self.rows]
        return SimpleNamespace(boundaries=bounds, lengths=lens)


def _brec():
    return {"id": "u1", "words": [["a", 0.0, 0.10], ["b", 0.10, 0.19]],
            "phones": [["DH", 0.0, 0.10], ["K", 0.10, 0.19]]}


def _bbatch(ids=("u1",), L=6):
    n = len(ids)
    return {"feats": torch.zeros(n, 4 * L, 80),
            "feat_lens": torch.tensor([L] * n), "ids": list(ids)}


def test_boundary_report_hand_case():
    b = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])     # frames 2, 4 -> 0.1025, 0.1825
    rep, bounds = boundary_report(_BEnc([b]), [_bbatch()], {"u1": _brec()},
                                  {"u1": 0.30}, device="cpu", baseline_trials=3)
    for tier in ("words", "phones"):                     # edges 0.10, 0.19: both hit
        m = rep["stage0"][tier]
        assert m["f1"] == 1.0 and m["n_utts"] == 1
        assert "_per_utt" not in m                       # stripped after baseline
        assert 0.0 <= m["random_baseline"]["f1"] <= 1.0
    assert bounds[0]["u1"] == [pytest.approx(0.1025), pytest.approx(0.1825)]
    b2 = torch.tensor([1.0, 1.0])                        # over stage-1 kept frames
    rep2, _ = boundary_report(_BEnc([b, b2]), [_bbatch()], {"u1": _brec()},
                              {"u1": 0.30}, device="cpu", baseline_trials=2)
    assert set(rep2) == {"stage0", "stage1"}


def test_boundary_report_guards():
    b = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    al = {"u1": _brec(), "uZ": {**_brec(), "id": "uZ"}}
    with pytest.raises(ValueError, match="no boundaries"):     # coverage mandate
        boundary_report(_BEnc([b]), [_bbatch()], al, {"u1": 0.3, "uZ": 0.3},
                        device="cpu")
    rep, _ = boundary_report(_BEnc([b]), [_bbatch()], al, {"u1": 0.3, "uZ": 0.3},
                             device="cpu", require_coverage=False,
                             baseline_trials=2)
    assert rep["stage0"]["words"]["missing_boundaries"] == ["uZ"]
    with pytest.raises(ValueError, match="duration"):          # dart-board mandate
        boundary_report(_BEnc([b]), [_bbatch()], {"u1": _brec()}, {}, device="cpu")
    with pytest.raises(ValueError, match="empty loader"):
        boundary_report(_BEnc([b]), [], {"u1": _brec()}, {"u1": 0.3}, device="cpu")


def test_boundary_report_durations_guard_unions_stages(monkeypatch):
    """A duck-typed encoder whose stages cover different utterance sets must
    not bypass the durations guard (scored set = union over ALL stages)."""
    import dcasr.interp.driver as drv
    fake = {0: {"u1": [0.1025]}, 1: {"u1": [0.1025], "u2": [0.5]}}
    monkeypatch.setattr(drv, "collect_boundaries", lambda e, l, d: fake)
    al = {"u1": _brec(), "u2": {**_brec(), "id": "u2"}}
    with pytest.raises(ValueError, match="duration"):
        drv.boundary_report(None, [object()], al, {"u1": 0.3}, device="cpu")


# ── probe report (fake encoder) ──────────────────────────────────────────────
class _PEnc:
    """features encode frame index; chunk z encodes chunk index (d dims)."""

    def __init__(self, b_row, d=4):
        self.b_row = b_row
        self.d = d

    def __call__(self, feats, lens):
        B, L = feats.shape[0], int(lens.max())
        f = torch.arange(L).float().view(1, L, 1).expand(B, L, self.d)
        b = self.b_row.unsqueeze(0).expand(B, -1)[:, :L]
        M = len(chunk_spans(self.b_row, L))
        z = torch.arange(M).float().view(1, M, 1).expand(B, M, self.d)
        return SimpleNamespace(features=f, lengths=lens,
                               boundaries=[(b, b)], chunk_embeddings=[z])


def _prec(uid):
    # frame centers 0.0425..0.2425: words the(0,1) cat(2,3) sat(4,5)
    return {"id": uid,
            "words": [["the", 0.0, 0.1225], ["cat", 0.1225, 0.2025],
                      ["sat", 0.2025, 0.2825]],
            "phones": [["DH", 0.0, 0.1225], ["K", 0.1225, 0.2825]]}


def test_probe_report_end_to_end():
    b = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    enc = _PEnc(b)
    out = probe_report(enc, [_bbatch(("t1",))], [_bbatch(("e1",))],
                       {"t1": _prec("t1")}, {"e1": _prec("e1")}, device="cpu",
                       n_stages=1, top_k_words=3, max_iter=200)
    assert set(out) == {"frames", "chunks_s0"}
    fr = out["frames"]
    # DH on frames 0-1 (features 0,1), K on 2-5 (2..5): linearly separable
    assert fr["phone_id"]["accuracy"] == 1.0
    assert fr["phone_class"]["n_classes"] == 2           # fricative vs stop
    assert fr["word_id"]["accuracy"] == 1.0
    assert fr["word_id"]["train_kept_fraction"] == 1.0   # top-3 of 3 words
    assert fr["word_id"]["test_kept_fraction"] == 1.0
    assert fr["word_id"]["top_k"] == 3
    assert fr["phone_id"]["n_collected_train"] == 6
    ch = out["chunks_s0"]                                # spans (0,3) DH-maj, (3,6) K
    assert ch["phone_id"]["n_collected_train"] == 2
    # top-2 words: 2-of-3 classes kept (ties resolve first-seen) -> 4/6 frames
    out1 = probe_report(enc, [_bbatch(("t1",))], [_bbatch(("e1",))],
                        {"t1": _prec("t1")}, {"e1": _prec("e1")}, device="cpu",
                        n_stages=1, levels=("frames",), top_k_words=2)
    w = out1["frames"]["word_id"]
    assert w["train_kept_fraction"] == pytest.approx(4 / 6)
    assert w["test_kept_fraction"] == pytest.approx(4 / 6)
    assert w["n_classes"] == 2
    with pytest.raises(ValueError, match="unknown probe levels"):
        probe_report(enc, [_bbatch(("t1",))], [_bbatch(("e1",))],
                     {"t1": _prec("t1")}, {"e1": _prec("e1")}, device="cpu",
                     n_stages=1, levels=("frame",))


def test_probe_report_contamination_raises():
    """The #11 mandate: shared utterances between the consumed loaders abort."""
    b = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="overlap"):
        probe_report(_PEnc(b), [_bbatch(("t1", "x9"))], [_bbatch(("x9",))],
                     {"t1": _prec("t1"), "x9": _prec("x9")},
                     {"x9": _prec("x9")}, device="cpu", n_stages=1)


# ── robustness scoring (pure hand cases) ─────────────────────────────────────
def _ral():
    return {"u1": {"id": "u1", "words": [["a", 0.0, 0.5], ["b", 0.5, 1.0]],
                   "phones": [["AA", 0.0, 0.5], ["B", 0.5, 1.0]]}}


def test_score_perturbation_identity_noise():
    clean = {0: {"u1": [0.5, 1.0]}}
    out = score_perturbation(NoisePerturbation(10), clean, clean, _ral(),
                             {"u1": 2.0}, baseline_trials=2)
    s = out["stage0"]
    assert s["words"]["f1"] == 1.0                       # edges 0.5, 1.0 both hit
    assert s["consistency"]["f1"] == 1.0                 # boundaries did not move
    assert s["consistency"]["mean_abs_shift_s"] == 0.0
    assert "random_baseline" in s["words"]


def test_score_perturbation_speed_and_silence():
    clean = {0: {"u1": [0.5, 1.0]}}
    # speed 2x: true edges + surviving boundaries live at t/2
    out = score_perturbation(SpeedPerturbation(2.0), clean, {0: {"u1": [0.25, 0.6]}},
                             _ral(), {"u1": 2.0}, baseline_trials=2)
    s = out["stage0"]
    assert s["words"]["n_hit"] == 1                      # 0.25 hits, 0.6 misses
    assert s["consistency"]["n_hit"] == 1                # clean -> [0.25, 0.5]
    # silence 0.5s at T0=1.0 (at_frac 0.5 of 2.0s): clean -> [0.5, 1.5]
    pert = SilencePerturbation(0.5, at_frac=0.5)
    out = score_perturbation(pert, clean, {0: {"u1": [0.5, 1.2, 1.5]}}, _ral(),
                             {"u1": 2.0}, baseline_trials=2)
    s = out["stage0"]
    c = s["consistency"]
    assert (c["n_hit"], c["n_pred"], c["n_true"]) == (2, 3, 2)
    assert s["inserted_window"]["n_in_window"] == 1      # 1.2 in (1.0, 1.5) interior
    assert s["inserted_window"]["n_total"] == 3
    assert s["inserted_window"]["window_rate_per_s"] == pytest.approx(1 / 0.5)
    with pytest.raises(ValueError, match="no shared"):
        score_perturbation(pert, {0: {}}, {0: {"u1": [0.5]}}, _ral(), {"u1": 2.0})


def test_robustness_report_dispatch():
    clean = {0: {"u1": [0.5, 1.0]}}
    perts = [NoisePerturbation(10), SpeedPerturbation(2.0)]
    seen = []

    def collect(p):
        seen.append(p.name)
        return ({0: {"u1": [0.5, 1.0]}} if p.kind == "noise"
                else {0: {"u1": [0.25, 0.5]}})

    out = robustness_report(perts, collect, clean, _ral(), {"u1": 2.0},
                            baseline_trials=2)
    assert set(out) == {"noise_snr10", "speed_2"} and seen == ["noise_snr10",
                                                               "speed_2"]
    assert out["speed_2"]["stage0"]["consistency"]["f1"] == 1.0
    with pytest.raises(ValueError, match="duplicate perturbation names"):
        robustness_report([NoisePerturbation(10), NoisePerturbation(10)],
                          collect, clean, _ral(), {"u1": 2.0})


# ── flatten + emergence ──────────────────────────────────────────────────────
def test_flatten_metrics_selects_curve_keys():
    nested = {"boundaries": {"stage0": {"words": {
        "f1": 0.5, "n_utts": 3, "missing_boundaries": [],
        "random_baseline": {"f1": 0.1}}}},
        "probes": {"frames": {"phone_id": {"accuracy": 0.7, "n_iter": 12}}}}
    flat = flatten_metrics(nested)
    assert flat == {"interp/boundaries/stage0/words/f1": 0.5,
                    "interp/boundaries/stage0/words/random_baseline/f1": 0.1,
                    "interp/probes/frames/phone_id/accuracy": 0.7}


class _MLog:
    def __init__(self):
        self.calls = []

    def log_scalars(self, values, step, **kw):
        self.calls.append((step, dict(values)))


def test_emergence_report_over_fake_checkpoints():
    good = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # hits both edges
    bad = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])   # frame 5 = 0.2225: miss
    enc = _BEnc([bad])
    model = SimpleNamespace(encoder=enc)
    weights = {"ck0": [bad], "ck5": [good]}

    def load_fn(m, path):
        m.encoder.rows = weights[Path(path).name]

    mlog = _MLog()
    rows = emergence_report(model, [(0, Path("ck0")), (5, Path("ck5"))],
                            [_bbatch()], {"u1": _brec()}, {"u1": 0.30},
                            device="cpu", baseline_trials=2, load_fn=load_fn,
                            probe_fn=lambda e: {"frames": {"phone_id":
                                                           {"accuracy": 0.7}}},
                            mlogger=mlog)
    assert [r["epoch"] for r in rows] == [0, 5]
    f1s = [r["boundaries"]["stage0"]["words"]["f1"] for r in rows]
    assert f1s[0] < 1.0 and f1s[1] == 1.0               # structure "emerges"
    assert all(r["probes"]["frames"]["phone_id"]["accuracy"] == 0.7 for r in rows)
    assert [c[0] for c in mlog.calls] == [0, 5]
    assert "interp/boundaries/stage0/words/f1" in mlog.calls[0][1]
    assert "interp/probes/frames/phone_id/accuracy" in mlog.calls[0][1]


# ── perturbed dataset (real frontend, tiny synthetic audio) ──────────────────
class _FakeTok:
    pad_id = 3

    def encode(self, text):
        return [1, 2]


def _write_audio(tmp_path, n=8000):
    t = np.arange(n) / 16000.0
    wave = 0.1 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "a.flac"
    sf.write(path, wave, 16000)
    return [{"id": "a-1", "audio": str(path), "text": "HI", "frames": n}]


def test_perturbed_dataset_silence_and_noise(tmp_path):
    entries = _write_audio(tmp_path)
    frontend = LogMelFrontend()
    clean = PerturbedDataset(entries, frontend, _FakeTok(), None,
                             Perturbation(), seed=1)
    item = clean[0]
    assert item["id"] == "a-1" and item["feats"].shape[0] == feat_frames(8000)
    assert clean.lengths == [feat_frames(8000)]
    sil = PerturbedDataset(entries, frontend, _FakeTok(), None,
                           SilencePerturbation(0.25, at_frac=0.5), seed=1)
    assert sil.lengths == [feat_frames(8000 + 4000)]
    assert sil[0]["feats"].shape[0] == feat_frames(12000)
    noisy = PerturbedDataset(entries, frontend, _FakeTok(), None,
                             NoisePerturbation(5.0), seed=1)
    nf = noisy[0]["feats"]
    assert nf.shape == item["feats"].shape               # same geometry
    assert not torch.allclose(nf, item["feats"])         # but perturbed content
    assert torch.equal(noisy[0]["feats"], nf)            # deterministic per utt
