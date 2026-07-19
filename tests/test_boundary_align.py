"""Unit tests for boundary scoring (src/dcasr/interp/boundary_align.py). CPU-only:
hand-computed matching/PRF/R-value cases, timing geometry, stage-2 mapping,
random-baseline properties, and a duck-typed fake encoder for collect_boundaries."""
from types import SimpleNamespace

import pytest
import torch

from dcasr.interp.boundary_align import (BOUNDARY_OFFSET_S, FRAME_PERIOD_S,
                                         aggregate, collect_boundaries,
                                         frame_boundary_times, match_boundaries,
                                         prf, r_value, random_baseline,
                                         score_utterances, stage2_boundary_times,
                                         true_edges)


# ── timing geometry ──────────────────────────────────────────────────────────
def test_frame_boundary_times_and_drop_first():
    b = [1, 0, 1, 1, 0, 1]
    t = frame_boundary_times(b, 6)
    assert t == [2 * 0.04 + 0.0225, 3 * 0.04 + 0.0225, 5 * 0.04 + 0.0225]
    t0 = frame_boundary_times(b, 6, drop_first=False)
    assert t0[0] == pytest.approx(0.0225) and len(t0) == 4
    assert frame_boundary_times(b, 4) == [2 * 0.04 + 0.0225, 3 * 0.04 + 0.0225]
    assert frame_boundary_times([0, 0, 0], 3) == []


def test_timing_constants_match_conv_geometry():
    # 100 Hz frame j center = 0.01j + 0.0125; two k3/s2 convs center on 4i+3
    for i in (0, 1, 7):
        center_100hz_frame = 4 * i + 3
        center = 0.01 * center_100hz_frame + 0.0125
        assert center == pytest.approx(FRAME_PERIOD_S * i + 0.0425)
    assert BOUNDARY_OFFSET_S == pytest.approx(0.0425 - FRAME_PERIOD_S / 2)


def test_stage2_mapping():
    b1 = [1, 0, 1, 0, 0, 1, 0]                          # kept fine frames: 0, 2, 5
    b2 = [1, 1, 1]                                      # all three stage-2 frames bound
    t = stage2_boundary_times(b1, b2, 7)
    assert t == [2 * 0.04 + 0.0225, 5 * 0.04 + 0.0225]  # frame 0 dropped (structural)
    t = stage2_boundary_times(b1, [1, 0, 1], 7)
    assert t == [5 * 0.04 + 0.0225]
    assert stage2_boundary_times(b1, [1, 1, 1, 1, 1], 7) == [2 * 0.04 + 0.0225,
                                                             5 * 0.04 + 0.0225]


# ── true edges ───────────────────────────────────────────────────────────────
def test_true_edges_dedupe_pause_and_initial():
    words = [["the", 0.0, 0.4], ["cat", 0.4, 1.0], ["sat", 1.2, 1.8]]  # pause 1.0-1.2
    e = true_edges(words)
    assert e == [0.4, 1.0, 1.2, 1.8]                    # t=0 dropped; abutting deduped
    assert true_edges([["a", 0.0, 0.02]]) == []         # everything below min_t


# ── matching + metrics ───────────────────────────────────────────────────────
def test_match_boundaries_one_to_one():
    assert match_boundaries([0.41, 1.11, 1.60], [0.40, 1.10, 2.00]) == (2, 3, 3)
    assert match_boundaries([0.41, 1.13, 1.60], [0.40, 1.10, 2.00]) == (1, 3, 3)  # 30ms > tol
    assert match_boundaries([1.0], [0.99, 1.01], tol=0.02) == (1, 1, 2)   # one hit only
    assert match_boundaries([0.99, 1.01], [1.0], tol=0.02) == (1, 2, 1)   # no double-claim
    assert match_boundaries([1.02], [1.0], tol=0.02) == (1, 1, 1)         # inclusive edge
    assert match_boundaries([1.021], [1.0], tol=0.02) == (0, 1, 1)
    assert match_boundaries([], [1.0]) == (0, 0, 1)
    assert match_boundaries([1.0], []) == (0, 1, 0)


def test_prf_and_aggregate_hand_computed():
    m = prf(2, 3, 4)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    agg = aggregate([(2, 3, 4), (1, 1, 2)])             # micro: 3/4 pred, 3/6 true
    assert agg["precision"] == 0.75 and agg["recall"] == 0.5
    assert agg["over_seg"] == pytest.approx(4 / 6 - 1)
    assert agg["n_utts"] == 2
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_r_value_identities():
    assert r_value(1.0, 0.0) == pytest.approx(1.0)      # perfect segmentation
    sprayed = r_value(1.0, 3.0)                         # recall 1 by spraying 4x
    assert sprayed < 0.2                                # heavily punished
    assert r_value(0.5, 0.0) > r_value(0.5, 1.0)        # over-seg strictly worse
    assert r_value(0.0, -1.0) < 0.5                     # no boundaries at all


# ── random baseline ──────────────────────────────────────────────────────────
def test_random_baseline_properties():
    per_utt = [{"n_pred": 5, "true": [0.5, 1.0, 1.5, 2.0], "duration": 3.0}
               for _ in range(30)]
    b1 = random_baseline(per_utt, seed=3, trials=5)
    b2 = random_baseline(per_utt, seed=3, trials=5)
    b3 = random_baseline(per_utt, seed=4, trials=5)
    assert b1 == b2 and b1 != b3                        # deterministic, seed-sensitive
    # 5 darts on 3.0s vs 4 edges with ±20ms windows: expect low but nonzero recall
    assert 0.0 < b1["recall"] < 0.5
    zero = random_baseline([{"n_pred": 0, "true": [1.0], "duration": 2.0}])
    assert zero["recall"] == 0.0 and zero["precision"] == 0.0


# ── end-to-end scoring ───────────────────────────────────────────────────────
def _alignment(uid="u1"):
    return {"id": uid,
            "words": [["the", 0.0, 0.4], ["cat", 0.4, 1.0]],
            "phones": [["DH", 0.0, 0.2], ["AH0", 0.2, 0.4], ["K", 0.4, 0.7],
                       ["AE1", 0.7, 0.85], ["T", 0.85, 1.0]]}


def test_score_utterances_tiers_and_guards():
    bounds = {"u1": [0.41, 0.86]}
    al = {"u1": _alignment()}
    w = score_utterances(bounds, al, "words")
    assert (w["n_hit"], w["n_pred"], w["n_true"]) == (1, 2, 2)   # word edges 0.4, 1.0
    p = score_utterances(bounds, al, "phones")
    assert p["n_true"] == 5                             # 0.2, 0.4, 0.7, 0.85, 1.0 (final
    assert p["n_hit"] == 2                              # speech->silence edge kept)
    assert p["tier"] == "phones" and p["_per_utt"][0]["duration"] == 1.0
    with pytest.raises(ValueError, match="tier"):
        score_utterances(bounds, al, "syllables")
    with pytest.raises(ValueError, match="overlap"):
        score_utterances({"zz": [0.1]}, al, "words")
    out = score_utterances({"u1": [0.41], "u9": [0.5]}, al, "words")
    assert out["missing_alignments"] == ["u9"]          # reported, not silently scored


def test_baseline_beaten_by_oracle_boundaries():
    al = {f"u{i}": _alignment(f"u{i}") for i in range(20)}
    oracle = {f"u{i}": [0.4, 1.0] for i in range(20)}   # exactly the word edges
    s = score_utterances(oracle, al, "words")
    assert s["f1"] == 1.0 and s["r_value"] == pytest.approx(1.0)
    base = random_baseline(s["_per_utt"], seed=1, trials=5)
    assert base["f1"] < 0.5 < s["f1"]                   # the gap the paper claims


# ── collect_boundaries (duck-typed encoder) ──────────────────────────────────
class _FakeEnc:
    def __init__(self, stages):
        self.stages = stages

    def __call__(self, feats, lens):
        B = feats.shape[0]
        bounds = [(torch.ones(B, b.shape[0]), b.unsqueeze(0).expand(B, -1))
                  for b in self.stages]
        return SimpleNamespace(boundaries=bounds, lengths=lens)


def test_collect_boundaries_single_and_two_stage():
    b1 = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    batch = {"feats": torch.zeros(2, 24, 80), "feat_lens": torch.tensor([6, 4]),
             "ids": ["a", "b"]}
    out = collect_boundaries(_FakeEnc([b1]), [batch], "cpu")
    assert out[0]["a"] == [2 * 0.04 + 0.0225, 4 * 0.04 + 0.0225]
    assert out[0]["b"] == [2 * 0.04 + 0.0225]           # length-masked at 4
    b2 = torch.tensor([1.0, 1.0, 0.0])
    out = collect_boundaries(_FakeEnc([b1, b2]), [batch], "cpu")
    assert 1 in out and out[1]["a"] == [2 * 0.04 + 0.0225]   # kept frames 0,2,4 → j=1→2


# ── fixes from the adversarial verification (wf_c2711701) ────────────────────
def test_missing_boundaries_reported_and_durations_board():
    al = {f"u{i}": _alignment(f"u{i}") for i in range(3)}
    out = score_utterances({"u0": [0.41]}, al, "words")
    assert out["missing_boundaries"] == ["u1", "u2"]    # no longer a silent drop
    # true-audio dart board: longer board -> sparser darts -> lower chance recall
    b_short = random_baseline(score_utterances({"u0": [0.4, 1.0]}, al, "words",
                                               )["_per_utt"], seed=0, trials=200)
    b_long = random_baseline(score_utterances({"u0": [0.4, 1.0]}, al, "words",
                                              durations={"u0": 3.0})["_per_utt"],
                             seed=0, trials=200)
    assert b_long["recall"] < b_short["recall"]         # 3.0s board vs 1.0s last-unit-end


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (mamba_ssm)")


@gpu
def test_timing_constants_empirical_not_circular():
    """Support probe through the REAL frontend + conv stack: 25 Hz frame i must
    respond to samples inside [640i, 640i+1360) and to nothing outside — pinning
    center 640i+680 samples = 0.04i + 0.0425 s. A frontend center-convention
    drift (e.g. center=True, -12.5 ms shift) fails this."""
    from dcasr.data.features import LogMelFrontend
    from dcasr.models.encoder import ConvSubsampling4
    torch.manual_seed(0)
    frontend = LogMelFrontend()
    conv = ConvSubsampling4(80, 32).eval()
    n = 16000 * 2
    base = torch.randn(1, n) * 0.01
    lens = torch.tensor([n])

    def frames_changed(sample):
        wave = base.clone()
        wave[0, sample] += 50.0
        with torch.no_grad():
            feats0, fl0 = frontend(base, lens)
            feats1, fl1 = frontend(wave, lens)
            f0, _ = conv(feats0, fl0)
            f1, _ = conv(feats1, fl1)
        return (f1 - f0).abs().sum(dim=2)[0] > 1e-4

    for i in (3, 10):
        lo, hi = 640 * i, 640 * i + 1360                 # frame i's receptive field
        assert frames_changed(lo + 100)[i]               # inside -> responds
        assert frames_changed(hi - 100)[i]
        assert not frames_changed(lo - 10)[i]            # outside -> silent
        assert not frames_changed(hi + 10)[i]
    # the boundary instant sits half a period before the empirical center 640i+680
    assert BOUNDARY_OFFSET_S == pytest.approx((640 + 680) / 16000 - 0.04 - 0.02)
