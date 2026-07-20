"""Unit tests for linear probes (src/dcasr/interp/probes.py). CPU-only: label
geometry hand cases, phone-class table completeness, chunk spans/majority,
subsample/top-k determinism, and probe separability on synthetic data."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dcasr.interp.probes import (EXCLUDED_PHONES, _PHONE_CLASSES, chunk_spans,
                                 collapse_stress, collect_probe_data, frame_labels,
                                 frame_time, majority_label, phone_class, subsample,
                                 to_classes, top_k_filter, train_probe)


# ── label geometry ───────────────────────────────────────────────────────────
def test_frame_time_matches_boundary_geometry():
    assert frame_time(0) == pytest.approx(0.0425)
    assert frame_time(10) == pytest.approx(0.4425)


def test_frame_labels_center_containment():
    rec = {"phones": [["DH", 0.0, 0.06], ["AH0", 0.06, 0.12], ["K", 0.20, 0.30]],
           "words": [["the", 0.0, 0.12], ["cat", 0.20, 0.30]]}
    # frame centers: 0.0425, 0.0825, 0.1225, 0.1625, 0.2025, 0.2425, 0.2825
    ph = frame_labels(rec, 7, "phones")
    assert ph == ["DH", "AH", None, None, "K", "K", "K"]      # stress collapsed, gap None
    wd = frame_labels(rec, 7, "words")
    assert wd == ["the", "the", None, None, "cat", "cat", "cat"]
    spn = {"phones": [["spn", 0.0, 0.30]], "words": [["shampooer", 0.0, 0.30]]}
    assert frame_labels(spn, 5, "phones") == [None] * 5       # excluded pseudo-phone
    assert frame_labels(spn, 5, "words")[0] == "shampooer"    # word itself kept


def test_frame_labels_half_open_advance():
    # MFA times sit on a 10 ms grid; frame centers on 42.5+40k ms — never equal.
    # Center 0.0825 with units meeting at 0.08: belongs to the SECOND unit.
    rec = {"phones": [["A", 0.0, 0.08], ["B", 0.08, 0.2]]}
    assert frame_labels(rec, 2, "phones") == ["A", "B"]


def test_collapse_stress_and_phone_classes():
    assert collapse_stress("AH0") == "AH" and collapse_stress("IY1") == "IY"
    assert collapse_stress("NG") == "NG"                       # no digit: untouched
    assert phone_class("AH2") == "vowel" and phone_class("CH") == "affricate"
    assert phone_class("spn") is None
    # the table must cover the FULL stress-collapsed ARPABET MFA emits (39 phones)
    assert len(_PHONE_CLASSES) == 39
    assert to_classes(["AH0", "T", "M"]) == ["vowel", "stop", "nasal"]
    with pytest.raises(ValueError, match="manner class"):
        to_classes(["XX"])


# ── chunk spans + majority ───────────────────────────────────────────────────
def test_chunk_spans():
    assert chunk_spans([1, 0, 0, 1, 0, 1], 6) == [(0, 3), (3, 5), (5, 6)]
    assert chunk_spans([0, 0, 1, 0], 4) == [(0, 4)]            # merges into chunk 0
    # matches membership = clamp(cumsum(b)-1, 0): pre-boundary frames JOIN chunk 0
    assert chunk_spans([0, 1, 0, 1], 4) == [(0, 3), (3, 4)]
    assert chunk_spans([1, 1, 1], 3) == [(0, 1), (1, 2), (2, 3)]
    assert chunk_spans([0, 0], 2) == [(0, 2)]
    assert chunk_spans([], 0) == []


def test_majority_label():
    assert majority_label(["A", "A", "B", None]) == "A"
    assert majority_label([None, None]) is None
    assert majority_label(["B", "A", "A", "B"]) in ("A", "B")  # tie: deterministic pick
    assert majority_label(["B", "A", "A", "B"]) == majority_label(["B", "A", "A", "B"])


# ── subsample + top-k ────────────────────────────────────────────────────────
def test_subsample_deterministic_and_paired():
    X = [np.array([i]) for i in range(100)]
    y = [str(i) for i in range(100)]
    x1, y1 = subsample(X, y, 30, seed=5)
    x2, y2 = subsample(X, y, 30, seed=5)
    x3, _ = subsample(X, y, 30, seed=6)
    assert [int(a[0]) for a in x1] == [int(a[0]) for a in x2] != [int(a[0]) for a in x3]
    assert all(int(a[0]) == int(lab) for a, lab in zip(x1, y1))  # pairing preserved
    xa, ya = subsample(X, y, 200, seed=1)
    assert len(xa) == 100                                        # under cap: all kept


def test_top_k_filter():
    y = ["a"] * 5 + ["b"] * 3 + ["c"] * 1
    X = [np.array([i]) for i in range(9)]
    Xf, yf, cov = top_k_filter(X, y, 2)
    assert set(yf) == {"a", "b"} and len(yf) == 8
    assert cov == pytest.approx(8 / 9)


# ── the probe itself ─────────────────────────────────────────────────────────
def _separable(n=60, d=8, classes=("p", "q", "r"), noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in range(n):
        c = i % len(classes)
        v = np.zeros(d)
        v[c] = 1.0
        X.append(v + rng.normal(0, noise, d))
        y.append(classes[c])
    return X, y


def test_train_probe_separable_vs_baselines():
    Xtr, ytr = _separable(seed=0)
    Xte, yte = _separable(seed=1)
    out = train_probe(Xtr, ytr, Xte, yte)
    assert out["accuracy"] > 0.95                                # linearly readable
    assert out["majority_baseline"] == pytest.approx(1 / 3, abs=0.05)
    assert out["chance"] == pytest.approx(1 / 3)
    assert out["n_classes"] == 3 and out["n_test_dropped_unseen"] == 0


def test_train_probe_unseen_test_class_dropped():
    Xtr, ytr = _separable(classes=("p", "q"))
    Xte, yte = _separable(classes=("p", "q", "zz"))
    out = train_probe(Xtr, ytr, Xte, yte)
    assert out["n_test_dropped_unseen"] == 20                    # zz never trained
    with pytest.raises(ValueError, match="empty probe"):
        train_probe(Xtr, ytr, [], [])


def test_probe_on_noise_is_at_chance():
    rng = np.random.default_rng(0)
    X = [rng.normal(0, 1, 8) for _ in range(300)]
    y = [("a", "b", "c")[i % 3] for i in range(300)]
    out = train_probe(X[:200], y[:200], X[200:], y[200:])
    assert abs(out["accuracy"] - 1 / 3) < 0.2                    # no free structure


# ── collection (duck-typed encoder) ──────────────────────────────────────────
class _FakeEnc:
    """features encode the frame index; chunk z encodes the chunk index."""

    def __init__(self, b_rows, d=4, two_stage=False):
        self.b_rows = b_rows
        self.d = d
        self.two_stage = two_stage

    def __call__(self, feats, lens):
        B, L = feats.shape[0], int(lens.max())
        f = torch.arange(L).float().view(1, L, 1).expand(B, L, self.d)
        b = torch.stack(self.b_rows)[:B, :L]
        M = int(chunk_len := max(len(chunk_spans(r, int(l))) for r, l in zip(b, lens)))
        z = torch.arange(M).float().view(1, M, 1).expand(B, M, self.d)
        bounds = [(b, b)]
        zs = [z]
        if self.two_stage:
            b2 = torch.ones(B, M)
            bounds.append((b2, b2))
            zs.append(z.clone())
        return SimpleNamespace(features=f, lengths=lens, boundaries=bounds,
                               chunk_embeddings=zs)


def _rec():
    return {"phones": [["DH", 0.0, 0.1225], ["K", 0.1225, 0.2825]],
            "words": [["the", 0.0, 0.1225], ["cat", 0.1225, 0.2825]]}


def test_collect_probe_data_frames_and_chunks():
    b = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    batch = {"feats": torch.zeros(1, 24, 80), "feat_lens": torch.tensor([6]),
             "ids": ["u1"]}
    al = {"u1": _rec()}
    enc = _FakeEnc([b])
    X, y = collect_probe_data(enc, [batch], al, "phones", "cpu", level="frames")
    # frames 0,1 -> DH; frames 2..5 centers 0.1225..0.2425 -> K (0.2825 end covers 5)
    assert y == ["DH", "DH", "K", "K", "K", "K"]
    assert [int(v[0]) for v in X] == [0, 1, 2, 3, 4, 5]          # right vectors paired
    Xc, yc = collect_probe_data(enc, [batch], al, "phones", "cpu", level="chunks")
    assert yc == ["DH", "K"]                                     # majority per span
    assert [int(v[0]) for v in Xc] == [0, 1]
    Xw, yw = collect_probe_data(enc, [batch], al, "words", "cpu", level="frames")
    assert yw[0] == "the" and yw[-1] == "cat"
    with pytest.raises(ValueError, match="level"):
        collect_probe_data(enc, [batch], al, "phones", "cpu", level="fram")
    with pytest.raises(ValueError, match="one stage"):
        collect_probe_data(enc, [batch], al, "phones", "cpu", level="chunks", stage=1)


def test_collect_skips_unaligned_utts():
    b = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    batch = {"feats": torch.zeros(2, 24, 80), "feat_lens": torch.tensor([6, 6]),
             "ids": ["u1", "zz"]}
    X, y = collect_probe_data(_FakeEnc([b, b]), [batch], {"u1": _rec()}, "phones",
                              "cpu", level="frames")
    assert len(y) == 6                                           # only u1 contributes


# ── fixes from the adversarial verification (wf_017fc8e0) ────────────────────
def test_chunk_spans_matches_model_membership():
    """Oracle: spans must equal frames grouped by clamp(cumsum(b)-1, 0)."""
    import itertools
    for L in range(1, 7):
        for bits in itertools.product([0, 1], repeat=L):
            memb = []
            c = 0
            for i, b in enumerate(bits):
                c += b
                memb.append(max(c - 1, 0))
            expect = []
            for j in sorted(set(memb)):
                idx = [i for i, m in enumerate(memb) if m == j]
                expect.append((idx[0], idx[-1] + 1))
            assert chunk_spans(list(bits), L) == expect, (bits, L)


def test_probe_majority_differs_from_chance_and_drop_pairing():
    """Skewed labels: majority != chance (swapping the fields would fail); the
    unseen-class drop must keep X/y pairing (accuracy stays perfect)."""
    rng = np.random.default_rng(0)
    slot = {"a": 0, "b": 1, "c": 2, "zz": 3}   # hash() is PYTHONHASHSEED-random
    def sep(labels):
        X = []
        for lab in labels:
            v = np.zeros(6)
            v[slot[lab]] = 1.0
            X.append(v + rng.normal(0, 0.01, 6))
        return X
    ytr = ["a"] * 60 + ["b"] * 30 + ["c"] * 10          # skewed: majority 0.6
    yte = ["a"] * 6 + ["b"] * 3 + ["zz"] * 4 + ["c"] * 1
    out = train_probe(sep(ytr), ytr, sep(yte), yte)
    assert out["majority_baseline"] == pytest.approx(0.6)
    assert out["chance"] == pytest.approx(1 / 3)
    assert out["majority_baseline"] != out["chance"]
    assert out["n_test"] == 10 and out["n_test_dropped_unseen"] == 4
    assert out["accuracy"] > 0.9                        # pairing survived the drop
    assert out["balanced_accuracy"] > 0.9 and out["n_iter"] >= 1
    with pytest.raises(ValueError, match=">= 2 training classes"):
        train_probe(sep(["a"] * 5), ["a"] * 5, sep(["a"]), ["a"])


def test_collect_two_stage_nontrivial_and_mixed_lengths():
    """Type B composition with a non-all-ones b2 + a padded mixed-length batch."""
    b_a = torch.tensor([1.0, 0, 1, 0, 1, 0, 1, 0])       # 4 chunks over 8 frames
    b_b = torch.tensor([1.0, 0, 1, 0, 0, 0, 0, 0])       # shorter row: 2 chunks over 4
    batch = {"feats": torch.zeros(2, 32, 80), "feat_lens": torch.tensor([8, 4]),
             "ids": ["u1", "u2"]}
    al = {"u1": {"phones": [["A", 0.0, 0.16], ["B", 0.16, 0.37]],
                 "words": [["w", 0.0, 0.37]]},
          "u2": {"phones": [["C", 0.0, 0.21]], "words": [["v", 0.0, 0.21]]}}

    class _TwoStage:
        def __call__(self, feats, lens):
            B, L = feats.shape[0], 8
            b1 = torch.stack([b_a, b_b])
            # stage-2 over each row's stage-1 chunks: row0 has 4, row1 has 2
            b2 = torch.tensor([[1.0, 0, 1, 0], [1.0, 1, 0, 0]])
            z1 = torch.arange(4).float().view(1, 4, 1).expand(B, 4, 3)
            z2 = torch.arange(2).float().view(1, 2, 1).expand(B, 2, 3)
            return SimpleNamespace(features=torch.zeros(B, L, 3), lengths=lens,
                                   boundaries=[(b1, b1), (b2, b2)],
                                   chunk_embeddings=[z1, z2])

    X, y = collect_probe_data(_TwoStage(), [batch], al, "phones", "cpu",
                              level="chunks", stage=1)
    # u1: s2 groups stage-0 chunks {0,1} and {2,3} -> fine spans (0,4),(4,8)
    # centers 0.0425..: frames 0-2 -> A/B mix; u2 (len 4): groups {0} {1} -> (0,2),(2,4)
    assert len(y) == 4                                   # 2 composed chunks per row
    assert y[0] in ("A", "B") and len(X[0]) == 3
    # padded row: u2's stage-2 z rows beyond its 2 chunks never sampled
    labels_u2 = y[2:]
    assert all(l == "C" for l in labels_u2)


# ── torch (GPU-capable) probe backend: parity with the sklearn reference ─────
def test_torch_backend_parity_separable_and_noise():
    """Convex objective with unique optimum: both backends must reach the same
    solution — accuracies equal on separable AND noisy data (CPU, fp64)."""
    Xtr, ytr = _separable(seed=0)
    Xte, yte = _separable(seed=1)
    sk = train_probe(Xtr, ytr, Xte, yte)
    th = train_probe(Xtr, ytr, Xte, yte, backend="torch", device="cpu",
                     max_iter=500)
    assert th["backend"] == "torch" and sk["backend"] == "sklearn"
    assert th["accuracy"] == sk["accuracy"] == pytest.approx(1.0, abs=0.02)
    rng = np.random.default_rng(3)
    Xn = [rng.normal(0, 1, 8) for _ in range(300)]
    yn = [("a", "b", "c")[i % 3] for i in range(300)]
    skn = train_probe(Xn[:200], yn[:200], Xn[200:], yn[200:])
    thn = train_probe(Xn[:200], yn[:200], Xn[200:], yn[200:], backend="torch",
                      device="cpu", max_iter=500)
    assert abs(thn["accuracy"] - skn["accuracy"]) <= 0.02   # same optimum
    assert thn["n_iter"] >= 1


def test_torch_backend_skewed_and_unseen_drop():
    """The shared pre-fit path (unseen-class drop, majority/chance) is backend-
    independent; skewed-label parity pins the fit itself."""
    rng = np.random.default_rng(0)
    slot = {"a": 0, "b": 1, "c": 2, "zz": 3}
    def sep(labels):
        return [np.eye(6)[slot[l]] + rng.normal(0, 0.01, 6) for l in labels]
    ytr = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    yte = ["a"] * 6 + ["b"] * 3 + ["zz"] * 4 + ["c"] * 1
    sk = train_probe(sep(ytr), ytr, sep(yte), yte)
    th = train_probe(sep(ytr), ytr, sep(yte), yte, backend="torch", device="cpu",
                     max_iter=500)
    for k in ("accuracy", "majority_baseline", "chance", "n_classes",
              "n_test", "n_test_dropped_unseen"):
        assert th[k] == sk[k], k
    assert th["balanced_accuracy"] == pytest.approx(sk["balanced_accuracy"])


def test_torch_backend_deterministic_and_guards():
    Xtr, ytr = _separable(seed=0)
    Xte, yte = _separable(seed=1)
    a = train_probe(Xtr, ytr, Xte, yte, backend="torch", device="cpu")
    b = train_probe(Xtr, ytr, Xte, yte, backend="torch", device="cpu")
    assert a == b                                     # zero-init LBFGS: no RNG
    with pytest.raises(ValueError, match="backend"):
        train_probe(Xtr, ytr, Xte, yte, backend="cuml")


def test_torch_backend_two_class_falls_back_to_sklearn():
    """2 classes: sklearn's binary-sigmoid optimum != a 2-column softmax's
    (verified |dP| up to 0.04) — the torch backend must defer to the reference."""
    Xtr, ytr = _separable(classes=("p", "q"), seed=0)
    Xte, yte = _separable(classes=("p", "q"), seed=1)
    out = train_probe(Xtr, ytr, Xte, yte, backend="torch", device="cpu")
    assert out["backend"] == "sklearn"                # auto-fallback recorded
