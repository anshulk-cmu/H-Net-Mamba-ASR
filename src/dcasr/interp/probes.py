"""Linear probes on frozen representations (plan #11 / §6.4): phone id, phone
class, word id — per stage, vs majority/chance baselines.

A probe is a plain multinomial logistic regression (sklearn, lazy-imported): too
weak to compute anything itself, so probe accuracy measures what is LINEARLY
readable from the representation. Labels come from the MFA ground truth (#9):
a 25 Hz frame gets the phone/word whose interval contains its center time
(boundary_align's verified 0.04*i + 0.0425 s geometry); a chunk gets the
majority label over its fine-frame span (chunk j spans kept frame j up to the
next boundary). Stress digits are collapsed (AH0 -> AH); 'spn' (OOV speech) and
unlabeled (silence) positions are excluded; the word probe is restricted to the
top-K most frequent words of the probe TRAINING set (coverage reported).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Mapping, Sequence

import torch

from dcasr.interp.boundary_align import BOUNDARY_OFFSET_S, FRAME_PERIOD_S
from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

FRAME0_CENTER_S = BOUNDARY_OFFSET_S + FRAME_PERIOD_S / 2      # 0.0425

_PHONE_CLASSES = {
    **{p: "vowel" for p in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                            "IH", "IY", "OW", "OY", "UH", "UW")},
    **{p: "stop" for p in ("B", "D", "G", "K", "P", "T")},
    **{p: "affricate" for p in ("CH", "JH")},
    **{p: "fricative" for p in ("DH", "F", "HH", "S", "SH", "TH", "V", "Z", "ZH")},
    **{p: "nasal" for p in ("M", "N", "NG")},
    **{p: "liquid" for p in ("L", "R")},
    **{p: "glide" for p in ("W", "Y")},
}
EXCLUDED_PHONES = {"spn"}                         # OOV pseudo-phone: not a phone class


def collapse_stress(phone: str) -> str:
    """MFA ARPA phones carry stress digits (AH0/AH1/AH2 -> AH)."""
    return phone.rstrip("012")


def phone_class(phone: str) -> str | None:
    """ARPABET manner class, None for excluded/unknown labels."""
    return _PHONE_CLASSES.get(collapse_stress(phone))


def to_classes(labels: Sequence[str]) -> list[str]:
    """Phone labels -> manner-class labels (raises on unknown phones — a label
    reaching here that has no class is a wiring bug, not data)."""
    out = []
    for lab in labels:
        c = phone_class(lab)
        if c is None:
            raise ValueError(f"phone {lab!r} has no manner class")
        out.append(c)
    return out


def frame_time(i: int) -> float:
    return FRAME_PERIOD_S * i + FRAME0_CENTER_S


def frame_labels(record: Mapping, n_frames: int, tier: str) -> list[str | None]:
    """Per 25 Hz frame: the unit whose interval contains the frame CENTER time
    (None where nothing does — silence/padding). Phones are stress-collapsed;
    excluded phones stay None."""
    units = record[tier]
    out: list[str | None] = [None] * int(n_frames)
    k = 0
    for i in range(int(n_frames)):
        t = frame_time(i)
        while k < len(units) and float(units[k][2]) <= t:
            k += 1
        if k < len(units) and float(units[k][1]) <= t < float(units[k][2]):
            label = str(units[k][0])
            if tier == "phones":
                if label in EXCLUDED_PHONES:
                    continue
                label = collapse_stress(label)
            out[i] = label
    return out


def chunk_spans(b_row: Sequence[float], length: int) -> list[tuple[int, int]]:
    """Boundary vector -> [(start_frame, end_frame_exclusive)] per chunk, in
    chunk order. Matches the model's membership = clamp(cumsum(b)-1, 0): frames
    BEFORE the first boundary merge INTO chunk 0 (the router forces b[0]=1 in
    production, so this only matters for duck-typed encoders)."""
    if int(length) <= 0:
        return []
    starts = [i for i in range(int(length)) if float(b_row[i]) >= 0.5]
    if not starts:
        starts = [0]
    elif starts[0] != 0:
        starts[0] = 0                             # pre-boundary frames join chunk 0
    ends = starts[1:] + [int(length)]
    return list(zip(starts, ends))


def majority_label(labels: Sequence[str | None]) -> str | None:
    """Most common non-None label of a span (ties -> first-seen); None if empty."""
    counts = Counter(l for l in labels if l is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


@torch.no_grad()
def collect_probe_data(model_encoder, loader, alignments: Mapping[str, Mapping],
                       tier: str, device, *, level: str = "frames",
                       stage: int = 0) -> tuple[list, list]:
    """Run the encoder; return (vectors, labels) for labeled positions only.

    level='frames': the fine-rate output per frame. level='chunks': stage-N chunk
    embeddings, labeled by majority over the chunk's fine-frame span (stage 1 =
    Type B second level, spans mapped through stage-0 chunks).

    CONTRACT: probe-train and probe-test loaders must cover DISJOINT utterance
    sets — contamination silently inflates probe accuracy (measured +0.77 on a
    random encoder). The driver enforces this; the returned (X, y) carry no
    utterance identity."""
    if level not in ("frames", "chunks"):
        raise ValueError(f"level must be 'frames' or 'chunks', got {level!r}")
    X, y = [], []
    for batch in loader:
        enc = model_encoder(batch["feats"].to(device), batch["feat_lens"].to(device))
        for bi, uid in enumerate(batch["ids"]):
            if uid not in alignments:
                continue
            L = int(enc.lengths[bi])
            labels = frame_labels(alignments[uid], L, tier)
            if level == "frames":
                feats = enc.features[bi, :L].detach().float().cpu()
                for i, lab in enumerate(labels):
                    if lab is not None:
                        X.append(feats[i].numpy())
                        y.append(lab)
                continue
            b0 = enc.boundaries[0][1][bi].detach().float().cpu()
            spans = chunk_spans(b0, L)
            if stage == 0:
                z = enc.chunk_embeddings[0][bi].detach().float().cpu()
            else:
                if len(enc.chunk_embeddings) < 2:
                    raise ValueError("stage 1 requested but encoder has one stage")
                z = enc.chunk_embeddings[1][bi].detach().float().cpu()
                b1 = enc.boundaries[1][1][bi].detach().float().cpu()
                s2 = chunk_spans(b1, len(spans))  # spans over stage-0 CHUNK indices
                spans = [(spans[a][0], spans[b - 1][1]) for a, b in s2 if b <= len(spans)]
            for j, (a, b) in enumerate(spans):
                if j >= z.shape[0]:
                    break
                lab = majority_label(labels[a:b])
                if lab is not None:
                    X.append(z[j].numpy())
                    y.append(lab)
    logger.info("probe data: level=%s tier=%s stage=%d -> %d labeled samples",
                level, tier, stage, len(X))
    return X, y


def subsample(X: Sequence, y: Sequence, cap: int, seed: int = 1):
    """Seeded uniform subsample to at most `cap` examples (keeps pairing)."""
    if len(X) <= cap:
        return list(X), list(y)
    idx = list(range(len(X)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:cap])
    return [X[i] for i in idx], [y[i] for i in idx]


def top_k_filter(X: Sequence, y: Sequence, k: int):
    """Restrict to the top-k most frequent labels (the word-id probe convention).
    Returns (X, y, coverage) where coverage = kept fraction. Call on the probe
    TRAINING set only — the keep-set must come from train frequencies; the test
    side is restricted automatically via train_probe's unseen-class drop."""
    counts = Counter(y)
    keep = {lab for lab, _ in counts.most_common(k)}
    pairs = [(x, lab) for x, lab in zip(X, y) if lab in keep]
    coverage = len(pairs) / max(1, len(y))
    return [p[0] for p in pairs], [p[1] for p in pairs], coverage


def train_probe(X_train, y_train, X_test, y_test, *, max_iter: int = 200,
                C: float = 1.0, seed: int = 1) -> dict:
    """Multinomial logistic regression; accuracy + balanced accuracy vs
    majority/chance baselines. Test items whose class was never seen in training
    are excluded and counted in n_test_dropped_unseen — REPORT that (and any
    top-k kept fraction) alongside accuracy; on skewed labels the headline can
    describe under half the frames otherwise. seed only matters for non-lbfgs
    solvers (lbfgs is deterministic)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    train_classes = set(y_train)
    if len(train_classes) < 2:
        raise ValueError(f"probe needs >= 2 training classes, got {len(train_classes)}")
    kept = [i for i, lab in enumerate(y_test) if lab in train_classes]
    dropped_test = len(y_test) - len(kept)
    X_test = [X_test[i] for i in kept]
    y_test = [y_test[i] for i in kept]
    if not y_train or not y_test:
        raise ValueError("empty probe train or test set")
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed)
    clf.fit(np.asarray(X_train), y_train)
    pred = clf.predict(np.asarray(X_test))
    acc = float(np.mean(pred == np.asarray(y_test)))
    majority = Counter(y_train).most_common(1)[0][0]
    maj_acc = sum(lab == majority for lab in y_test) / len(y_test)
    return {"accuracy": acc,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "majority_baseline": maj_acc,
            "chance": 1.0 / len(train_classes), "n_classes": len(train_classes),
            "n_train": len(y_train), "n_test": len(y_test),
            "n_test_dropped_unseen": dropped_test,
            "n_iter": int(np.max(clf.n_iter_))}
