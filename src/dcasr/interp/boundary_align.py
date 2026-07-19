"""Boundary alignment scoring (plan #10 / §6.4, H4): learned H-Net boundaries vs
MFA phone/word ground truth — precision/recall/F1 within ±20 ms, R-value, and a
matched-count random baseline.

Timing model (derived from the frontend + conv geometry, all center=False):
100 Hz STFT frame j covers samples [160j, 160j+400) -> center 0.01*j + 0.0125 s;
each k=3/s=2 conv output centers on its middle input, twice -> 25 Hz frame i has
center 0.04*i + 0.0425 s. A boundary "at frame i" (chunk starts there) marks the
transition from frame i-1, i.e. the midpoint of their centers: 0.04*i + 0.0225 s.
Frame 0's boundary is structural (p_1 = 1 by construction) and is excluded, as is
the utterance-initial true edge. Matching is greedy one-to-one on sorted times, a
hit iff |t_pred - t_true| <= tol; R-value follows Rasanen et al. (2009). Stage-2
(Type B) boundaries live on stage-1's kept frames and are mapped back through the
stage-1 boundary vector before timing. Pure CPU except collect_boundaries (duck-
typed encoder forward).
"""
from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import torch

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

FRAME_PERIOD_S = 0.04                       # 25 Hz encoder frames
BOUNDARY_OFFSET_S = 0.0225                  # transition instant of frame i (see docstring)
DEFAULT_TOL_S = 0.02                        # plan §6.4: ±20 ms


def frame_boundary_times(b_row: Sequence[float], length: int,
                         drop_first: bool = True) -> list[float]:
    """Binary boundary vector [L] (b_t >= 0.5 = chunk start) -> boundary times (s)."""
    start = 1 if drop_first else 0
    return [i * FRAME_PERIOD_S + BOUNDARY_OFFSET_S
            for i in range(start, int(length)) if float(b_row[i]) >= 0.5]


def stage2_boundary_times(b1_row: Sequence[float], b2_row: Sequence[float],
                          length1: int, drop_first: bool = True) -> list[float]:
    """Type B stage-2 boundaries -> times: stage-2 frame j IS stage-1's j-th kept
    frame, so map j through the positions of 1s in stage-1's boundary vector."""
    kept = [i for i in range(int(length1)) if float(b1_row[i]) >= 0.5]
    start = 1 if drop_first else 0
    return [kept[j] * FRAME_PERIOD_S + BOUNDARY_OFFSET_S
            for j in range(start, min(len(b2_row), len(kept)))
            if float(b2_row[j]) >= 0.5]


def true_edges(units: Sequence[Sequence], min_t: float = 0.03,
               dedupe_tol: float = 1e-4) -> list[float]:
    """Alignment triples [label, start, end] -> sorted internal edge times.
    Starts AND ends kept (a pause makes both real edges), deduped when abutting;
    edges near t=0 dropped (utterance-initial is structural, mirrored on the
    model side by drop_first)."""
    times: list[float] = []
    for _, s, e in units:
        times.extend((float(s), float(e)))
    times.sort()
    out: list[float] = []
    for t in times:
        if t < min_t:
            continue
        if out and t - out[-1] <= dedupe_tol:
            continue
        out.append(t)
    return out


def match_boundaries(pred: Sequence[float], true: Sequence[float],
                     tol: float = DEFAULT_TOL_S) -> tuple[int, int, int]:
    """Greedy one-to-one matching on sorted times -> (n_hit, n_pred, n_true)."""
    pred, true = sorted(pred), sorted(true)
    hits = i = j = 0
    eps = 1e-9                                       # exact-tol hits survive float repr
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol + eps:
            hits += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return hits, len(pred), len(true)


def prf(n_hit: int, n_pred: int, n_true: int) -> dict[str, float]:
    p = n_hit / n_pred if n_pred else 0.0
    r = n_hit / n_true if n_true else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def r_value(recall: float, over_seg: float) -> float:
    """Rasanen et al. 2009 (fractions): 1 at perfect segmentation, penalises
    boundary-spraying that plain recall rewards."""
    r1 = ((1.0 - recall) ** 2 + over_seg ** 2) ** 0.5
    r2 = (-over_seg + recall - 1.0) / (2 ** 0.5)
    return 1.0 - (abs(r1) + abs(r2)) / 2.0


def aggregate(counts: Sequence[tuple[int, int, int]]) -> dict[str, float]:
    """Corpus micro-average over per-utterance (hit, pred, true) counts."""
    h = sum(c[0] for c in counts)
    p = sum(c[1] for c in counts)
    t = sum(c[2] for c in counts)
    out = prf(h, p, t)
    out["over_seg"] = (p / t - 1.0) if t else 0.0
    out["r_value"] = r_value(out["recall"], out["over_seg"])
    out.update(n_hit=h, n_pred=p, n_true=t, n_utts=len(counts))
    return out


def random_baseline(per_utt: Sequence[Mapping[str, Any]], tol: float = DEFAULT_TOL_S,
                    seed: int = 1, trials: int = 10) -> dict[str, float]:
    """Chance floor: per utterance, place the SAME NUMBER of boundaries uniformly
    at random in (0, duration); average the corpus metrics over seeded trials."""
    rng = random.Random(seed)
    agg: dict[str, float] = {}
    for _ in range(trials):
        counts = []
        for u in per_utt:
            n = u["n_pred"]
            fake = sorted(rng.uniform(0.0, u["duration"]) for _ in range(n))
            counts.append(match_boundaries(fake, u["true"], tol))
        m = aggregate(counts)
        for k in ("precision", "recall", "f1", "r_value", "over_seg"):
            agg[k] = agg.get(k, 0.0) + m[k] / trials
    return agg


def score_utterances(boundaries: Mapping[str, Sequence[float]],
                     alignments: Mapping[str, Mapping], tier: str,
                     tol: float = DEFAULT_TOL_S, *, min_t: float = 0.03,
                     durations: Mapping[str, float] | None = None) -> dict:
    """{utt: pred times} x {utt: alignment record} -> corpus metrics + baseline
    inputs. Only utterances present in BOTH are scored; BOTH directions of
    coverage gaps are reported (missing_alignments / missing_boundaries).
    `durations` (utt -> true audio seconds) sets the random-baseline dart board;
    without it the board is the last aligned-unit end, which excludes trailing
    silence and inflates the chance floor a few % relative — pass real durations
    whenever the baseline row is reported."""
    if tier not in ("words", "phones"):
        raise ValueError(f"tier must be 'words' or 'phones', got {tier!r}")
    counts, per_utt = [], []
    missing = sorted(set(boundaries) - set(alignments))
    missing_b = sorted(set(alignments) - set(boundaries))
    for uid in sorted(set(boundaries) & set(alignments)):
        rec = alignments[uid]
        edges = true_edges(rec[tier], min_t=min_t)
        pred = sorted(boundaries[uid])
        counts.append(match_boundaries(pred, edges, tol))
        duration = max((e for _, _, e in rec[tier]), default=0.0)
        if durations is not None and uid in durations:
            duration = float(durations[uid])
        per_utt.append({"n_pred": len(pred), "true": edges, "duration": duration})
    if not counts:
        raise ValueError("no utterances overlap between boundaries and alignments")
    if missing_b:                                   # the dangerous silent direction
        logger.warning("%d aligned utterances have no boundaries and are excluded "
                       "from the corpus metric", len(missing_b))
    out = aggregate(counts)
    out["tier"] = tier
    out["tol_s"] = tol
    out["missing_alignments"] = missing
    out["missing_boundaries"] = missing_b
    out["_per_utt"] = per_utt                       # feeds random_baseline
    return out


@torch.no_grad()
def collect_boundaries(model_encoder, loader, device) -> dict[int, dict[str, list[float]]]:
    """Run the encoder over a loader -> {stage: {utt id: boundary times}}.
    Duck-typed: encoder(feats, lens) returns .boundaries [(p, b), ...] and .lengths;
    stage 2 (Type B) is mapped through stage 1's kept frames."""
    out: dict[int, dict[str, list[float]]] = {}
    for batch in loader:
        enc = model_encoder(batch["feats"].to(device), batch["feat_lens"].to(device))
        n_stages = len(enc.boundaries)
        for s in range(n_stages):
            out.setdefault(s, {})
        for bi, uid in enumerate(batch["ids"]):
            L = int(enc.lengths[bi])
            b1 = enc.boundaries[0][1][bi].detach().float().cpu()
            out[0][uid] = frame_boundary_times(b1, L)
            if n_stages > 1:
                b2 = enc.boundaries[1][1][bi].detach().float().cpu()
                out[1][uid] = stage2_boundary_times(b1, b2, L)
    logger.info("collected boundaries: %d stages, %d utts",
                len(out), len(next(iter(out.values()), {})))
    return out
