"""Interp driver (plan #12 / §6.4): orchestrates boundary alignment (#10) and
linear probes (#11) over trained checkpoints — single-checkpoint reports,
emergence curves over retained epoch checkpoints (H4), and boundary robustness
under waveform perturbations (noise / speed / silence insertion).

Verification mandates baked in (from the #10/#11 adversarial verifications):
(a) probe train/test utterance sets are asserted disjoint on the ids ACTUALLY
consumed from the loaders (contamination silently inflates accuracy +0.77);
(b) true audio durations are required for every scored utterance (the random-
baseline dart board; the last-unit-end fallback inflates the chance floor);
(c) the word probe reports its top-k kept fractions on both sides beside
accuracy (top-500 covers ~49% of test frames); (d) the random-baseline floor is
computed and persisted next to every boundary metric — phones floors are high
(F1 ~ 0.33) and every phones plot must show them.
"""
from __future__ import annotations

import zlib
from pathlib import Path
from typing import Callable, Mapping, Sequence

import soundfile as sf
import torch

from dcasr.data.librispeech import (SAMPLE_RATE, LibriSpeechDataset,
                                    apply_speed_perturb, feat_frames)
from dcasr.interp.boundary_align import (DEFAULT_TOL_S, aggregate,
                                         collect_boundaries, match_boundaries,
                                         random_baseline, score_utterances)
from dcasr.interp.probes import (collect_probe_data, subsample, to_classes,
                                 top_k_filter, train_probe)
from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


# ── waveform perturbations (plan §6.4 item 6) ────────────────────────────────
def utt_seed(seed: int, uid: str) -> int:
    """Per-utterance RNG seed, stable across processes (hash() is not)."""
    return (zlib.crc32(str(uid).encode("utf-8")) ^ (int(seed) * 0x9E3779B9)) & 0x7FFFFFFF


class Perturbation:
    """Identity base: apply_wave perturbs audio; _t maps a CLEAN time to its
    perturbed-time coordinate (applied to true edges and clean boundaries)."""

    kind = "identity"
    name = "identity"

    def apply_wave(self, wave: torch.Tensor, uid: str, *, seed: int = 1,
                   sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
        return wave

    def _t(self, t: float, duration: float) -> float:
        return t

    def transform_times(self, times: Sequence[float], duration: float) -> list[float]:
        return [self._t(float(t), duration) for t in times]

    def transform_record(self, record: Mapping, duration: float) -> dict:
        out = {k: v for k, v in record.items() if k not in ("words", "phones")}
        for tier in ("words", "phones"):
            out[tier] = [[l, self._t(float(s), duration), self._t(float(e), duration)]
                         for l, s, e in record[tier]]
        return out

    def transform_duration(self, duration: float) -> float:
        return duration

    def transform_samples(self, n: int) -> int:
        return int(n)


class NoisePerturbation(Perturbation):
    """Additive white Gaussian noise at a target SNR; timings unchanged."""

    kind = "noise"

    def __init__(self, snr_db: float):
        self.snr_db = float(snr_db)
        self.name = f"noise_snr{self.snr_db:g}"

    def apply_wave(self, wave, uid, *, seed=1, sample_rate=SAMPLE_RATE):
        power = float(wave.pow(2).mean())
        if power <= 0.0:
            return wave
        g = torch.Generator().manual_seed(utt_seed(seed, uid))
        scale = (power / (10.0 ** (self.snr_db / 10.0))) ** 0.5
        return wave + scale * torch.randn(wave.shape, generator=g)


class SpeedPerturbation(Perturbation):
    """Resampling speed change: audio at `factor`x speed, times scale 1/factor."""

    kind = "speed"

    def __init__(self, factor: float):
        if factor <= 0:
            raise ValueError(f"speed factor must be positive, got {factor}")
        self.factor = float(factor)
        self.name = f"speed_{self.factor:g}"

    def apply_wave(self, wave, uid, *, seed=1, sample_rate=SAMPLE_RATE):
        return apply_speed_perturb(wave, sample_rate, self.factor)

    def _t(self, t, duration):
        return t / self.factor

    def transform_duration(self, duration):
        return duration / self.factor

    def transform_samples(self, n):
        return round(n / self.factor)


class SilencePerturbation(Perturbation):
    """Insert `duration_s` of digital silence at fraction `at_frac` of the
    utterance; times at/after the insertion point shift by duration_s."""

    kind = "silence"

    def __init__(self, duration_s: float, at_frac: float = 0.5):
        if duration_s <= 0 or not 0.0 <= at_frac <= 1.0:
            raise ValueError(f"need duration_s > 0 and at_frac in [0, 1], got "
                             f"{duration_s}, {at_frac}")
        self.duration_s = float(duration_s)
        self.at_frac = float(at_frac)
        self.name = f"silence_{self.duration_s:g}s_at{self.at_frac:g}"

    def apply_wave(self, wave, uid, *, seed=1, sample_rate=SAMPLE_RATE):
        n = wave.shape[-1]
        at = round(self.at_frac * n)
        gap = wave.new_zeros(*wave.shape[:-1], round(self.duration_s * sample_rate))
        return torch.cat([wave[..., :at], gap, wave[..., at:]], dim=-1)

    def _t(self, t, duration):
        return t if t < self.at_frac * duration else t + self.duration_s

    def window(self, duration: float) -> tuple[float, float]:
        """Inserted-silence interval in perturbed-time coordinates."""
        t0 = self.at_frac * duration
        return t0, t0 + self.duration_s

    def transform_duration(self, duration):
        return duration + self.duration_s

    def transform_samples(self, n):
        return int(n) + round(self.duration_s * SAMPLE_RATE)


def perturbations_from_config(cfg: Mapping) -> list[Perturbation]:
    """robustness config block -> perturbation list (empty block -> defaults)."""
    perts: list[Perturbation] = []
    for snr in cfg.get("noise_snr_db", (20, 10, 5, 0)):
        perts.append(NoisePerturbation(snr))
    for f in cfg.get("speed_factors", (0.9, 1.1)):
        perts.append(SpeedPerturbation(f))
    sil = cfg.get("silence", {})
    if sil is not None and sil is not False:      # silence: null/false disables;
        sil = sil or {}                           # {} or absent -> defaults
        perts.append(SilencePerturbation(float(sil.get("duration_s", 0.5)),
                                         float(sil.get("at_frac", 0.5))))
    return perts


class PerturbedDataset(LibriSpeechDataset):
    """Eval dataset applying a waveform perturbation before the frontend.
    `lengths` are recomputed from perturbed sample counts (approximate for
    speed — resampler rounding; the sampler only budgets with them)."""

    def __init__(self, entries, frontend, tokenizer, cmvn, perturbation: Perturbation,
                 seed: int = 1):
        super().__init__(entries, frontend, tokenizer, cmvn=cmvn, augment=False)
        self.perturbation = perturbation
        self.perturb_seed = int(seed)
        self.lengths = [feat_frames(perturbation.transform_samples(e["frames"]))
                        for e in self.entries]

    def __getitem__(self, i):
        idx, _ = self._items[i]
        e = self.entries[idx]
        wave_np, _ = sf.read(e["audio"])
        wave = torch.from_numpy(wave_np).unsqueeze(0).float()
        wave = self.perturbation.apply_wave(wave, e["id"], seed=self.perturb_seed,
                                            sample_rate=self.sample_rate)
        feats, _ = self.frontend(wave)
        if self.cmvn is not None:
            feats = self.cmvn(feats)
        tokens = torch.tensor(self.tokenizer.encode(e["text"]), dtype=torch.long)
        return {"feats": feats[0], "tokens": tokens, "id": e["id"]}


# ── shared plumbing ──────────────────────────────────────────────────────────
def durations_from_entries(entries: Sequence[Mapping]) -> dict[str, float]:
    """Manifest entries -> {id: true audio seconds} (frames = raw sample count)."""
    return {e["id"]: e["frames"] / SAMPLE_RATE for e in entries}


def assert_disjoint(a: set, b: set, what: str = "probe train/test") -> None:
    overlap = set(a) & set(b)
    if overlap:
        raise ValueError(
            f"{what} utterance sets overlap: {len(overlap)} shared ids "
            f"(e.g. {sorted(overlap)[:5]}) — contamination silently inflates "
            "probe accuracy; use disjoint splits")
    logger.info("%s disjointness verified: %d vs %d utts, 0 shared",
                what, len(set(a)), len(set(b)))


class RecordingLoader:
    """Loader wrapper accumulating every consumed utterance id in .seen."""

    def __init__(self, loader):
        self.loader = loader
        self.seen: set[str] = set()

    def __iter__(self):
        for batch in self.loader:
            self.seen.update(batch["ids"])
            yield batch


def list_epoch_checkpoints(ckpt_dir: str | Path) -> list[tuple[int, Path]]:
    """checkpoints/<run>/epoch*.pt -> [(epoch, path)] numerically sorted."""
    out = []
    for p in Path(ckpt_dir).glob("epoch*.pt"):
        try:
            out.append((int(p.stem[5:]), p))
        except ValueError:
            continue
    if not out:
        raise FileNotFoundError(
            f"no epoch*.pt checkpoints under {ckpt_dir} — emergence needs "
            "retained epochs (keep_all_checkpoints: true)")
    return sorted(out)


def matched_deltas(pred: Sequence[float], true: Sequence[float],
                   tol: float = DEFAULT_TOL_S) -> list[float]:
    """Signed pred-true offsets of the greedy matcher's hit pairs (same walk
    as match_boundaries; len(result) == its hit count)."""
    pred, true = sorted(pred), sorted(true)
    eps = 1e-9
    out: list[float] = []
    i = j = 0
    while i < len(pred) and j < len(true):
        d = pred[i] - true[j]
        if abs(d) <= tol + eps:
            out.append(d)
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return out


# ── boundary report (mandates b + coverage) ──────────────────────────────────
def boundary_report(encoder, loader, alignments: Mapping[str, Mapping],
                    durations: Mapping[str, float], *, device,
                    tol: float = DEFAULT_TOL_S, baseline_trials: int = 20,
                    baseline_seed: int = 1, require_coverage: bool = True,
                    ) -> tuple[dict, dict]:
    """Collect + score boundaries per stage x tier, with the random-baseline
    floor attached to every metric block. Returns (report, raw boundary times);
    raises if any scored utterance lacks a true duration or (by default) if any
    aligned utterance produced no boundaries."""
    bounds = collect_boundaries(encoder, loader, device)
    if not bounds or not next(iter(bounds.values()), {}):
        raise ValueError("collect_boundaries returned nothing — empty loader?")
    scored = set().union(*bounds.values()) & set(alignments)
    missing_dur = sorted(scored - set(durations))
    if missing_dur:
        raise ValueError(f"{len(missing_dur)} scored utterances lack a true "
                         f"duration (e.g. {missing_dur[:5]}) — the random "
                         "baseline needs real audio durations")
    report: dict = {}
    for s in sorted(bounds):
        per_stage: dict = {}
        for tier in ("words", "phones"):
            m = score_utterances(bounds[s], alignments, tier, tol,
                                 durations=durations)
            if require_coverage and m["missing_boundaries"]:
                raise ValueError(
                    f"stage {s}: {len(m['missing_boundaries'])} aligned "
                    f"utterances have no boundaries (e.g. "
                    f"{m['missing_boundaries'][:5]}) — a partial collection "
                    "biases the corpus metric")
            m["random_baseline"] = random_baseline(
                m.pop("_per_utt"), tol, seed=baseline_seed, trials=baseline_trials)
            per_stage[tier] = m
        report[f"stage{s}"] = per_stage
    return report, bounds


# ── probe report (mandates a + c) ────────────────────────────────────────────
def _fit_probe(Xtr, ytr, Xte, yte, *, train_cap, test_cap, max_iter, C, seed,
               backend="sklearn"):
    n_tr, n_te = len(ytr), len(yte)
    Xtr, ytr = subsample(Xtr, ytr, train_cap, seed=seed)
    Xte, yte = subsample(Xte, yte, test_cap, seed=seed)
    out = train_probe(Xtr, ytr, Xte, yte, max_iter=max_iter, C=C, seed=seed,
                      backend=backend)
    out.update(n_collected_train=n_tr, n_collected_test=n_te)
    return out


def probe_report(encoder, train_loader, test_loader,
                 train_alignments: Mapping[str, Mapping],
                 test_alignments: Mapping[str, Mapping], *, device, n_stages: int,
                 levels: Sequence[str] = ("frames", "chunks"),
                 top_k_words: int = 500, train_cap: int = 50000,
                 test_cap: int = 20000, max_iter: int = 1000, C: float = 1.0,
                 seed: int = 1, backend: str = "sklearn") -> dict:
    """phone_id / phone_class / word_id probes per representation level.
    Disjointness of the utterance sets actually consumed from the two loaders
    is asserted after every collection round (mandate a); word probes carry
    top-k kept fractions on both sides (mandate c). For word_id the
    n_collected_* fields count POST-top-k-filter samples (raw counts =
    n_collected / kept_fraction); phone probes count all labeled samples."""
    unknown = set(levels) - {"frames", "chunks"}
    if unknown:
        raise ValueError(f"unknown probe levels {sorted(unknown)}; "
                         "choose from 'frames', 'chunks'")
    train_loader = RecordingLoader(train_loader)
    test_loader = RecordingLoader(test_loader)
    slots = [("frames", 0)] if "frames" in levels else []
    if "chunks" in levels:
        slots += [("chunks", s) for s in range(int(n_stages))]
    if not slots:
        raise ValueError(f"no probe levels selected from {levels!r}")
    report: dict = {}
    for level, stage in slots:
        key = "frames" if level == "frames" else f"chunks_s{stage}"
        entry: dict = {}
        Xtr, ytr = collect_probe_data(encoder, train_loader, train_alignments,
                                      "phones", device, level=level, stage=stage)
        Xte, yte = collect_probe_data(encoder, test_loader, test_alignments,
                                      "phones", device, level=level, stage=stage)
        assert_disjoint(train_loader.seen, test_loader.seen)
        entry["phone_id"] = _fit_probe(Xtr, ytr, Xte, yte, train_cap=train_cap,
                                       test_cap=test_cap, max_iter=max_iter,
                                       C=C, seed=seed, backend=backend)
        entry["phone_class"] = _fit_probe(Xtr, to_classes(ytr), Xte, to_classes(yte),
                                          train_cap=train_cap, test_cap=test_cap,
                                          max_iter=max_iter, C=C, seed=seed,
                                          backend=backend)
        Xtr, ytr = collect_probe_data(encoder, train_loader, train_alignments,
                                      "words", device, level=level, stage=stage)
        Xte, yte = collect_probe_data(encoder, test_loader, test_alignments,
                                      "words", device, level=level, stage=stage)
        assert_disjoint(train_loader.seen, test_loader.seen)
        Xtr, ytr, train_cov = top_k_filter(Xtr, ytr, top_k_words)
        keep = set(ytr)
        kept = [i for i, lab in enumerate(yte) if lab in keep]
        test_kept = len(kept) / max(1, len(yte))
        Xte, yte = [Xte[i] for i in kept], [yte[i] for i in kept]
        w = _fit_probe(Xtr, ytr, Xte, yte, train_cap=train_cap, test_cap=test_cap,
                       max_iter=max_iter, C=C, seed=seed, backend=backend)
        w.update(top_k=top_k_words, train_kept_fraction=train_cov,
                 test_kept_fraction=test_kept)
        entry["word_id"] = w
        report[key] = entry
    return report


# ── robustness (plan §6.4 item 6) ────────────────────────────────────────────
def score_perturbation(pert: Perturbation, clean_bounds: Mapping[int, Mapping],
                       pert_bounds: Mapping[int, Mapping],
                       alignments: Mapping[str, Mapping],
                       durations: Mapping[str, float], *,
                       tol: float = DEFAULT_TOL_S, baseline_trials: int = 10,
                       baseline_seed: int = 1) -> dict:
    """Perturbed boundaries vs (i) time-transformed MFA truth (F1 + floor) and
    (ii) time-transformed clean boundaries (consistency: do boundaries move?)."""
    t_align = {u: pert.transform_record(alignments[u], durations[u])
               for u in alignments if u in durations}
    t_dur = {u: pert.transform_duration(d) for u, d in durations.items()}
    out: dict = {}
    for s in sorted(pert_bounds):
        stage: dict = {}
        for tier in ("words", "phones"):
            m = score_utterances(pert_bounds[s], t_align, tier, tol,
                                 durations=t_dur)
            m["random_baseline"] = random_baseline(
                m.pop("_per_utt"), tol, seed=baseline_seed, trials=baseline_trials)
            stage[tier] = m
        shared = sorted(set(pert_bounds[s]) & set(clean_bounds.get(s, {}))
                        & set(durations))
        if not shared:
            raise ValueError(f"stage {s}: no shared utterances between clean and "
                             f"{pert.name} boundary collections")
        counts, deltas = [], []
        for u in shared:
            t_clean = pert.transform_times(clean_bounds[s][u], durations[u])
            counts.append(match_boundaries(pert_bounds[s][u], t_clean, tol))
            deltas.extend(matched_deltas(pert_bounds[s][u], t_clean, tol))
        cons = aggregate(counts)
        cons["mean_abs_shift_s"] = (sum(abs(d) for d in deltas) / len(deltas)
                                    if deltas else 0.0)
        stage["consistency"] = cons
        if isinstance(pert, SilencePerturbation):
            stage["inserted_window"] = _window_stats(pert, pert_bounds[s], durations)
        out[f"stage{s}"] = stage
    return out


def _window_stats(pert: SilencePerturbation, bounds: Mapping[str, Sequence[float]],
                  durations: Mapping[str, float]) -> dict:
    """Boundary rate strictly inside the inserted-silence window vs overall —
    boundaries in pure silence track acoustics, not linguistic content (the
    window EDGES are real speech/silence transitions and are not counted)."""
    in_win = total = 0
    total_t = win_t = 0.0
    for u, times in bounds.items():
        if u not in durations:
            continue
        lo, hi = pert.window(durations[u])
        in_win += sum(1 for t in times if lo < t < hi)
        total += len(times)
        win_t += pert.duration_s
        total_t += pert.transform_duration(durations[u])
    return {"n_in_window": in_win, "n_total": total,
            "window_rate_per_s": in_win / win_t if win_t else 0.0,
            "overall_rate_per_s": total / total_t if total_t else 0.0}


def robustness_report(perturbations: Sequence[Perturbation],
                      collect_fn: Callable[[Perturbation], Mapping[int, Mapping]],
                      clean_bounds: Mapping[int, Mapping],
                      alignments: Mapping[str, Mapping],
                      durations: Mapping[str, float], *,
                      tol: float = DEFAULT_TOL_S, baseline_trials: int = 10,
                      baseline_seed: int = 1) -> dict:
    """Run collect_fn per perturbation and score against truth + clean run."""
    names = [p.name for p in perturbations]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate perturbation names {dupes} would silently "
                         "overwrite each other's results")
    out = {}
    for pert in perturbations:
        logger.info("robustness: collecting boundaries under %s", pert.name)
        out[pert.name] = score_perturbation(
            pert, clean_bounds, collect_fn(pert), alignments, durations,
            tol=tol, baseline_trials=baseline_trials, baseline_seed=baseline_seed)
    return out


# ── emergence curves (H4) ────────────────────────────────────────────────────
_CURVE_KEYS = {"precision", "recall", "f1", "r_value", "over_seg", "accuracy",
               "balanced_accuracy", "majority_baseline", "chance",
               "train_kept_fraction", "test_kept_fraction", "mean_abs_shift_s"}


def flatten_metrics(nested: Mapping, prefix: str = "interp") -> dict[str, float]:
    """Nested report -> {slash/joined/key: float} for curve-worthy leaves only
    (includes each random_baseline floor — plotted under phones, mandate d)."""
    out: dict[str, float] = {}

    def walk(node, path):
        for k, v in node.items():
            if isinstance(v, Mapping):
                walk(v, path + [str(k)])
            elif k in _CURVE_KEYS and isinstance(v, (int, float)):
                out["/".join(path + [str(k)])] = float(v)

    walk(nested, [prefix])
    return out


def emergence_report(model, checkpoints: Sequence[tuple[int, Path]], loader,
                     alignments: Mapping[str, Mapping],
                     durations: Mapping[str, float], *, device,
                     tol: float = DEFAULT_TOL_S, baseline_trials: int = 10,
                     baseline_seed: int = 1, load_fn: Callable | None = None,
                     probe_fn: Callable | None = None, mlogger=None) -> list[dict]:
    """Boundary metrics (+ optional probes via probe_fn(encoder)) per retained
    epoch checkpoint; scalars go to mlogger at step=epoch. Mutates the model's
    weights — reload the target checkpoint afterwards if it is used again."""
    if load_fn is None:
        from dcasr.tasks.decode_task import load_model_weights
        load_fn = load_model_weights
    rows = []
    for epoch, path in checkpoints:
        load_fn(model, path)
        rep, _ = boundary_report(model.encoder, loader, alignments, durations,
                                 device=device, tol=tol,
                                 baseline_trials=baseline_trials,
                                 baseline_seed=baseline_seed)
        row = {"epoch": int(epoch), "checkpoint": str(path), "boundaries": rep}
        if probe_fn is not None:
            row["probes"] = probe_fn(model.encoder)
        if mlogger is not None:
            mlogger.log_scalars(flatten_metrics({k: row[k] for k in row
                                                 if isinstance(row[k], dict)}),
                                step=int(epoch), epoch=int(epoch))
        logger.info("emergence: epoch %d done (%s)", epoch, Path(path).name)
        rows.append(row)
    return rows
