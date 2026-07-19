"""Scoring: decode JSONLs -> WER/CER tables with S/D/I, paired-bootstrap significance,
and the go/no-go gate (plan §6.3 / goal.sane_test_clean_wer_below).

Consumes the per-utterance {id, ref, hyp, decode_s, audio_s} records decode.py writes
per cell × split. Word-level error counts are computed once per utterance (via the
verified eval/metrics.py Levenshtein) and reused for the corpus WER, the persisted
per-utterance counts (paper figures / custom significance), and the paired bootstrap
(Bisani & Ney): resample utterances with replacement, all cells sharing one index
stream, so pair deltas are exactly paired. Pure CPU, deterministic given seed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dcasr.eval.metrics import ErrorStats, levenshtein_counts, normalize_text
from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def load_decode_records(path: str | Path) -> list[dict]:
    """One cell×split decode JSONL -> records. Loud on empty/malformed/duplicate ids."""
    records = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} malformed JSON: {e}") from e
            if not isinstance(r, dict):
                raise ValueError(f"{path}:{ln} record is {type(r).__name__}, not an object")
            for key in ("id", "ref", "hyp"):
                if key not in r:
                    raise ValueError(f"{path}:{ln} missing {key!r}")
                if not isinstance(r[key], str):
                    raise ValueError(f"{path}:{ln} {key!r} is {type(r[key]).__name__}, "
                                     "not a string")
            if r["id"] in seen:
                raise ValueError(f"{path}:{ln} duplicate utterance id {r['id']!r}")
            seen.add(r["id"])
            records.append(r)
    if not records:
        raise ValueError(f"{path}: no decode records")
    return records


def score_records(records: Sequence[Mapping], normalize: bool = True) -> dict:
    """Corpus WER + CER ErrorStats and per-utterance word counts for one cell×split."""
    norm = normalize_text if normalize else (lambda s: s)
    wer, cer = ErrorStats(), ErrorStats()
    utts = []
    dec_s = aud_s = 0.0
    n_missing_audio = 0
    for r in records:
        ref_w, hyp_w = norm(r["ref"]).split(), norm(r["hyp"]).split()
        s, d, i, c = levenshtein_counts(ref_w, hyp_w)
        wer.n_ref += len(ref_w); wer.sub += s; wer.dele += d; wer.ins += i; wer.cor += c
        wer.n_utt += 1; wer.n_correct += int(ref_w == hyp_w)
        utts.append({"id": r["id"], "n_ref": len(ref_w), "sub": s, "del": d, "ins": i})
        ref_c = list(norm(r["ref"]).replace(" ", ""))
        hyp_c = list(norm(r["hyp"]).replace(" ", ""))
        s, d, i, c = levenshtein_counts(ref_c, hyp_c)
        cer.n_ref += len(ref_c); cer.sub += s; cer.dele += d; cer.ins += i; cer.cor += c
        cer.n_utt += 1; cer.n_correct += int(ref_c == hyp_c)
        dec_s += float(r.get("decode_s", 0.0))
        a_s = float(r.get("audio_s", 0.0))
        aud_s += a_s
        n_missing_audio += a_s <= 0.0
    if n_missing_audio:                       # partial audio would silently inflate RTF
        logger.warning("%d/%d records lack a positive audio_s — RTF suppressed",
                       n_missing_audio, len(utts))
    return {"wer": wer, "cer": cer, "utts": utts,
            "decode_s": round(dec_s, 3), "audio_s": round(aud_s, 3),
            "rtf": (round(dec_s / aud_s, 5)
                    if aud_s > 0 and not n_missing_audio else None)}


def cell_summary(scored: Mapping) -> dict:
    """Flat percent numbers for scores.json / the report table (wer_exact unrounded
    so the gate never decides on a display-rounded value)."""
    w, c = scored["wer"], scored["cer"]
    return {"n_utts": w.n_utt, "n_ref_words": w.n_ref,
            "wer": round(100 * w.er, 2), "wer_exact": 100 * w.er,
            "wer_sub": round(100 * w.sub_rate, 2),
            "wer_del": round(100 * w.del_rate, 2), "wer_ins": round(100 * w.ins_rate, 2),
            "sent_acc": round(100 * w.sentence_acc, 2), "cer": round(100 * c.er, 2),
            "decode_s": scored["decode_s"], "audio_s": scored["audio_s"],
            "rtf": scored["rtf"]}


def check_same_utterances(cells_utts: Mapping[str, Sequence[Mapping]],
                          split: str = "?") -> None:
    """Cells of one split must score the SAME utterances with the SAME reference
    lengths (else side-by-side WERs and the paired bootstrap are meaningless)."""
    names = list(cells_utts)
    first = {u["id"]: u["n_ref"] for u in cells_utts[names[0]]}
    zero = [uid for uid, n in first.items() if n == 0]
    if zero:
        raise ValueError(f"split {split!r}: zero-reference-word utterance(s) "
                         f"{zero[:5]} — WER is undefined for them")
    for n in names[1:]:
        other = {u["id"]: u["n_ref"] for u in cells_utts[n]}
        if set(other) != set(first):
            raise ValueError(f"split {split!r}: cells {names[0]!r} and {n!r} scored "
                             "different utterance sets")
        bad = [uid for uid, nr in other.items() if nr != first[uid]]
        if bad:
            raise ValueError(f"split {split!r}: cells {names[0]!r} and {n!r} disagree "
                             f"on reference length for {bad[:5]}")


def _aligned_arrays(cells_utts: Mapping[str, Sequence[Mapping]]):
    """Sort each cell's per-utt counts by id; all cells must cover the same utterances."""
    check_same_utterances(cells_utts)
    names = list(cells_utts)
    by_id = {n: sorted(cells_utts[n], key=lambda u: u["id"]) for n in names}
    E = np.array([[u["sub"] + u["del"] + u["ins"] for u in by_id[n]] for n in names],
                 dtype=np.int64)
    L = np.array([u["n_ref"] for u in by_id[names[0]]], dtype=np.int64)
    return names, E, L


def bootstrap_split(cells_utts: Mapping[str, Sequence[Mapping]], n_resamples: int = 10000,
                    seed: int = 0, chunk: int = 1000) -> dict:
    """Per-cell WER 95% CIs + pairwise paired-bootstrap deltas over one split.

    One shared resample-index stream: every cell (and so every pair) is evaluated on
    the same resampled utterance sets. p_value is two-sided with the +1 correction.
    """
    names, E, L = _aligned_arrays(cells_utts)
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    err_sums = np.empty((len(names), n_resamples), dtype=np.int64)
    ref_sums = np.empty(n_resamples, dtype=np.int64)
    for lo in range(0, n_resamples, chunk):
        hi = min(lo + chunk, n_resamples)
        idx = rng.integers(0, n, size=(hi - lo, n))
        ref_sums[lo:hi] = L[idx].sum(axis=1)
        for ci, _ in enumerate(names):
            err_sums[ci, lo:hi] = E[ci][idx].sum(axis=1)
    wer_r = 100.0 * err_sums / np.maximum(ref_sums, 1)          # resampled WERs, percent

    full_wer = 100.0 * E.sum(axis=1) / L.sum()
    cells = {}
    for ci, name in enumerate(names):
        lo95, hi95 = np.percentile(wer_r[ci], [2.5, 97.5])
        cells[name] = {"wer": round(float(full_wer[ci]), 2),
                       "wer_ci95": [round(float(lo95), 2), round(float(hi95), 2)]}
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            deltas = wer_r[i] - wer_r[j]
            delta = float(full_wer[i] - full_wer[j])
            p_le = (np.count_nonzero(deltas <= 0) + 1) / (n_resamples + 1)
            p_ge = (np.count_nonzero(deltas >= 0) + 1) / (n_resamples + 1)
            lo95, hi95 = np.percentile(deltas, [2.5, 97.5])
            pairs.append({"a": names[i], "b": names[j], "delta": round(delta, 2),
                          "delta_ci95": [round(float(lo95), 2), round(float(hi95), 2)],
                          "p_value": round(min(1.0, 2 * min(p_le, p_ge)), 5),
                          "n_resamples": n_resamples})
    return {"cells": cells, "pairs": pairs}


def discover_cells(decode_dir: str | Path) -> dict[str, dict[str, Path]]:
    """Scan decode.py's output tree -> {split: {cell: jsonl path}}."""
    decode_dir = Path(decode_dir)
    if not decode_dir.is_dir():
        raise ValueError(f"decode dir not found: {decode_dir}")
    found: dict[str, dict[str, Path]] = {}
    for cell_dir in sorted(p for p in decode_dir.iterdir() if p.is_dir() and p.name != "score"):
        for jl in sorted(cell_dir.glob("*.jsonl")):
            if jl.is_file():
                found.setdefault(jl.stem, {})[cell_dir.name] = jl
    if not found:
        raise ValueError(f"no decode outputs (<cell>/<split>.jsonl) under {decode_dir}")
    return found


def gate_check(split_cells: Mapping[str, Mapping[str, Mapping]], goal_cfg: Mapping | None,
               gate_split: str = "test-clean", gate_cell: str | None = None) -> dict:
    """goal.sane_test_clean_wer_below check (strict <, on the UNROUNDED WER); best
    (min-WER) cell unless one is pinned; ties break to the alphabetically first cell."""
    threshold = (goal_cfg or {}).get("sane_test_clean_wer_below")
    if threshold is None:
        return {"evaluated": False, "reason": "no goal.sane_test_clean_wer_below in config"}
    gate = {"evaluated": False, "threshold": float(threshold), "split": gate_split}
    cells = split_cells.get(gate_split)
    if not cells:
        gate["reason"] = f"split {gate_split!r} not decoded"
        return gate
    exact = {c: s.get("wer_exact", s["wer"]) for c, s in cells.items()}
    if gate_cell is not None:
        if gate_cell not in cells:
            gate["reason"] = f"gate cell {gate_cell!r} not decoded on {gate_split!r}"
            return gate
        best = gate_cell
    else:
        best = min(sorted(exact), key=lambda c: exact[c])
    gate.update(evaluated=True, cell=best, wer=exact[best],
                passed=bool(exact[best] < float(threshold)))
    return gate


def format_report(splits: Mapping[str, Mapping], gate: Mapping) -> str:
    """Human-readable tables: per split WER/S/D/I/CER rows + significance + gate."""
    lines = []
    for split, block in splits.items():
        cells = block["cells"]
        n = next(iter(cells.values()))["n_utts"] if cells else 0
        lines.append(f"== {split} ({n} utts) ==")
        header = f"{'cell':<16}{'WER':>8}{'Sub':>8}{'Del':>8}{'Ins':>8}{'CER':>8}{'SentAcc':>9}{'RTF':>10}"
        lines.append(header)
        for name, s in cells.items():
            rtf = f"{s['rtf']:.4f}" if s.get("rtf") is not None else "-"
            lines.append(f"{name:<16}{s['wer']:>8.2f}{s['wer_sub']:>8.2f}"
                         f"{s['wer_del']:>8.2f}{s['wer_ins']:>8.2f}{s['cer']:>8.2f}"
                         f"{s['sent_acc']:>9.2f}{rtf:>10}")
        for p in block.get("significance", {}).get("pairs", []):
            sig = "*" if p["p_value"] < 0.05 else " "
            lines.append(f"  {sig} {p['a']} vs {p['b']}: dWER {p['delta']:+.2f} "
                         f"[{p['delta_ci95'][0]:+.2f}, {p['delta_ci95'][1]:+.2f}] "
                         f"p={p['p_value']:.5f}")
        lines.append("")
    if gate.get("evaluated"):
        verdict = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"GATE {verdict}: {gate['cell']} {gate['split']} WER "
                     f"{gate['wer']:.3f} vs < {gate['threshold']:.2f}")
    else:
        lines.append(f"GATE not evaluated: {gate.get('reason', '?')}")
    return "\n".join(lines)


def write_per_utt(utts: Sequence[Mapping], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as w:
        for u in utts:
            w.write(json.dumps(u) + "\n")


def score_decode_dir(decode_dir: str | Path, *, normalize: bool = True,
                     n_bootstrap: int = 10000, seed: int = 0,
                     goal_cfg: Mapping | None = None, gate_split: str = "test-clean",
                     gate_cell: str | None = None) -> dict:
    """Score every cell×split under a decode dir; write per-utt counts under score/."""
    decode_dir = Path(decode_dir)
    tree = discover_cells(decode_dir)
    if (decode_dir / "score").is_dir():                  # rerun: no stale per-utt files
        shutil.rmtree(decode_dir / "score")
    splits: dict[str, Any] = {}
    for split, cells in tree.items():
        cell_stats, cell_utts = {}, {}
        for cell, path in cells.items():
            scored = score_records(load_decode_records(path), normalize=normalize)
            cell_stats[cell] = cell_summary(scored)
            cell_utts[cell] = scored["utts"]
        check_same_utterances(cell_utts, split)          # loud even with n_bootstrap=0
        for cell in cells:
            write_per_utt(cell_utts[cell], decode_dir / "score" / cell / f"{split}.jsonl")
        splits[split] = {"cells": cell_stats}
        if n_bootstrap > 0:
            boot = bootstrap_split(cell_utts, n_resamples=n_bootstrap, seed=seed)
            for cell in cell_stats:                          # bootstrapped CI joins the row
                cell_stats[cell]["wer_ci95"] = boot["cells"][cell]["wer_ci95"]
            splits[split]["significance"] = {"pairs": boot["pairs"],
                                             "n_resamples": n_bootstrap, "seed": seed}
    gate = gate_check({s: b["cells"] for s, b in splits.items()}, goal_cfg,
                      gate_split=gate_split, gate_cell=gate_cell)
    return {"splits": splits, "gate": gate}
