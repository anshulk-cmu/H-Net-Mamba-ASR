"""MFA forced-alignment support (plan #9): corpus prep, TextGrid parsing, integrity.

The Montreal Forced Aligner runs as an external CLI (its own conda env); this module
holds everything testable around it: laying out a manifest as an MFA corpus
(<speaker>/<utt>.flac symlink + <utt>.lab transcript), selecting a seeded train
subset under an hours budget, parsing MFA's long-format TextGrid output into
{id, words: [[w, start, end]], phones: [[p, start, end]]} records, and checking
each record against its manifest transcript and audio duration. Consumers:
boundary_align.py (±20 ms boundary F1) and probes.py (frame labels).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000
# MFA 3.4.1 marks silence EXCLUSIVELY with the empty label (verified census of real
# output) — matching any word-like label ('sil', 'silence') would delete real words
# from the ground truth. OOV speech keeps its word with a single 'spn' phone: kept.
SILENCE_LABELS = {""}


def speaker_of(utt_id: str) -> str:
    """LibriSpeech id '1272-128104-0000' -> speaker '1272'."""
    return str(utt_id).split("-")[0]


def load_manifest(path: str | Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    if not entries:
        raise ValueError(f"{path}: empty manifest")
    return entries


def prepare_corpus(entries: Sequence[Mapping], corpus_dir: str | Path) -> int:
    """Lay out an MFA corpus: <speaker>/<utt>.flac (symlink) + <utt>.lab (transcript)."""
    corpus_dir = Path(corpus_dir)
    seen: set[str] = set()
    for e in entries:
        if e["id"] in seen:
            raise ValueError(f"duplicate utterance id in manifest: {e['id']!r}")
        seen.add(e["id"])
        audio = Path(e["audio"])
        if not audio.is_file():
            raise FileNotFoundError(f"audio missing for {e['id']}: {audio}")
        spk_dir = corpus_dir / speaker_of(e["id"])
        spk_dir.mkdir(parents=True, exist_ok=True)
        link = spk_dir / f"{e['id']}{audio.suffix}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(audio.resolve())              # relative targets would dangle
        (spk_dir / f"{e['id']}.lab").write_text(e["text"] + "\n", encoding="utf-8")
    logger.info("prepared MFA corpus: %d utts -> %s", len(entries), corpus_dir)
    return len(entries)


def select_subset(entries: Sequence[Mapping], hours: float, seed: int = 1) -> list[dict]:
    """Seeded random utterance subset filling an audio-hours budget (deterministic)."""
    if hours <= 0:
        raise ValueError(f"hours must be positive, got {hours}")
    order = list(entries)
    random.Random(seed).shuffle(order)
    budget_s = hours * 3600.0
    picked, total = [], 0.0
    for e in order:
        dur = e["frames"] / SAMPLE_RATE
        if total + dur > budget_s and picked:
            continue
        picked.append(e)
        total += dur
        if total >= budget_s:
            break
    picked.sort(key=lambda e: e["id"])
    logger.info("subset: %d utts, %.2f h (budget %.2f h, seed %d)",
                len(picked), total / 3600, hours, seed)
    return picked


# ── TextGrid parsing (MFA long format; no external deps) ─────────────────────
_ITEM_RE = re.compile(r"item\s*\[\d+\]:")
_KV_RE = re.compile(r'^\s*(\w+)\s*=\s*(.+?)\s*$')


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return s.replace('""', '"')                       # Praat escapes " by doubling


def parse_textgrid(text: str) -> dict[str, list[tuple[str, float, float]]]:
    """MFA long-format TextGrid -> {tier_name: [(label, xmin, xmax), ...]}.

    Keeps every non-empty-label interval (silence filtering is the caller's call).
    """
    head = text[:200]
    if "ooTextFile" not in head or "TextGrid" not in head:
        raise ValueError('not a TextGrid (needs File type = "ooTextFile" + '
                         'Object class = "TextGrid" header)')
    tiers: dict[str, list[tuple[str, float, float]]] = {}
    name = None
    xmin = xmax = None
    label: str | None = None
    in_intervals = False
    for raw in text.splitlines():
        line = raw.strip()
        if _ITEM_RE.match(line):
            name, in_intervals = None, False
            continue
        m = _KV_RE.match(line)
        if not m:
            if line.startswith("intervals ["):
                in_intervals = True
                xmin = xmax = label = None
            continue
        key, val = m.group(1), m.group(2)
        if key == "name":
            name = _unquote(val)
            tiers.setdefault(name, [])
            in_intervals = False
        elif in_intervals and name is not None:
            if key == "xmin":
                xmin = float(val)
            elif key == "xmax":
                xmax = float(val)
            elif key == "text":
                if val.startswith('"') and (len(val) < 2 or not val.endswith('"')):
                    raise ValueError(f"unterminated label in tier {name!r} "
                                     "(multi-line labels unsupported)")
                label = _unquote(val)
                if xmin is None or xmax is None:
                    raise ValueError(f"interval text before xmin/xmax in tier {name!r}")
                tiers[name].append((label, xmin, xmax))
    if not tiers:
        raise ValueError("no tiers found in TextGrid")
    return tiers


def alignment_record(utt_id: str, tiers: Mapping[str, Sequence], *,
                     drop_silence: bool = True) -> dict:
    """Parsed tiers -> {id, words, phones} with [label, start, end] triples."""
    out: dict = {"id": utt_id, "words": [], "phones": []}
    for tier_name, key in (("words", "words"), ("phones", "phones")):
        for label, s, e in tiers.get(tier_name, []):
            if drop_silence and label in SILENCE_LABELS:
                continue
            out[key].append([label, round(float(s), 6), round(float(e), 6)])
    if not out["words"] or not out["phones"]:
        raise ValueError(f"{utt_id}: TextGrid lacks words/phones intervals "
                         f"(tiers: {sorted(tiers)})")
    return out


def check_alignment(record: Mapping, text: str, duration_s: float,
                    tol: float = 0.05) -> list[str]:
    """Integrity problems (empty list = clean): word sequence must equal the
    transcript (case-insensitive; edge apostrophes stripped — MFA normalizes
    BUSH' -> bush), tiers monotone/non-overlapping, times in range."""
    problems = []
    ref = [w.lower().strip("'") for w in text.split()]
    hyp = [w.lower().strip("'") for w, _, _ in record["words"]]
    if hyp != ref:
        diff = next((i for i, (a, b) in enumerate(zip(ref, hyp)) if a != b),
                    min(len(ref), len(hyp)))
        problems.append(f"word sequence != transcript ({len(hyp)} vs {len(ref)} words; "
                        f"first diff at {diff}: ref {ref[diff:diff + 1]} vs "
                        f"hyp {hyp[diff:diff + 1]})")
    for tier in ("words", "phones"):
        prev_end = 0.0
        for label, s, e in record[tier]:
            if s < 0:
                problems.append(f"{tier}: negative start {label!r} at {s}")
            if e <= s:
                problems.append(f"{tier}: non-positive interval {label!r} [{s}, {e}]")
            if s < prev_end - 1e-6:
                problems.append(f"{tier}: overlap at {label!r} ({s} < {prev_end})")
            prev_end = max(prev_end, e)
        if record[tier] and record[tier][-1][2] > duration_s + tol:
            problems.append(f"{tier}: end {record[tier][-1][2]} beyond audio "
                            f"{duration_s:.2f}s")
    return problems


def write_alignments(records: Iterable[Mapping], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")
            n += 1
    logger.info("wrote %d alignment records -> %s", n, path)
    return n


def load_alignments(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
