"""Structured metrics logging for DC-ASR: TensorBoard + append-only JSONL.

Text logs (logging_utils) carry human-readable messages; this module stores the
*numbers* for plotting. Every process that produces metrics (trainer, decode/eval,
interp) creates ONE MetricsLogger next to setup_logging(), then calls log_scalar /
log_scalars / log_histogram / update_summary. Rank-0 only under DDP — other ranks
get a no-op instance so call sites stay clean.

Per-run artifacts live under <repo>/experiments/<run>/ (symlinked to the data node;
override with $DCASR_METRICS_DIR):
    tb/            TensorBoard event files (live curves)
    metrics.jsonl  one JSON record per scalar {wall_time, step, epoch, split, key, value}
    summary.json   run metadata + final headline numbers (atomic overwrite)
metrics.jsonl is opened append-only and keyed by global_step, so a resumed run
continues its curves without clobbering earlier records.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def default_metrics_dir() -> Path:
    """$DCASR_METRICS_DIR if set, else <repo>/experiments (symlinked to the data node)."""
    env = os.environ.get("DCASR_METRICS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "experiments"


def _auto_rank(rank: int | None) -> int:
    if rank is not None:
        return rank
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


class MetricsLogger:
    """Rank-0 metrics sink: TensorBoard scalars/histograms + an append-only JSONL.

    Non-main ranks (rank != 0) build a no-op instance: every method returns
    immediately, run_dir is None, and no files are created.
    """

    def __init__(self, run_name: str = "dcasr", root: str | Path | None = None,
                 rank: int | None = None, resume: bool = False):
        self.run_name = run_name
        self.rank = _auto_rank(rank)
        self.is_main = self.rank == 0
        self.run_dir: Path | None = None
        self._writer = None
        self._jsonl = None
        self._summary: dict = {}
        if not self.is_main:
            return
        root = Path(root) if root is not None else default_metrics_dir()
        self.run_dir = root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        from torch.utils.tensorboard import SummaryWriter          # lazy: heavy import
        self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        self._jsonl = open(self.jsonl_path, "a" if resume else "w", encoding="utf-8")
        if resume and self.summary_path.exists():
            try:
                self._summary = json.loads(self.summary_path.read_text())
            except json.JSONDecodeError:
                logger.warning("unreadable summary.json at %s; starting fresh", self.summary_path)
        logger.info("metrics -> %s (tb + metrics.jsonl)%s", self.run_dir,
                    " [resume]" if resume else "")

    # ---- scalars ------------------------------------------------------------
    def log_scalar(self, key: str, value: Any, step: int, *, split: str | None = None,
                   epoch: int | None = None) -> None:
        if not self.is_main:
            return
        v = _to_float(value)
        self._writer.add_scalar(key, v, step)
        rec = {"wall_time": time.time(), "step": int(step), "epoch": epoch,
               "split": split, "key": key, "value": v}
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()                                        # preemption-safe

    def log_scalars(self, values: Mapping[str, Any], step: int, *,
                    split: str | None = None, epoch: int | None = None) -> None:
        """Log a dict of {key: value} at one step (e.g. all loss components)."""
        if not self.is_main:
            return
        for key, value in values.items():
            self.log_scalar(key, value, step, split=split, epoch=epoch)

    # ---- histograms ---------------------------------------------------------
    def log_histogram(self, key: str, values: torch.Tensor, step: int, *,
                      split: str | None = None, epoch: int | None = None) -> None:
        """TB histogram (e.g. per-stage boundary-prob distribution) + JSONL summary stats."""
        if not self.is_main:
            return
        v = values.detach().float().flatten()
        self._writer.add_histogram(key, v, step)
        if v.numel():
            stats = {f"{key}/mean": v.mean(), f"{key}/min": v.min(), f"{key}/max": v.max()}
            if v.numel() > 1:
                stats[f"{key}/std"] = v.std()
            self.log_scalars(stats, step, split=split, epoch=epoch)

    # ---- run-level summary --------------------------------------------------
    def update_summary(self, **kwargs: Any) -> None:
        """Merge key/values into summary.json (run metadata + final headline WERs)."""
        if not self.is_main:
            return
        self._summary.update(kwargs)
        self._write_summary()

    def _write_summary(self) -> None:
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._summary, indent=2, default=str))
        os.replace(tmp, self.summary_path)                         # atomic

    # ---- lifecycle ----------------------------------------------------------
    def flush(self) -> None:
        if not self.is_main:
            return
        self._writer.flush()
        self._jsonl.flush()

    def close(self) -> None:
        if not self.is_main:
            return
        if self._summary:
            self._write_summary()
        self._writer.close()
        self._jsonl.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
