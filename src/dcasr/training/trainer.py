"""DC-ASR training loop: config-driven, DDP-capable, resumable (plan §6.3, ESPnet-style).

Model-agnostic: takes any `model(feats, feat_lens, targets, target_lens) -> (loss, stats)`
(built by tasks.asr_task.build_model) and never imports a concrete encoder/head — so CTC-now
and AED-later share it. Optimizer/scheduler come from the config via build_optimizer/
build_scheduler. Loss aggregation is weighted-mean (weight = batch rows, or the model's
stats["batch_weight"] when provided — the LM emits its token count so exp(valid/loss) is a
true token-weighted perplexity), DDP all-reduced.

Validation runs every `valid_interval_epoch` on ALL dev splits (dev-clean, dev-other): per-split
WER/CER/loss are logged separately and an aggregate mean feeds the monitor. Checkpoint selection
mirrors ESPnet `best_model_criterion` (list of [phase, metric, mode], keep union of top-N per
criterion, fp32-averaged). Early stopping is the custom AND: stop only when EVERY configured
criterion has gone `> patience` epochs without a new best. Checkpoints are atomic, rank-0, saved
at EPOCH boundaries every valid_interval_epoch; resume (auto = latest) restarts from the last
checkpointed epoch (e.g. preempted at 83 -> resume from the epoch-80 checkpoint). Resume restores
all state exactly; run-to-run bit-reproducibility is bounded by non-deterministic Mamba CUDA kernels
(a resumed run diverges no more than two identical straight runs do).
"""
from __future__ import annotations

import contextlib
import gc
import os
import random
import time
from pathlib import Path

import editdistance
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dcasr.logging_utils import get_logger
from dcasr.optim import build_optimizer, build_scheduler

logger = get_logger(__name__)


# ── distributed / reproducibility helpers ────────────────────────────────────
def init_distributed() -> tuple[int, int, int, bool]:
    """Read torchrun env; init the process group if WORLD_SIZE>1."""
    ws = int(os.environ.get("WORLD_SIZE", 1))
    if ws <= 1:
        return 1, 0, 0, False
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return ws, rank, local, True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())               # map_location may have moved it to GPU
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def word_errors(hyp: str, ref: str) -> tuple[int, int]:
    """Word-level (edits, #reference-words) for WER = Σedits / Σref-words."""
    rw = ref.split()
    return editdistance.eval(hyp.split(), rw), len(rw)


def char_errors(hyp: str, ref: str) -> tuple[int, int]:
    """Char-level (edits, #reference-chars) for CER (spaces stripped)."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    return editdistance.eval(h, r), len(r)


class Trainer:
    """Config-driven, model-agnostic training loop. `model` must return (loss, stats)."""

    def __init__(self, model, train_loader, cfg, *, dev_loaders=None, train_sampler=None,
                 tokenizer=None, metrics=None, device="cuda", ckpt_dir="checkpoints",
                 world_size=1, rank=0, provenance=None):
        self.raw_model = model.to(device)
        self.world_size, self.rank, self.is_main = world_size, rank, rank == 0
        # broadcast_buffers=False is LOAD-BEARING: the only buffer is a constant
        # positional encoding, and a buffer broadcast inside the OOM-recovery
        # forward would emit an extra collective on the OOM rank only (deadlock).
        self.model = (DDP(self.raw_model, device_ids=[torch.cuda.current_device()],
                          broadcast_buffers=False,
                          find_unused_parameters=cfg.get("find_unused_parameters", False))
                      if world_size > 1 else self.raw_model)
        self.train_loader = train_loader
        self.dev_loaders = dict(dev_loaders or {})       # {split_name: loader}
        self.train_sampler = train_sampler
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = dict(cfg)
        self.metrics = metrics
        self.provenance = provenance
        self.ckpt_dir = Path(ckpt_dir)
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        g = self.cfg.get
        self.max_epoch = int(g("max_epoch", 120))
        self.grad_clip = float(g("grad_clip", 5.0))
        self.grad_clip_type = float(g("grad_clip_type", 2.0))
        self.accum_grad = max(1, int(g("accum_grad", 1)))
        self.log_interval = int(g("log_interval", 50))
        self.valid_interval = int(g("valid_interval_epoch", 10))   # eval + checkpoint cadence (epochs)
        self.keep_nbest = int(g("keep_nbest_models", 5))
        self.keep_all_checkpoints = bool(g("keep_all_checkpoints", False))  # H4 emergence curves
        self.max_steps = g("max_steps")
        self.best_model_criterion = [tuple(c) for c in g("best_model_criterion",
                                                         [["valid", "loss", "min"]])]
        self.early_stopping = dict(g("early_stopping", {}) or {})

        self.precision = g("precision", "bf16")
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[self.precision]
        self.device_type = str(device).split(":")[0]     # "cuda:N" (torchrun) -> "cuda"
        self.scaler = torch.amp.GradScaler(self.device_type if self.device_type == "cuda" else "cpu",
                                           enabled=self.precision == "fp16")
        oc = dict(g("optim_conf", {}) or {})
        router_mult = float(oc.pop("router_lr_mult", 1.0))
        router_eps = oc.pop("router_eps", None)
        named = list(self.raw_model.named_parameters())

        def _is_router(n):
            parts = n.split(".")
            return "router" in parts and any(w in parts for w in ("W_q", "W_k"))

        # Weight-decay hygiene (runlog 2026-07-20): decay only >=2-D weight
        # matrices. 1-D params (biases, LayerNorm/RMSNorm/QK-norm gains) and the
        # Mamba SSM parameters A_log/D/dt_bias (tagged `_no_weight_decay`; decaying
        # them perturbs the SSM recurrent eigenvalues) get weight_decay=0. Router
        # W_q/W_k (2-D) get their own damped-lr group.
        wd = float(oc.get("weight_decay", 0.0))
        router = [p for n, p in named if _is_router(n)]
        rest = [(n, p) for n, p in named if not _is_router(n)]
        router_active = bool(router) and (router_mult != 1.0 or router_eps is not None)

        def _no_decay(p):
            return p.ndim < 2 or getattr(p, "_no_weight_decay", False)

        groups = []
        if wd > 0:
            groups.append({"params": [p for _, p in rest if not _no_decay(p)]})
            groups.append({"params": [p for _, p in rest if _no_decay(p)],
                           "weight_decay": 0.0})
        else:
            groups.append({"params": [p for _, p in rest]})
        if router_active:
            rg = {"params": router, "lr": float(oc.get("lr", 1e-3)) * router_mult}
            if router_eps is not None:
                rg["eps"] = float(router_eps)
            groups.append(rg)                          # router W_q/W_k: 2-D -> decays
        elif router:
            groups[0]["params"] = groups[0]["params"] + router   # 2-D -> decay group
        n_nd = sum(len(gr["params"]) for gr in groups if gr.get("weight_decay") == 0.0)
        logger.info("optim groups: %d (wd=%s, no-decay tensors=%d, router group=%s)",
                    len(groups), wd, n_nd, router_active)
        self.optimizer = build_optimizer(groups, g("optim", "adamw"), oc)
        self.scheduler = build_scheduler(self.optimizer, g("scheduler"), dict(g("scheduler_conf", {}) or {}))

        self.epoch, self.global_step = 0, 0
        self.oom_skips = 0                           # batches skipped by the OOM guard
        self.metric_history: dict[tuple[str, str], dict[int, float]] = {}
        logger.info("Trainer: world=%d precision=%s accum_grad=%d optim=%s sched=%s dev_splits=%s",
                    world_size, self.precision, self.accum_grad, g("optim", "adamw"),
                    g("scheduler"), list(self.dev_loaders))

    # ---- helpers ------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device_type, dtype=self.amp_dtype)

    def _reduce(self, values: list[float]) -> list[float]:
        """Sum a list of scalars across DDP ranks (identity on 1 GPU)."""
        if self.world_size > 1:
            t = torch.tensor(values, dtype=torch.float64, device=self.device)
            dist.all_reduce(t)
            return t.tolist()
        return list(values)

    def _any_rank_oom(self, oom_local: bool) -> bool:
        """Group OOM flag (identity on 1 GPU). Every rank calls this exactly once
        per micro-batch, so the collective stays matched: a rank-local skip would
        desync DDP's gradient buckets and corrupt or hang the job."""
        if self.world_size > 1:
            t = torch.tensor([1.0 if oom_local else 0.0], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item() > 0)
        return oom_local

    def _oom_recovery_step(self, batch: dict) -> None:
        """After a local forward OOM under DDP: forward+backward a minimal slice
        of the batch so this rank still joins the group's gradient collectives
        (the group skip zeroes every rank's grads right after)."""
        feats = batch["feats"][:1, :32]
        mini = self._to_device({
            "feats": feats,
            "feat_lens": torch.clamp(batch["feat_lens"][:1], max=feats.shape[1]),
            "tokens": batch["tokens"][:1, :1],
            "token_lens": torch.ones_like(batch["token_lens"][:1])})
        with self._autocast():
            loss, _ = self.model(mini["feats"], mini["feat_lens"],
                                 mini["tokens"], mini["token_lens"])
        self.scaler.scale(loss / self.accum_grad).backward()

    def _record(self, phase: str, metric: str, value: float) -> None:
        self.metric_history.setdefault((phase, metric), {})[self.epoch] = float(value)
        if self.is_main and self.metrics is not None:    # monitor values also persist to TB/JSONL
            self.metrics.log_scalar(f"{phase}/{metric}", float(value), self.global_step,
                                    split=phase, epoch=self.epoch)

    def _best_epoch(self, phase: str, metric: str, mode: str) -> int | None:
        hist = self.metric_history.get((phase, metric))
        if not hist:
            return None
        pick = min if mode == "min" else max
        return pick(hist, key=lambda e: hist[e])

    # ---- one epoch ----------------------------------------------------------
    def _train_epoch(self) -> None:
        self.raw_model.train()
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self.epoch)
        ds = getattr(self.train_loader, "dataset", None)     # deterministic augmentation per epoch
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(self.epoch)
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()                                     # per-epoch housekeeping (cheap here,
        if self.device_type == "cuda":                   # never per-step — throughput)
            torch.cuda.empty_cache()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)   # on-device: sync once/epoch
        weight_sum, seen, t0, micro = 0, 0, time.time(), 0
        win, win_n = {}, 0                               # stats summed over the accumulation window
        for batch in self.train_loader:
            oom_local = False
            try:
                batch = self._to_device(batch)
                b = int(batch["feats"].shape[0])
                with self._autocast():
                    loss, stats = self.model(batch["feats"], batch["feat_lens"],
                                             batch["tokens"], batch["token_lens"])
                    scaled = loss / self.accum_grad
            except torch.cuda.OutOfMemoryError:
                oom_local = True
                self.oom_skips += 1
                logger.warning("OOM in forward: skipping batch at step %d (%d skips so far)",
                               self.global_step, self.oom_skips)
                if self.device_type == "cuda":
                    torch.cuda.empty_cache()
            if self._any_rank_oom(oom_local):            # ALL ranks drop this window together
                if self.world_size > 1:
                    if oom_local:
                        self._oom_recovery_step(batch)   # join the group's backward collectives
                    else:
                        self.scaler.scale(scaled).backward()   # complete DDP reduction, discard
                        del loss, stats, scaled
                self.optimizer.zero_grad(set_to_none=True)
                del batch
                win, win_n = {}, 0
                micro = (micro // self.accum_grad) * self.accum_grad
                if self.device_type == "cuda":
                    torch.cuda.empty_cache()
                continue
            try:
                self.scaler.scale(scaled).backward()
            except torch.cuda.OutOfMemoryError:
                if self.world_size > 1:                  # bucket collectives already in flight:
                    raise RuntimeError(                  # not skippable rank-locally
                        "OOM during DDP backward — relaunch resumes from the last "
                        "checkpoint (--resume auto)") from None
                self.oom_skips += 1
                logger.warning("OOM in backward: skipping batch at step %d (%d skips so far)",
                               self.global_step, self.oom_skips)
                self.optimizer.zero_grad(set_to_none=True)
                del batch
                win, win_n = {}, 0
                micro = (micro // self.accum_grad) * self.accum_grad
                if self.device_type == "cuda":
                    torch.cuda.empty_cache()
                continue
            w = float(stats.get("batch_weight", b))  # model-provided weight (e.g. LM tokens) or rows
            loss_sum += stats["loss/total"].detach().double() * w   # stays on device
            for k, v in stats.items():
                win[k] = win.get(k, 0.0) + v
            win_n += 1
            weight_sum += w
            seen += b
            micro += 1
            if micro % self.accum_grad != 0:
                continue
            self.scaler.unscale_(self.optimizer)
            gnorm = torch.nn.utils.clip_grad_norm_(
                self.raw_model.parameters(),
                self.grad_clip if self.grad_clip > 0 else float("inf"),
                norm_type=self.grad_clip_type)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            applied_lr = self.optimizer.param_groups[0]["lr"]   # LR actually used for THIS step
            if self.scheduler is not None:
                self.scheduler.step()                           # advances LR for the NEXT step
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            if self.is_main and self.global_step % self.log_interval == 0:
                step_stats = {k: v / win_n for k, v in win.items()}   # window mean == this step
                self._log_train(step_stats, gnorm, seen, time.time() - t0, applied_lr)
                seen, t0 = 0, time.time()
            win, win_n = {}, 0
            if self.max_steps and self.global_step >= self.max_steps:
                break
        # epoch-mean train loss (DDP-reduced), recorded for best-model/early-stop
        s, w = self._reduce([loss_sum, weight_sum])
        self._record("train", "loss", s / max(1.0, w))

    def _log_train(self, stats, gnorm, seen, dt, lr) -> None:
        if self.metrics is None:
            return
        payload = {k: float(v) for k, v in stats.items()}
        payload["train/lr"] = float(lr)                      # LR applied to this step (not lookahead)
        payload["train/grad_norm"] = float(gnorm)
        payload["train/samples_per_s"] = seen / dt if dt > 0 else 0.0
        payload["sys/oom_skips"] = self.oom_skips
        if self.device_type == "cuda" and torch.cuda.is_available():
            payload["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()         # per-interval peak, not run-lifetime
        self.metrics.log_scalars(payload, self.global_step, split="train", epoch=self.epoch)

    # ---- validation over ALL dev splits -------------------------------------
    @torch.no_grad()
    def validate(self) -> dict:
        self.raw_model.eval()
        per_split, agg_loss, agg_wer, agg_cer, n_split = {}, [], [], [], 0
        for name, loader in self.dev_loaders.items():
            ls = torch.zeros((), device=self.device, dtype=torch.float64)   # on-device: sync once/split
            ws_, werr, wtot, cerr, ctot, skips = 0, 0, 0, 0, 0, 0
            for batch in loader:
                try:
                    batch = self._to_device(batch)
                    b = int(batch["feats"].shape[0])
                    with self._autocast():
                        loss, vstats = self.model(batch["feats"], batch["feat_lens"],
                                                  batch["tokens"], batch["token_lens"])
                    w = float(vstats.get("batch_weight", b))
                    d_we = d_wc = d_ce = d_cc = 0
                    if self.tokenizer is not None:
                        hyps = self.raw_model.greedy_decode(batch["feats"], batch["feat_lens"])
                        for j, hyp_ids in enumerate(hyps):
                            ref_ids = batch["tokens"][j, :int(batch["token_lens"][j])].tolist()
                            hyp = self.tokenizer.decode(hyp_ids)
                            ref = self.tokenizer.decode(ref_ids)
                            we, wc = word_errors(hyp, ref)
                            ce, cc = char_errors(hyp, ref)
                            d_we += we; d_wc += wc; d_ce += ce; d_cc += cc
                    # commit only after the WHOLE batch succeeded: a decode OOM
                    # must not leave the loss counted but WER/CER skipped
                    ls += loss.detach().double() * w
                    ws_ += w
                    werr += d_we; wtot += d_wc; cerr += d_ce; ctot += d_cc
                except torch.cuda.OutOfMemoryError:      # rank-local skip is safe here:
                    skips += 1                           # collectives run once per split
                    logger.warning("OOM in validation (%s): batch skipped (%d this split; "
                                   "dev metric slightly biased)", name, skips)
                    if self.device_type == "cuda":
                        torch.cuda.empty_cache()
                    continue
            ls, ws_, werr, wtot, cerr, ctot, skips = self._reduce(
                [ls.item(), float(ws_), werr, wtot, cerr, ctot, float(skips)])
            if ws_ <= 0:                                 # an all-OOM split must fail
                raise RuntimeError(f"validation split {name}: every batch was "
                                   "OOM-skipped — 0.0 metrics would corrupt "
                                   "best-model selection")
            m = {"loss": ls / ws_}
            if self.tokenizer is not None and wtot > 0:
                m["wer"] = 100.0 * werr / wtot
            if self.tokenizer is not None and ctot > 0:
                m["cer"] = 100.0 * cerr / ctot
            if skips:
                m["oom_skips"] = int(skips)              # summed across ranks
            per_split[name] = m
            agg_loss.append(m["loss"])
            if "wer" in m:
                agg_wer.append(m["wer"])
            if "cer" in m:
                agg_cer.append(m["cer"])
            n_split += 1
            if self.is_main and self.metrics is not None:
                self.metrics.log_scalars({f"dev_{name}/{k}": v for k, v in m.items()},
                                         self.global_step, split=name, epoch=self.epoch)
        # aggregates (mean over dev splits) drive best-model / early-stop
        self._record("valid", "loss", sum(agg_loss) / max(1, len(agg_loss)))
        if agg_wer:
            self._record("valid", "wer", sum(agg_wer) / len(agg_wer))
        if agg_cer:
            self._record("valid", "cer", sum(agg_cer) / len(agg_cer))
        self.raw_model.train()
        return per_split

    # ---- checkpoint selection + early stop ----------------------------------
    def _update_best_symlinks(self) -> None:
        if not self.is_main:
            return
        for phase, metric, mode in self.best_model_criterion:
            if self._best_epoch(phase, metric, mode) == self.epoch:
                link = self.ckpt_dir / f"{phase}.{metric}.best.pt"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(f"epoch{self.epoch:04d}.pt")

    def _prune_checkpoints(self) -> None:
        if not self.is_main or self.keep_nbest <= 0 or self.keep_all_checkpoints:
            return
        keep = {self.epoch}                              # always keep latest epoch (resume)
        for phase, metric, mode in self.best_model_criterion:
            hist = self.metric_history.get((phase, metric), {})
            top = sorted(hist, key=lambda e: hist[e], reverse=(mode == "max"))[:self.keep_nbest]
            keep.update(top)
        for p in self.ckpt_dir.glob("epoch*.pt"):
            e = int(p.stem[5:])
            if e not in keep:
                p.unlink(missing_ok=True)

    def _last_significant_best(self, phase: str, metric: str, mode: str,
                               min_delta: float) -> int | None:
        """Epoch of the last SIGNIFICANT best, i.e. one that beat the running
        best by more than `min_delta` (Keras convention: a sub-threshold move
        neither resets patience nor moves the reference).

        Separate from `_best_epoch`, which stays a plain argmin/argmax because
        CHECKPOINT selection must always track the true best. Early stopping
        needs the thresholded version: without min_delta, noise-sized
        'improvements' reset patience forever — our valid/wer improved 6.118 ->
        6.098 across 35 epochs (0.02, a quarter of its ~0.07 noise sd) and kept
        the run alive indefinitely (runlog 2026-07-21).
        """
        hist = self.metric_history.get((phase, metric))
        if not hist:
            return None
        best_ep = best_val = None
        for e in sorted(hist):
            v = hist[e]
            better = (best_val is None
                      or (v < best_val - min_delta if mode == "min"
                          else v > best_val + min_delta))
            if better:
                best_ep, best_val = e, v
        return best_ep

    def _should_early_stop(self) -> bool:
        es = self.early_stopping
        if not es.get("enable", False):
            return False
        criteria = es.get("criteria", [])
        results = []
        for c in criteria:
            be = self._last_significant_best(c["phase"], c["metric"],
                                             c.get("mode", "min"),
                                             float(c.get("min_delta", 0.0)))
            results.append(be is not None and (self.epoch - be) > int(c["patience"]))
        if not results:
            return False
        stop = all(results) if es.get("require_all", True) else any(results)
        if stop:
            logger.info("early stop at epoch %d (criteria stalled: %s)", self.epoch, results)
        return stop

    def _average_nbest(self) -> None:
        """fp32-average the top-N epochs per criterion into {phase}.{metric}.ave.pt."""
        if not self.is_main or self.keep_nbest <= 0:
            return
        for phase, metric, mode in self.best_model_criterion:
            hist = self.metric_history.get((phase, metric), {})
            top = sorted(hist, key=lambda e: hist[e], reverse=(mode == "max"))[:self.keep_nbest]
            paths = [self.ckpt_dir / f"epoch{e:04d}.pt" for e in top]
            paths = [p for p in paths if p.exists()]
            if not paths:
                continue
            averaged = [e for e in top if (self.ckpt_dir / f"epoch{e:04d}.pt").exists()]
            avg = None
            for p in paths:
                sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
                if avg is None:
                    avg = {k: v.float().clone() for k, v in sd.items()}
                else:
                    for k in avg:
                        avg[k] += sd[k].float()
            for k in avg:
                avg[k] /= len(paths)
            self._atomic_save({"model": avg, "averaged_epochs": averaged},
                              self.ckpt_dir / f"{phase}.{metric}.ave.pt")
            logger.info("averaged %d ckpts -> %s.%s.ave.pt", len(paths), phase, metric)

    # ---- checkpoints --------------------------------------------------------
    def save_checkpoint(self) -> None:
        """Epoch-boundary checkpoint (rank-0, atomic): the saved epoch is COMPLETE."""
        if not self.is_main:
            return
        state = {"model": self.raw_model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                 "scaler": self.scaler.state_dict(), "epoch": self.epoch,
                 "global_step": self.global_step, "metric_history": self.metric_history,
                 "config": self.cfg, "rng": _rng_state()}
        self._atomic_save(state, self.ckpt_dir / f"epoch{self.epoch:04d}.pt")
        self._atomic_save(state, self.ckpt_dir / "latest.pt")

    @staticmethod
    def _atomic_save(state: dict, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def _resolve_resume(self, resume) -> Path | None:
        if resume in (None, ""):
            return None
        if resume == "auto":
            latest = self.ckpt_dir / "latest.pt"
            if latest.exists():
                return latest
            eps = sorted(self.ckpt_dir.glob("epoch*.pt"))
            return eps[-1] if eps else None
        p = Path(resume)
        if not p.exists():                               # never silently restart from scratch
            raise FileNotFoundError(f"--resume checkpoint not found: {p}")
        return p

    def load_checkpoint(self, path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.metric_history = state.get("metric_history", {})
        _set_rng_state(state["rng"])
        self.epoch = state["epoch"] + 1                  # saved epoch completed -> continue at next
        logger.info("resumed %s -> continue at epoch %d (step %d)",
                    path, self.epoch, self.global_step)

    # ---- driver -------------------------------------------------------------
    def train(self, resume=None) -> None:
        if self.is_main and self.metrics is not None and self.provenance is not None:
            self.metrics.append_summary("provenance", self.provenance)   # at start: survives early crash
        ck = self._resolve_resume(resume)
        if ck is not None and ck.exists():
            self.load_checkpoint(ck)
        for epoch in range(self.epoch, self.max_epoch):
            if self.max_steps and self.global_step >= self.max_steps:
                break                                    # resumed run already at budget: no extra step
            self.epoch = epoch
            self._train_epoch()
            final = (epoch + 1) == self.max_epoch
            save_now = (epoch + 1) % self.valid_interval == 0 or final   # every N epochs (+ final)
            if save_now and self.dev_loaders:
                per_split = self.validate()
                if self.is_main:
                    logger.info("epoch %d valid: %s", epoch,
                                {k: {m: round(x, 3) for m, x in v.items()}
                                 for k, v in per_split.items()})
            if save_now:
                self.save_checkpoint()
                if self.dev_loaders:
                    self._update_best_symlinks()         # after save: target exists, no dangling link
                self._prune_checkpoints()
                if self.dev_loaders and self._should_early_stop():
                    break
            if self.max_steps and self.global_step >= self.max_steps:
                if not save_now:                         # a max_steps exit still leaves a checkpoint
                    self.save_checkpoint()
                break
        self._average_nbest()
        if self.is_main and self.metrics is not None:
            summary = {"final_step": self.global_step, "epochs": self.epoch + 1,
                       "world_size": self.world_size, "config": self.cfg}
            for phase, metric, mode in self.best_model_criterion:
                be = self._best_epoch(phase, metric, mode)
                if be is not None:
                    summary[f"best_{phase}_{metric}"] = self.metric_history[(phase, metric)][be]
                    summary[f"best_{phase}_{metric}_epoch"] = be
            self.metrics.update_summary(**summary)
