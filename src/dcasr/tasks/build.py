"""Run-assembly seam: resolved config -> frontend / augmentation / data loaders + the flat
Trainer config. Kept CUDA-model-free (features/librispeech/tokenizer only) so the data path
is importable and testable without the Mamba stack; `build_model` lives in asr_task.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dcasr.data.features import GlobalCMVN, LogMelFrontend, SpecAugment
from dcasr.data.librispeech import LibriSpeechDataset, make_dataloader
from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def _plain(cfg: Any) -> Any:
    """OmegaConf -> plain (resolved) container; pass-through for dict/list."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg


def _resolve(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


# ── flat Trainer config ───────────────────────────────────────────────────────
def flatten_config(cfg: Mapping[str, Any]) -> dict:
    """Map the nested YAML to the flat keys the Trainer reads (train.*/eval.* hoisted)."""
    c = _plain(cfg)
    train = c.get("train", {}) or {}
    ev = c.get("eval", {}) or {}
    flat = {
        "max_epoch": train.get("max_epoch", 120),
        "grad_clip": train.get("grad_clip", 5.0),
        "grad_clip_type": train.get("grad_clip_type", 2.0),
        "precision": train.get("precision", "bf16"),
        "log_interval": train.get("log_interval", 50),
        "max_steps": train.get("max_steps"),
        "accum_grad": c.get("accum_grad", 1),
        "valid_interval_epoch": ev.get("valid_interval_epoch", 10),
        "keep_nbest_models": c.get("keep_nbest_models", 5),
        "keep_all_checkpoints": c.get("keep_all_checkpoints", False),
        "best_model_criterion": c.get("best_model_criterion", [["valid", "loss", "min"]]),
        "early_stopping": c.get("early_stopping", {}) or {},
        "optim": c.get("optim", "adamw"),
        "optim_conf": c.get("optim_conf", {}) or {},
        "scheduler": c.get("scheduler"),
        "scheduler_conf": c.get("scheduler_conf", {}) or {},
        "find_unused_parameters": c.get("find_unused_parameters", False),
    }
    return flat


# ── frontend / CMVN / augmentation ────────────────────────────────────────────
def build_frontend(cfg: Mapping[str, Any]) -> LogMelFrontend:
    fc = _plain(cfg).get("frontend_conf", {}) or {}
    return LogMelFrontend(sample_rate=int(fc.get("sample_rate", 16000)),
                          n_mels=int(fc.get("n_mels", 80)),
                          win_length=int(fc.get("win_length", 400)),
                          hop_length=int(fc.get("hop_length", 160)))


def build_cmvn(cfg: Mapping[str, Any], repo_root: str | Path) -> GlobalCMVN | None:
    fc = _plain(cfg).get("frontend_conf", {}) or {}
    path = fc.get("cmvn")
    if not path:
        return None
    return GlobalCMVN.load(_resolve(path, Path(repo_root)))


def build_specaugment(cfg: Mapping[str, Any]) -> SpecAugment | None:
    """SpecAugment from specaug_conf. `time_mask_width_ratio_range` -> adaptive time masks;
    else `time_mask_width_range` -> fixed absolute width."""
    sc = _plain(cfg).get("specaug_conf")
    if not sc:
        return None
    freq_masks = int(sc.get("num_freq_mask", 2))
    freq_width = int((sc.get("freq_mask_width_range") or [0, 27])[1])
    time_masks = int(sc.get("num_time_mask", 2))
    ratio = sc.get("time_mask_width_ratio_range")
    if ratio is not None:
        return SpecAugment(freq_masks=freq_masks, freq_width=freq_width,
                           time_masks=time_masks, time_width_ratio=float(ratio[1]))
    time_width = int((sc.get("time_mask_width_range") or [0, 100])[1])
    return SpecAugment(freq_masks=freq_masks, freq_width=freq_width,
                       time_masks=time_masks, time_width=time_width)


# ── manifests / data loaders ──────────────────────────────────────────────────
def resolve_manifests(cfg: Mapping[str, Any], repo_root: str | Path) -> tuple[Path, dict[str, Path]]:
    """(train manifest path, {dev_split_name: manifest path}) from cfg.data."""
    data = _plain(cfg).get("data", {}) or {}
    mdir = _resolve(data.get("manifests_dir", "manifests"), Path(repo_root))
    train = mdir / f"{data.get('train_manifest', 'train-960')}.jsonl"
    dev = {name: mdir / f"{name}.jsonl" for name in data.get("dev_splits", [])}
    return train, dev


def build_dataloaders(cfg, repo_root, tokenizer, frontend, *, cmvn=None, specaugment=None,
                      world_size=1, rank=0, seed=0):
    """Train loader (+ its sampler, augmented) and one dev loader per dev split (no aug)."""
    c = _plain(cfg)
    batch_bins = int(c["batch_bins"])
    num_workers = int(c.get("num_workers", 4))
    speed = (c.get("train", {}) or {}).get("speed_perturb")   # e.g. [0.9, 1.0, 1.1]; train-only
    train_manifest, dev_manifests = resolve_manifests(c, repo_root)

    train_ds = LibriSpeechDataset(train_manifest, frontend, tokenizer, cmvn=cmvn,
                                  specaugment=specaugment, augment=True, seed=seed,
                                  speed_perturb=speed)
    train_loader, train_sampler = make_dataloader(train_ds, batch_bins, augment=True,
                                                  num_workers=num_workers, seed=seed,
                                                  world_size=world_size, rank=rank)
    dev_loaders = {}
    for name, mpath in dev_manifests.items():
        ds = LibriSpeechDataset(mpath, frontend, tokenizer, cmvn=cmvn, specaugment=None,
                                augment=False, seed=seed)
        # dev is NOT sharded: the DDP equal-count trim would drop the longest (last) batches,
        # biasing dev WER; every rank scores the full split, ratios survive the all-reduce
        loader, _ = make_dataloader(ds, batch_bins, augment=False, num_workers=num_workers,
                                    seed=seed, world_size=1, rank=0)
        dev_loaders[name] = loader
    logger.info("dataloaders: train=%d batches (%s), dev=%s", len(train_sampler),
                train_manifest.name, {k: len(v) for k, v in dev_loaders.items()})
    return train_loader, train_sampler, dev_loaders
