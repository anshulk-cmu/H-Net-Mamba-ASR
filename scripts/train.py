"""Entry point: train one DC-ASR model from a YAML config.

Thin builder — all logic lives in tested library code. Loads + merges the config,
sets the seed, initialises DDP, builds the tokenizer / frontend / CMVN / SpecAugment /
data loaders (tasks.build) and the model (tasks.asr_task.build_model), captures run
provenance, then hands everything to the model-agnostic Trainer.

Usage:
    python scripts/train.py --config configs/typeA_small_N1_ctc.yaml
    python scripts/train.py --config <cfg> --resume auto              # resume latest ckpt
    python scripts/train.py --config <cfg> train.max_epoch=1 train.max_steps=2   # dotlist overrides
Multi-GPU (constant global batch via the sampler budget):
    torchrun --nproc_per_node=4 scripts/train.py --config <cfg>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from dcasr.data.tokenizer import Tokenizer
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.metrics_logger import MetricsLogger
from dcasr.provenance import collect_provenance
from dcasr.tasks.asr_task import build_model
from dcasr.tasks.build import (build_cmvn, build_dataloaders, build_frontend,
                               build_specaugment, flatten_config, resolve_manifests)
from dcasr.training.trainer import Trainer, init_distributed, set_seed

REPO = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def load_config(config_path, overrides):
    """Load the YAML and merge trailing `key=value` dotlist overrides."""
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run(cfg, resume=None, repo_root=REPO):
    """Build every component from `cfg` and run training. Returns the Trainer."""
    world_size, rank, local, is_dist = init_distributed()
    seed = int(cfg.experiment.seed)
    set_seed(seed)
    run_name = str(cfg.experiment.name)

    device = (f"cuda:{local}" if is_dist else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = repo_root / "checkpoints" / run_name
    # append to metrics only when a checkpoint will ACTUALLY be resumed — 'auto' with no
    # checkpoint is a fresh start (else a crash-before-first-save relaunch duplicates records)
    resuming = bool(resume) and (resume != "auto" or (ckpt_dir / "latest.pt").exists()
                                 or any(ckpt_dir.glob("epoch*.pt")))
    metrics = MetricsLogger(run_name, rank=rank, resume=resuming)

    tokenizer = Tokenizer(repo_root / cfg.bpemodel)
    frontend = build_frontend(cfg)
    cmvn = build_cmvn(cfg, repo_root)
    specaug = build_specaugment(cfg)
    train_loader, train_sampler, dev_loaders = build_dataloaders(
        cfg, repo_root, tokenizer, frontend, cmvn=cmvn, specaugment=specaug,
        world_size=world_size, rank=rank, seed=seed)

    model = build_model(cfg, tokenizer.vocab_size)

    provenance = None
    if rank == 0:                                        # only rank 0 persists it
        train_manifest, dev_manifests = resolve_manifests(cfg, repo_root)
        provenance = collect_provenance(
            cfg, repo_root=repo_root, world_size=world_size, seed=seed,
            manifests=[train_manifest, *dev_manifests.values()],
            extra={"run_name": run_name, "resume_arg": resume, "device": device})

    flat = flatten_config(cfg)
    trainer = Trainer(model, train_loader, flat, dev_loaders=dev_loaders,
                      train_sampler=train_sampler, tokenizer=tokenizer, metrics=metrics,
                      device=device, ckpt_dir=ckpt_dir,
                      world_size=world_size, rank=rank, provenance=provenance)
    try:
        trainer.train(resume=resume)
    finally:
        metrics.close()
        if is_dist and dist.is_initialized():
            dist.destroy_process_group()
    return trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to the experiment YAML")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path, or 'auto' for the latest in checkpoints/<run>")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. train.max_epoch=1")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(f"train_{cfg.experiment.name}")
    logger.info("config=%s resume=%s overrides=%s", args.config, args.resume, args.overrides)
    run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
