"""Entry point: train the external Transformer LM from a YAML config (plan #6).

Mirrors scripts/train.py: thin builder over tested library code. The LM reuses the
model-agnostic Trainer (tokenizer=None => loss-only validation on the dev transcripts).

Usage:
    python scripts/train_lm.py --config configs/lm_transformer_500.yaml
    python scripts/train_lm.py --config <cfg> --resume auto
    python scripts/train_lm.py --config <cfg> train.max_epoch=1 batch_tokens=4000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch.distributed as dist

from dcasr.data.tokenizer import Tokenizer
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.metrics_logger import MetricsLogger
from dcasr.provenance import collect_provenance
from dcasr.tasks.build import flatten_config, resolve_manifests
from dcasr.tasks.lm_task import LMModel, build_lm, build_lm_dataloaders
from dcasr.training.trainer import Trainer, init_distributed, set_seed

REPO = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def load_config(config_path, overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run(cfg, resume=None, repo_root=REPO):
    """Build every component from `cfg` and train the LM. Returns the Trainer."""
    world_size, rank, local, is_dist = init_distributed()
    seed = int(cfg.experiment.seed)
    set_seed(seed)
    run_name = str(cfg.experiment.name)

    import torch
    device = (f"cuda:{local}" if is_dist else
              ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = repo_root / "checkpoints" / run_name
    resuming = bool(resume) and (resume != "auto" or (ckpt_dir / "latest.pt").exists()
                                 or any(ckpt_dir.glob("epoch*.pt")))
    metrics = MetricsLogger(run_name, rank=rank, resume=resuming)

    tokenizer = Tokenizer(repo_root / cfg.bpemodel)
    train_loader, train_sampler, dev_loaders = build_lm_dataloaders(
        cfg, repo_root, tokenizer, world_size=world_size, rank=rank, seed=seed)
    model = LMModel(build_lm(cfg, tokenizer.vocab_size))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("TransformerLM params=%.1fM vocab=%d", n_params / 1e6, tokenizer.vocab_size)

    provenance = None
    if rank == 0:
        corpus = Path(repo_root) / str(cfg.data.lm_corpus)
        _, dev_manifests = resolve_manifests(cfg, repo_root)
        provenance = collect_provenance(
            cfg, repo_root=repo_root, world_size=world_size, seed=seed,
            manifests=list(dev_manifests.values()), extra_files={"lm_corpus": corpus},
            extra={"run_name": run_name, "resume_arg": resume, "device": device})

    trainer = Trainer(model, train_loader, flatten_config(cfg), dev_loaders=dev_loaders,
                      train_sampler=train_sampler, tokenizer=None, metrics=metrics,
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
    ap.add_argument("--config", required=True, help="path to the LM YAML")
    ap.add_argument("--resume", default=None, help="checkpoint path, or 'auto'")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(f"train_lm_{cfg.experiment.name}")
    logger.info("config=%s resume=%s overrides=%s", args.config, args.resume, args.overrides)
    run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
