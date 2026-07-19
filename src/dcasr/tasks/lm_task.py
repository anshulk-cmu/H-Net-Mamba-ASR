"""LM task builders: config -> TransformerLM wired for the model-agnostic Trainer.

`LMModel` adapts `TransformerLM.loss(tokens, lengths)` to the Trainer contract
`forward(feats, feat_lens, targets, target_lens) -> (loss, stats)` (the LM batch
carries tokens in both slots; feats/feat_lens are ignored). CUDA-free imports —
the LM is pure torch, so this task trains and tests without the Mamba stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from dcasr.data.lm_text import LMTextDataset, make_lm_dataloader
from dcasr.decoders.lm_fusion import TransformerLM
from dcasr.logging_utils import get_logger
from dcasr.tasks.build import _plain, _resolve, resolve_manifests

logger = get_logger(__name__)


class LMModel(nn.Module):
    """Trainer-contract wrapper: loss = next-token CE over the target tokens."""

    def __init__(self, lm: TransformerLM):
        super().__init__()
        self.lm = lm

    def forward(self, feats, feat_lens, targets, target_lens):
        loss, acc = self.lm.loss(targets, target_lens, return_acc=True)
        # batch_weight = scored tokens (+1 eos/line): the Trainer weights loss aggregation
        # by it, so exp(valid/loss) is the TRUE token-weighted dev perplexity
        stats = {"loss/total": loss.detach(), "lm/token_acc": acc,
                 "batch_weight": (target_lens + 1).sum().detach()}
        return loss, stats


def build_lm(config: Mapping[str, Any], vocab_size: int) -> TransformerLM:
    lc = _plain(config).get("lm_conf", {}) or {}
    return TransformerLM(int(vocab_size), d_model=int(lc.get("d_model", 512)),
                         n_layers=int(lc.get("n_layers", 8)), n_heads=int(lc.get("n_heads", 8)),
                         d_ff=int(lc.get("d_ff", 2048)), dropout=float(lc.get("dropout", 0.1)),
                         lsm_weight=float(lc.get("lsm_weight", 0.0)))


def build_lm_dataloaders(cfg, repo_root, tokenizer, *, world_size=1, rank=0, seed=0):
    """Train loader on the raw LM corpus + one dev loader per dev split (manifest text)."""
    c = _plain(cfg)
    data = c.get("data", {}) or {}
    batch_tokens = int(c["batch_tokens"])
    num_workers = int(c.get("num_workers", 2))
    max_tokens = int((c.get("lm_conf", {}) or {}).get("max_line_tokens", 512))

    corpus = _resolve(data["lm_corpus"], Path(repo_root))
    train_ds = LMTextDataset(tokenizer, corpus_path=corpus, max_tokens=max_tokens)
    train_loader, train_sampler = make_lm_dataloader(
        train_ds, batch_tokens, shuffle=True, num_workers=num_workers, seed=seed,
        world_size=world_size, rank=rank)

    _, dev_manifests = resolve_manifests(c, repo_root)
    dev_loaders = {}
    for name, mpath in dev_manifests.items():            # dev is never sharded (Phase-0 F2)
        ds = LMTextDataset.from_manifest(mpath, tokenizer, max_tokens=max_tokens)
        loader, _ = make_lm_dataloader(ds, batch_tokens, shuffle=False,
                                       num_workers=num_workers, seed=seed,
                                       world_size=1, rank=0)
        dev_loaders[name] = loader
    logger.info("lm dataloaders: train=%d batches (%s), dev=%s", len(train_sampler),
                corpus.name, {k: len(v.batch_sampler) for k, v in dev_loaders.items()})
    return train_loader, train_sampler, dev_loaders
