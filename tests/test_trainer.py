"""Unit tests for the config-driven Trainer (src/dcasr/training/trainer.py).

CPU-only: a tiny fake model with the (loss, stats) contract exercises the loop mechanics,
config-driven optim/scheduler, multi-dev-split validation, resume, keep-N-best, and the
early-stop AND logic — no Mamba/GPU needed.
"""
import pytest
import torch
import torch.nn as nn

from dcasr.metrics_logger import MetricsLogger
from dcasr.training.trainer import Trainer

FEAT = 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEAT, 8)

    def forward(self, feats, feat_lens, targets, target_lens):
        loss = self.lin(feats).pow(2).mean()
        z = torch.zeros(())
        stats = {"loss/total": loss.detach(), "loss/ctc": loss.detach(),
                 "loss/aed": z, "loss/ratio": z, "kept_fraction": torch.tensor(1.0)}
        return loss, stats


class _Sampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, e):
        self.epochs.append(e)


def _batch(B=2, T=5, U=3, V=6):
    return {"feats": torch.randn(B, T, FEAT), "feat_lens": torch.full((B,), T),
            "tokens": torch.randint(0, V, (B, U)), "token_lens": torch.full((B,), U),
            "ids": [f"u{i}" for i in range(B)]}


def _loader(n=4):
    return [_batch() for _ in range(n)]


def _cfg(**over):
    cfg = {"optim": "adamw", "optim_conf": {"lr": 0.05}, "scheduler": "warmuplr",
           "scheduler_conf": {"warmup_steps": 3}, "max_epoch": 2, "grad_clip": 1.0,
           "grad_clip_type": 2.0, "accum_grad": 1, "precision": "fp32", "log_interval": 1,
           "valid_interval_epoch": 1, "keep_nbest_models": 5,
           "best_model_criterion": [["valid", "loss", "min"]], "early_stopping": {"enable": False}}
    cfg.update(over)
    return cfg


def _trainer(tmp, cfg, dev=None, metrics=None):
    return Trainer(_Model(), _loader(), cfg, dev_loaders=dev, train_sampler=_Sampler(),
                   metrics=metrics, device="cpu", ckpt_dir=tmp / "ckpts")


def test_train_config_driven(tmp_path):
    tr = _trainer(tmp_path, _cfg(), metrics=MetricsLogger("run", root=tmp_path))
    assert isinstance(tr.optimizer, torch.optim.AdamW)
    assert tr.scheduler.base_lrs[0] == 0.05           # configured peak; warmuplr scales live lr
    w0 = tr.raw_model.lin.weight.detach().clone()
    tr.train()
    assert tr.global_step == 8                       # 4 batches * 2 epochs, accum 1
    assert not torch.equal(w0, tr.raw_model.lin.weight)
    assert tr.train_sampler.epochs == [0, 1]
    assert (tmp_path / "run" / "metrics.jsonl").exists()


def test_grad_accum(tmp_path):
    tr = _trainer(tmp_path, _cfg(accum_grad=2, max_epoch=1))
    tr.train()
    assert tr.global_step == 2                        # 4 batches / accum 2


def test_lr_schedule_advances(tmp_path):
    tr = _trainer(tmp_path, _cfg(max_epoch=1))
    tr.train()
    assert tr.scheduler is not None
    assert tr.scheduler.get_last_lr()[0] != 0.05      # warmuplr moved the LR off base


def test_validate_multi_dev_split(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=1),
                  dev={"dev-clean": _loader(2), "dev-other": _loader(2)}, metrics=ml)
    per_split = tr.validate()
    assert set(per_split) == {"dev-clean", "dev-other"}
    assert all("loss" in v for v in per_split.values())
    assert ("valid", "loss") in tr.metric_history     # aggregate recorded
    ml.close()
    keys = {r_key for r_key in _read_keys(tr, tmp_path)}
    assert "dev_dev-clean/loss" in keys and "dev_dev-other/loss" in keys


def _read_keys(tr, tmp_path):
    import json
    with open(tmp_path / "run" / "metrics.jsonl") as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def test_checkpoint_resume(tmp_path):
    tr = _trainer(tmp_path, _cfg(max_epoch=1))
    tr.train()
    step, params = tr.global_step, [p.detach().clone() for p in tr.raw_model.parameters()]
    tr2 = _trainer(tmp_path, _cfg(max_epoch=1))
    tr2.load_checkpoint(tr2._resolve_resume("auto"))
    assert tr2.global_step == step and tr2.epoch == 1          # epoch complete -> next epoch
    assert tr2.metric_history == tr.metric_history
    for a, b in zip(params, tr2.raw_model.parameters()):
        assert torch.equal(a, b)


def test_best_symlink_and_prune(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=3, keep_nbest_models=2),
                  dev={"dev-clean": _loader(2)}, metrics=ml)
    tr.train()
    assert (tr.ckpt_dir / "valid.loss.best.pt").exists()       # best symlink written
    assert len(list(tr.ckpt_dir.glob("epoch*.pt"))) <= 2 + 1   # top-2 + latest epoch


def test_early_stop_and_logic(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    tr.early_stopping = {"enable": True, "require_all": True, "criteria": [
        {"phase": "valid", "metric": "wer", "mode": "min", "patience": 2},
        {"phase": "train", "metric": "loss", "mode": "min", "patience": 2}]}
    tr.metric_history = {("valid", "wer"): {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0},
                         ("train", "loss"): {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0}}
    tr.epoch = 3
    assert tr._should_early_stop() is True                     # both stalled > patience 2
    tr.metric_history[("valid", "wer")][3] = 1.0               # valid wer improves at ep 3
    assert tr._should_early_stop() is False                    # AND: one not stalled -> continue
    tr.early_stopping["require_all"] = False
    assert tr._should_early_stop() is True                     # OR: train loss still stalled


def test_best_epoch_min_max(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    tr.metric_history = {("valid", "wer"): {0: 9.0, 1: 4.0, 2: 7.0}}
    assert tr._best_epoch("valid", "wer", "min") == 1
    assert tr._best_epoch("valid", "wer", "max") == 0
    assert tr._best_epoch("valid", "nope", "min") is None
