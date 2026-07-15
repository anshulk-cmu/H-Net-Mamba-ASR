"""Config-driven optimizer + LR-scheduler factories (ESPnet-style name + _conf registry).

`build_optimizer(params, name, conf)` and `build_scheduler(optimizer, name, conf)` select a
class by string name and spread the `_conf` dict as constructor kwargs — so experiments swap
optimizer/schedule from YAML with no code change. WarmupLR is ESPnet's warmuplr (linear warmup
→ inverse-sqrt decay), stepped once per optimizer step. Schedulers here are per-step.
"""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import _LRScheduler

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


class WarmupLR(_LRScheduler):
    """ESPnet warmuplr: lr = base_lr · warmup^0.5 · min(step^-0.5, step·warmup^-1.5).

    Linear ramp 0→base_lr over `warmup_steps`, then ∝ 1/sqrt(step); peak == base_lr at
    step == warmup_steps. `base_lr` is the optimizer's configured lr. Step once per optim step.
    """

    def __init__(self, optimizer, warmup_steps: int | float = 25000, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.warmup_steps ** 0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [base_lr * scale for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """Noam schedule: lr = model_size^-0.5 · min(step^-0.5, step·warmup^-1.5) (base_lr-agnostic)."""

    def __init__(self, optimizer, model_size: int, warmup_steps: int | float = 25000,
                 last_epoch: int = -1):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        scale = self.model_size ** -0.5 * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        return [scale for _ in self.base_lrs]


# ── registries (string name -> class) ────────────────────────────────────────
OPTIMIZERS = {
    "adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD,
    "adadelta": torch.optim.Adadelta, "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop, "radam": torch.optim.RAdam,
}
SCHEDULERS = {
    "warmuplr": WarmupLR, "noamlr": NoamLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
}


def build_optimizer(params, name: str = "adamw", conf: dict | None = None):
    """Instantiate a torch optimizer by name with `conf` kwargs (betas list -> tuple)."""
    key = name.lower()
    if key not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; choices: {sorted(OPTIMIZERS)}")
    kw = dict(conf or {})
    if "betas" in kw:
        kw["betas"] = tuple(kw["betas"])
    logger.info("optimizer=%s conf=%s", key, kw)
    return OPTIMIZERS[key](params, **kw)


def build_scheduler(optimizer, name: str | None = None, conf: dict | None = None):
    """Instantiate an LR scheduler by name with `conf` kwargs; None/'none' -> no scheduler."""
    if name in (None, "none", "None"):
        return None
    key = name.lower()
    if key not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; choices: {sorted(SCHEDULERS)}")
    logger.info("scheduler=%s conf=%s", key, dict(conf or {}))
    return SCHEDULERS[key](optimizer, **(conf or {}))
