"""Hybrid DC-ASR training loss (plan §4.4): weighted CTC + AED + ratio.

    total = w_ctc · CTC + w_aed · AED + w_ratio · Σ_stage ratio

The recognition heads compute their own scalar losses (CTCHead.loss; AED later) and
the encoder returns the already-summed ratio loss; HybridLoss just weights and adds
them, returning the raw per-component values for metric logging. Weights are the three
independent config knobs (the plan's λ_ctc / (1−λ_ctc) split is w_ctc / w_aed here);
the go/no-go run is CTC-only (w_ctc=1, w_aed=0, w_ratio=0 with N=1 → ratio≡0).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LossOutput:
    total: torch.Tensor        # weighted sum — the scalar to backprop
    ctc: torch.Tensor          # raw (unweighted) CTC loss
    aed: torch.Tensor          # raw (unweighted) AED loss  (0 when unused)
    ratio: torch.Tensor        # raw summed ratio loss      (0 at N=1)

    def items(self, prefix: str = "loss") -> dict[str, torch.Tensor]:
        """Namespaced raw components for MetricsLogger.log_scalars."""
        return {f"{prefix}/total": self.total, f"{prefix}/ctc": self.ctc,
                f"{prefix}/aed": self.aed, f"{prefix}/ratio": self.ratio}


class HybridLoss(nn.Module):
    """Weighted combination of the CTC, AED, and ratio losses (plan §4.4)."""

    def __init__(self, ctc_weight: float = 1.0, aed_weight: float = 0.0,
                 ratio_weight: float = 0.0):
        super().__init__()
        self.ctc_weight = float(ctc_weight)
        self.aed_weight = float(aed_weight)
        self.ratio_weight = float(ratio_weight)
        logger.debug("HybridLoss(w_ctc=%.3g, w_aed=%.3g, w_ratio=%.3g)",
                     self.ctc_weight, self.aed_weight, self.ratio_weight)

    @classmethod
    def from_config(cls, loss_cfg) -> "HybridLoss":
        """Build from a config `train.loss` block (dict / OmegaConf)."""
        g = loss_cfg.get
        return cls(g("ctc_weight", 1.0), g("aed_weight", 0.0), g("ratio_weight", 0.0))

    def forward(self, *, ctc_loss: torch.Tensor | None = None,
                aed_loss: torch.Tensor | None = None,
                ratio_loss: torch.Tensor | None = None) -> LossOutput:
        provided = [t for t in (ctc_loss, aed_loss, ratio_loss) if t is not None]
        if not provided:
            raise ValueError("HybridLoss.forward got no loss components")
        zero = provided[0].new_zeros(())
        # a positive weight with no matching component is a wiring bug — fail loudly
        for name, loss, w in (("ctc", ctc_loss, self.ctc_weight),
                              ("aed", aed_loss, self.aed_weight),
                              ("ratio", ratio_loss, self.ratio_weight)):
            if w > 0.0 and loss is None:
                raise ValueError(f"{name}_weight={w} but no {name}_loss was provided")
        ctc = ctc_loss if ctc_loss is not None else zero
        aed = aed_loss if aed_loss is not None else zero
        ratio = ratio_loss if ratio_loss is not None else zero
        total = self.ctc_weight * ctc + self.aed_weight * aed + self.ratio_weight * ratio
        return LossOutput(total=total, ctc=ctc, aed=aed, ratio=ratio)
