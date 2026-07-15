"""ASR task builders: turn a resolved config into wired model objects.

The single seam between YAML and Python. `build_*` functions select a class by string
name via in-repo registries (ESPnet name+_conf style) and adapt the config block to the
constructor, so experiments swap encoder/head/loss/optimizer/scheduler from config alone.
`DCASRModel(encoder, head, loss)` assembles them via constructor injection; its
`forward(feats, feat_lens, targets, target_lens) -> (loss, stats)` is the model-agnostic
contract the Trainer depends on (it never imports a concrete encoder/head).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dcasr.decoders.ctc import CTCHead
from dcasr.logging_utils import get_logger
from dcasr.models.encoder import DCASREncoder
from dcasr.optim import build_optimizer, build_scheduler        # re-exported (single seam)
from dcasr.training.loss import HybridLoss

logger = get_logger(__name__)


# ── encoder ──────────────────────────────────────────────────────────────────
def _build_dcasr_encoder(config) -> nn.Module:
    ec = config["encoder_conf"]
    h = ec.get("hnet", {}) or {}
    return DCASREncoder(
        n_mels=int(config["frontend_conf"]["n_mels"]),
        d_outer=int(ec["d_outer"]), d_main=int(ec["d_main"]),
        n_enc=int(ec["n_enc"]), n_main=int(ec["n_main"]), n_dec=int(ec["n_dec"]),
        n_mid=int(ec.get("n_mid", 4)), arch_type=str(ec["arch_type"]),
        N=int(h.get("compression_N", 1)),
        bidirectional=bool(ec.get("bidirectional", True)),
        hnet_ema=bool(h.get("ema_smoothing", True)))


ENCODER_BUILDERS = {"dcasr": _build_dcasr_encoder}
HEAD_BUILDERS = {
    "ctc": lambda config, vocab_size: CTCHead(int(config["encoder_conf"]["d_outer"]),
                                              int(vocab_size)),
}


def build_encoder(config) -> nn.Module:
    name = str(config["encoder"]).lower()
    if name not in ENCODER_BUILDERS:
        raise ValueError(f"unknown encoder {name!r}; choices: {sorted(ENCODER_BUILDERS)}")
    return ENCODER_BUILDERS[name](config)


def build_head(config, vocab_size: int) -> nn.Module:
    name = str(config["head"]).lower()
    if name not in HEAD_BUILDERS:
        raise ValueError(f"unknown head {name!r}; choices: {sorted(HEAD_BUILDERS)}")
    return HEAD_BUILDERS[name](config, vocab_size)


def build_loss(config) -> HybridLoss:
    mc = config.get("model_conf", {}) or {}
    return HybridLoss(ctc_weight=float(mc.get("ctc_weight", 1.0)),
                      aed_weight=float(mc.get("aed_weight", 0.0)),
                      ratio_weight=float(mc.get("hnet_ratio_beta", 0.0)))


# ── assembled model ──────────────────────────────────────────────────────────
class DCASRModel(nn.Module):
    """encoder + head + hybrid loss. forward -> (total_loss, detached stats dict)."""

    def __init__(self, encoder: nn.Module, head: nn.Module, loss: HybridLoss):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.loss_fn = loss

    def forward(self, feats, feat_lens, targets, target_lens):
        enc = self.encoder(feats, feat_lens)
        ctc = self.head.loss(enc.features, enc.lengths, targets, target_lens)
        lo = self.loss_fn(ctc_loss=ctc, ratio_loss=enc.ratio_loss)   # aed wired later
        stats = {k: v.detach() for k, v in lo.items().items()}
        if enc.kept_fractions:
            stats["kept_fraction"] = enc.kept_fractions[0].detach()
        return lo.total, stats

    @torch.no_grad()
    def greedy_decode(self, feats, feat_lens) -> list[list[int]]:
        enc = self.encoder(feats, feat_lens)
        return self.head.greedy_decode(enc.features, enc.lengths)


def build_model(config, vocab_size: int) -> DCASRModel:
    """Assemble the full DC-ASR model (encoder + head + loss) from a resolved config."""
    model = DCASRModel(build_encoder(config), build_head(config, vocab_size), build_loss(config))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("build_model: encoder=%s head=%s vocab=%d params=%.1fM",
                config["encoder"], config["head"], vocab_size, n_params / 1e6)
    return model
