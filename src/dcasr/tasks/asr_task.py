"""ASR task builders: turn a resolved config into wired model objects.

The single seam between YAML and Python. `build_*` functions select a class by string
name via in-repo registries (ESPnet name+_conf style) and adapt the config block to the
constructor, so experiments swap encoder/head/loss/optimizer/scheduler from config alone.
`DCASRModel(encoder, ctc_head, aed_head, loss)` assembles them via constructor injection —
one hybrid model carries BOTH a CTC and an AED head (which are built iff their weight > 0);
its `forward(feats, feat_lens, targets, target_lens) -> (loss, stats)` is the model-agnostic
contract the Trainer depends on (it never imports a concrete encoder/head).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dcasr.decoders.aed import AEDHead
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
        hnet_ema=bool(h.get("ema_smoothing", True)),
        chunker=str(h.get("chunker", "dynamic")))


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


def build_aed_head(config, vocab_size: int) -> AEDHead:
    """AED attention head over the encoder's fine-rate output (aed_conf; lsm from model_conf)."""
    ec = config["encoder_conf"]
    ac = config.get("aed_conf", {}) or {}
    mc = config.get("model_conf", {}) or {}
    return AEDHead(int(vocab_size), int(ec["d_outer"]),
                   n_layers=int(ac.get("n_layers", 6)), n_heads=int(ac.get("n_heads", 4)),
                   d_ff=int(ac.get("d_ff", 2048)), dropout=float(ac.get("dropout", 0.1)),
                   lsm_weight=float(mc.get("lsm_weight", 0.1)),
                   max_decode_len=int(ac.get("max_decode_len", 512)))


def build_loss(config) -> HybridLoss:
    mc = config.get("model_conf", {}) or {}
    return HybridLoss(ctc_weight=float(mc.get("ctc_weight", 1.0)),
                      aed_weight=float(mc.get("aed_weight", 0.0)),
                      ratio_weight=float(mc.get("hnet_ratio_beta", 0.0)))


# ── assembled model ──────────────────────────────────────────────────────────
class DCASRModel(nn.Module):
    """encoder + CTC and/or AED heads + hybrid loss. forward -> (total_loss, detached stats).

    A head is present iff its weight > 0 (build_model decides). Validation greedy_decode
    reads out CTC (fast, non-autoregressive) when a CTC head exists, else AED — the full
    read-out matrix (AED-beam, joint) lives in the decode stage.
    """

    def __init__(self, encoder: nn.Module, ctc_head: nn.Module | None = None,
                 aed_head: nn.Module | None = None, loss: HybridLoss | None = None):
        super().__init__()
        if ctc_head is None and aed_head is None:
            raise ValueError("DCASRModel needs at least one of ctc_head / aed_head")
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.aed_head = aed_head
        self.loss_fn = loss

    def forward(self, feats, feat_lens, targets, target_lens):
        enc = self.encoder(feats, feat_lens)
        ctc = (self.ctc_head.loss(enc.features, enc.lengths, targets, target_lens)
               if self.ctc_head is not None else None)
        aed = (self.aed_head.loss(enc.features, enc.lengths, targets, target_lens)
               if self.aed_head is not None else None)
        lo = self.loss_fn(ctc_loss=ctc, aed_loss=aed, ratio_loss=enc.ratio_loss)
        stats = {k: v.detach() for k, v in lo.items().items()}
        for i, kf in enumerate(enc.kept_fractions):    # per stage: Type B's stage-2 must be visible
            stats["kept_fraction" if i == 0 else f"kept_fraction_{i}"] = kf.detach()
        if self.ctc_head is not None:
            # zero_infinity silently zeroes utts with enc_len < token_len + #adjacent-repeats
            # (speed-perturb 1.1x can create them) — count so training health is observable
            U = targets.shape[1]
            reps = targets.new_zeros(targets.shape[0])
            if U > 1:
                pair_ok = (torch.arange(U - 1, device=targets.device)[None, :]
                           < (target_lens - 1)[:, None])
                reps = ((targets[:, 1:] == targets[:, :-1]) & pair_ok).sum(1)
            stats["ctc_infeasible"] = (enc.lengths < target_lens + reps).sum()
        return lo.total, stats

    @torch.no_grad()
    def greedy_decode(self, feats, feat_lens) -> list[list[int]]:
        enc = self.encoder(feats, feat_lens)
        head = self.ctc_head if self.ctc_head is not None else self.aed_head
        return head.greedy_decode(enc.features, enc.lengths)


def build_model(config, vocab_size: int) -> DCASRModel:
    """Assemble the full DC-ASR model (encoder + CTC/AED heads + hybrid loss) from config.

    CTC head built iff ctc_weight > 0; AED head iff aed_weight > 0 (so one YAML switches a
    run from CTC-only to hybrid CTC/AED by setting the weights).
    """
    mc = config.get("model_conf", {}) or {}
    ctc_w, aed_w = float(mc.get("ctc_weight", 1.0)), float(mc.get("aed_weight", 0.0))
    if ctc_w <= 0.0 and aed_w <= 0.0:
        raise ValueError("model_conf needs ctc_weight > 0 or aed_weight > 0")
    ctc_head = build_head(config, vocab_size) if ctc_w > 0.0 else None
    aed_head = build_aed_head(config, vocab_size) if aed_w > 0.0 else None
    model = DCASRModel(build_encoder(config), ctc_head, aed_head, build_loss(config))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("build_model: encoder=%s ctc=%s aed=%s vocab=%d params=%.1fM",
                config["encoder"], ctc_head is not None, aed_head is not None,
                vocab_size, n_params / 1e6)
    return model
