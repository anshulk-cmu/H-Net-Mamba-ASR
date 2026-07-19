"""DC-ASR encoder: the Mamba-H-Net sandwich (plan §4.1-4.5).

Conv-subsample ×4 (100→25 Hz) → enc Mamba stack → H-Net chunk → project to the wider
main dim → main Mamba stack (on the compressed sequence) → project back → dechunk →
residual (enc output + dechunked, the fine-detail bypass) → dec Mamba stack. Type A has
one chunk level; Type B nests two at per-block factor √N. N=1 makes every H-Net block a
100%-keep passthrough, so the encoder reduces to a pure bidirectional-Mamba stack.

Fine path runs at d_outer, the compressed main at d_main (wider where the sequence is
short, so extra capacity is cheap). Returns per-stage boundaries / chunk embeddings /
kept-fractions for the interpretability program and the summed ratio loss for training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from dcasr.logging_utils import get_logger
from dcasr.models.fixed_pool import FixedPoolChunker
from dcasr.models.hnet_chunk import DynamicChunker
from dcasr.models.mamba_block import MambaStack

logger = get_logger(__name__)


# chunker registry: config picks learned dynamic chunking or the fixed-pool H2 control.
_CHUNKERS = {"dynamic": DynamicChunker, "fixed": FixedPoolChunker}


def build_chunker(kind: str, d_model: int, N, ema_smoothing: bool = True) -> nn.Module:
    kind = str(kind).lower()
    if kind not in _CHUNKERS:
        raise ValueError(f"unknown chunker {kind!r}; choices: {sorted(_CHUNKERS)}")
    return _CHUNKERS[kind](d_model, N, ema_smoothing=ema_smoothing)


@dataclass
class EncoderOutput:
    features: torch.Tensor              # [B, L0, d_outer]  frame-rate encoder output
    lengths: torch.Tensor              # [B]               valid frames after subsampling
    ratio_loss: torch.Tensor           # scalar            Σ_stage L_ratio (0 at N=1)
    boundaries: list                   # per stage: (p [B,L], b [B,L])
    chunk_embeddings: list             # per stage: z [B, M, d]
    kept_fractions: list               # per stage: scalar realised keep-fraction


def _subsampled_length(lengths: torch.Tensor) -> torch.Tensor:
    """Valid length after two k=3,s=2 conv layers: ((L-1)//2 - 1)//2."""
    return (((lengths - 1) // 2 - 1) // 2).clamp_min(0)


class ConvSubsampling4(nn.Module):
    """×4 time downsample via two Conv2d(k=3, s=2) layers, then flatten freq → d_model."""

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2), nn.ReLU())
        f = ((n_mels - 1) // 2 - 1) // 2
        self.proj = nn.Linear(d_model * f, d_model)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        x = self.conv(feats.unsqueeze(1))                       # [B, d, T', F']
        B, C, T, F = x.shape
        x = self.proj(x.transpose(1, 2).reshape(B, T, C * F))   # [B, T', d_model]
        return x, _subsampled_length(lengths)


def _lengths_to_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


class DCASREncoder(nn.Module):
    """Type A (1-stage) or Type B (2-stage) Mamba-H-Net encoder.

    Type B design note: the plan fixes only d_outer/d_main, so the 2nd chunk level and the
    main both run at d_main (provisional d_mid = d_main); the enc/dec Mamba budget is split
    via n_mid. Revisit when Type B is trained.
    """

    def __init__(self, n_mels: int = 80, d_outer: int = 384, d_main: int = 512,
                 n_enc: int = 4, n_main: int = 12, n_dec: int = 4, n_mid: int = 4,
                 arch_type: str = "A", N: int = 1, bidirectional: bool = True,
                 hnet_ema: bool = True, chunker: str = "dynamic"):
        super().__init__()
        if arch_type not in ("A", "B"):
            raise ValueError(f"arch_type must be 'A' or 'B', got {arch_type!r}")
        self.arch_type = arch_type
        self.N = N
        self.chunker = chunker
        self.subsample = ConvSubsampling4(n_mels, d_outer)
        self.enc = MambaStack(n_enc, d_outer, bidirectional)
        self.dec = MambaStack(n_dec, d_outer, bidirectional)

        if arch_type == "A":
            self.chunk = build_chunker(chunker, d_outer, N, hnet_ema)
            self.proj_in = nn.Linear(d_outer, d_main)
            self.main = MambaStack(n_main, d_main, bidirectional)
            self.proj_out = nn.Linear(d_main, d_outer)
        else:                                                   # Type B: two √N stages
            nb = math.sqrt(N)
            self.chunk1 = build_chunker(chunker, d_outer, nb, hnet_ema)
            self.proj1_in = nn.Linear(d_outer, d_main)
            self.mid = MambaStack(n_mid, d_main, bidirectional)
            self.chunk2 = build_chunker(chunker, d_main, nb, hnet_ema)
            self.main = MambaStack(n_main, d_main, bidirectional)
            self.mid_dec = MambaStack(n_mid, d_main, bidirectional)
            self.proj1_out = nn.Linear(d_main, d_outer)
        logger.debug("DCASREncoder(type=%s, N=%s, d_outer=%d, d_main=%d, chunker=%s)",
                     arch_type, N, d_outer, d_main, chunker)

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor) -> EncoderOutput:
        x, lengths = self.subsample(feats, feat_lengths)        # [B, L0, d_outer]
        mask = _lengths_to_mask(lengths, x.shape[1])
        x_enc = self.enc(x, lengths)
        if self.arch_type == "A":
            return self._forward_A(x_enc, mask, lengths)
        return self._forward_B(x_enc, mask, lengths)

    def _forward_A(self, x_enc, mask, lengths) -> EncoderOutput:
        co = self.chunk.chunk(x_enc, mask)
        z = self.proj_in(co.z)
        z = self.main(z, co.z_mask.sum(1))
        z = self.proj_out(z)
        x_dech = self.chunk.dechunk(z, co)
        x_out = self.dec(x_enc + x_dech, lengths)               # H-Net residual: fine bypass
        return EncoderOutput(x_out, lengths, co.ratio_loss,
                             [(co.p, co.b)], [co.z], [co.kept_fraction])

    def _forward_B(self, x_enc, mask, lengths) -> EncoderOutput:
        co1 = self.chunk1.chunk(x_enc, mask)                    # stage 1: frames → units
        z1 = self.mid(self.proj1_in(co1.z), co1.z_mask.sum(1))
        co2 = self.chunk2.chunk(z1, co1.z_mask)                 # stage 2: units → words
        z2 = self.main(co2.z, co2.z_mask.sum(1))
        z1_dec = self.mid_dec(z1 + self.chunk2.dechunk(z2, co2), co1.z_mask.sum(1))
        x_dech = self.chunk1.dechunk(self.proj1_out(z1_dec), co1)
        x_out = self.dec(x_enc + x_dech, lengths)
        return EncoderOutput(x_out, lengths, co1.ratio_loss + co2.ratio_loss,
                             [(co1.p, co1.b), (co2.p, co2.b)], [co1.z, co2.z],
                             [co1.kept_fraction, co2.kept_fraction])
