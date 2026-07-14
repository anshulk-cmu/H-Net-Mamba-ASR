"""Mamba-2 backbone blocks for DC-ASR: pre-norm residual Mamba-2 layer + a stack.

Uses the official mamba_ssm CUDA kernels (GPU-only). Bidirectional mode runs a forward
and a length-aware reversed Mamba-2 and sums them, giving each frame past+future context
for the offline ASR encoder. Shape-preserving [B, T, d_model] -> [B, T, d_model];
device/rank-agnostic. Requires expand*d_model divisible by headdim.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba2

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def reverse_sequences(x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Reverse each sequence along time. With lengths, only the valid span is reversed
    (padding stays in place), so a double-reverse is the identity on real frames."""
    if lengths is None:
        return torch.flip(x, dims=[1])
    B, T, _ = x.shape
    pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    L = lengths.to(x.device).view(B, 1)
    idx = torch.where(pos < L, L - 1 - pos, pos).clamp_(0, T - 1)      # reversed within [0,L)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand_as(x))


class MambaBlock(nn.Module):
    """Pre-norm residual Mamba-2 layer. bidirectional=True adds a length-aware reversed pass.

    y = x + Mamba2_fwd(norm(x)) [+ reverse(Mamba2_bwd(reverse(norm(x))))]. Causal by
    construction, so trailing padding never leaks into valid-frame outputs.
    """

    def __init__(self, d_model: int, bidirectional: bool = True, d_state: int = 128,
                 d_conv: int = 4, expand: int = 2, headdim: int = 64):
        super().__init__()
        assert (expand * d_model) % headdim == 0, \
            f"expand*d_model ({expand * d_model}) must be divisible by headdim ({headdim})"
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)
        kw = dict(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.fwd = Mamba2(**kw)
        self.bwd = Mamba2(**kw) if bidirectional else None
        logger.debug("MambaBlock(d_model=%d, bidir=%s, d_state=%d)", d_model, bidirectional, d_state)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            hr = reverse_sequences(h, lengths).contiguous()
            y = y + reverse_sequences(self.bwd(hr), lengths)
        return x + y


class MambaStack(nn.Module):
    """n_layers MambaBlocks + a final LayerNorm. Shape-preserving [B, T, d_model]."""

    def __init__(self, n_layers: int, d_model: int, bidirectional: bool = True, **block_kw):
        super().__init__()
        self.layers = nn.ModuleList(
            MambaBlock(d_model, bidirectional=bidirectional, **block_kw) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        logger.debug("MambaStack(n_layers=%d, d_model=%d, bidir=%s)",
                     n_layers, d_model, bidirectional)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, lengths)
        return self.norm(x)
