"""Fixed-stride pooling chunker for DC-ASR: the H2 control (learned vs fixed).

FixedPoolChunker is a drop-in replacement for DynamicChunker that compresses to the same
rate 1/N by placing boundaries on a FIXED schedule (every N frames) and mean-pooling each
window, instead of *learning* where the boundaries are. It carries no parameters and no
ratio loss — the compression rate is fixed by construction. Same chunk()/dechunk() and
ChunkOutput contract as DynamicChunker, so the encoder swaps one for the other from config
alone (plan §6.2 internal control; H2: "learned beats fixed at equal compression").

    chunk:   x [B,L,D] --masked mean over fixed windows of N--> z [B,M,D],  M = ceil(L/N)
    dechunk: z_proc [B,M,D] --broadcast each window vector back--> x̃ [B,L,D]

N=1 is an exact identity passthrough — it coincides field-for-field with DynamicChunker's
N=1 (pure Mamba), so the no-chunk control is identical under both chunkers. Fixed-stride
pooling needs an INTEGER window: Type A passes N directly (always integer); Type B passes
√N, so fixed-pool Type B is only defined at perfect-square N (a non-integer √N raises).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dcasr.logging_utils import get_logger
from dcasr.models.hnet_chunk import ChunkOutput

logger = get_logger(__name__)


class FixedPoolChunker(nn.Module):
    """Fixed-stride masked mean pooling to rate 1/N (no learned boundaries; H2 control)."""

    def __init__(self, d_model: int, N=1, ema_smoothing: bool = True):
        super().__init__()
        n = float(N)
        stride = int(round(n))
        if abs(n - stride) > 1e-6:
            raise ValueError(
                f"FixedPoolChunker needs an integer stride; got N={N!r}. Fixed-stride "
                "pooling has no fractional window — Type B fixed-pool is only defined at "
                "perfect-square N (so √N is an integer).")
        if stride < 1:
            raise ValueError(f"FixedPoolChunker stride must be >= 1, got {stride}")
        self.d_model = d_model
        self.stride = stride
        self.N = stride
        self.identity = (stride == 1)
        # accepted for interface parity with DynamicChunker; fixed pooling has no
        # confidence/probability signal, so EMA smoothing is a no-op here.
        self.ema_smoothing = ema_smoothing
        logger.debug("FixedPoolChunker(d_model=%d, stride=%d) identity=%s",
                     d_model, stride, self.identity)

    # ---- chunk (mean-pool) --------------------------------------------------
    def chunk(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> ChunkOutput:
        B, L, D = x.shape
        s = self.stride
        if self.identity:
            # stride 1: every frame is its own chunk — exact identity (== DynamicChunker N=1).
            ones = x.new_ones(B, L)
            memb = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L).clone()
            if mask is not None:
                ones = ones * mask.to(x.dtype)
            return ChunkOutput(z=x, z_mask=(mask if mask is not None
                                            else x.new_ones(B, L, dtype=torch.bool)),
                               p=ones, b=ones, membership=memb,
                               ratio_loss=x.new_zeros(()),
                               kept_fraction=x.new_ones(()))

        if mask is not None:
            lengths = mask.sum(dim=1)                       # [B] int64, exact
            m = mask.to(x.dtype)                            # [B,L]
        else:
            lengths = torch.full((B,), L, device=x.device, dtype=torch.long)
            m = x.new_ones(B, L)
        nwin = ((lengths + s - 1) // s).clamp_min(1)        # [B] windows per row
        M = int(nwin.max().item())

        pos = torch.arange(L, device=x.device)
        # window index of each frame; padded tail frames (pos beyond max valid length) can
        # exceed M-1, so clamp — they contribute 0 to the masked pool and are gathered back
        # into a padded slot in dechunk, so the clamp is harmless for real frames.
        memb = (pos // s).clamp(max=M - 1).unsqueeze(0).expand(B, L)   # [B,L]
        idxD = memb.unsqueeze(-1).expand(B, L, D)           # [B,L,D]
        # masked mean per window; accumulate in >=fp32 (bf16 sums drift at speech lengths;
        # promote_types keeps fp64 exact so autograd gradcheck passes).
        acc = torch.promote_types(x.dtype, torch.float32)
        z = x.new_zeros(B, M, D, dtype=acc).scatter_add_(
            1, idxD, (x * m.unsqueeze(-1)).to(acc))
        cnt = x.new_zeros(B, M, dtype=acc).scatter_add_(1, memb, m.to(acc))
        z = (z / cnt.clamp_min(1.0).unsqueeze(-1)).to(x.dtype)
        z_mask = cnt > 0                                    # [B,M] True where a real window
        # fixed boundaries: b_t = 1 at each window start within the valid region
        b = (pos % s == 0).to(x.dtype).unsqueeze(0).expand(B, L) * m   # [B,L]
        kept = nwin.sum().float() / lengths.sum().clamp_min(1).float()
        return ChunkOutput(z=z, z_mask=z_mask, p=b, b=b, membership=memb,
                           ratio_loss=x.new_zeros(()), kept_fraction=kept)

    # ---- dechunk (broadcast) ------------------------------------------------
    def dechunk(self, z_proc: torch.Tensor, co: ChunkOutput) -> torch.Tensor:
        """Broadcast each processed window vector back over its fine frames (identity at N=1)."""
        if self.identity:
            return z_proc
        B, L = co.membership.shape
        D = z_proc.shape[-1]
        idx = co.membership.clamp(max=z_proc.shape[1] - 1).unsqueeze(-1).expand(B, L, D)
        return torch.gather(z_proc, dim=1, index=idx)

    def forward(self, x, mask=None):
        """Convenience: returns the ChunkOutput of the pooling half only."""
        return self.chunk(x, mask)
