"""H-Net dynamic chunking for DC-ASR: router, downsampler, dechunk, ratio loss.

Faithful to Hwang, Wang & Gu, "Dynamic Chunking for End-to-End Hierarchical
Sequence Modeling" (arXiv:2507.07955). This module is the *scientific core* of
DC-ASR and is 100% pure PyTorch — developed and unit-tested on CUDA GPUs on
Babel (the Mamba backbone additionally needs the mamba-ssm CUDA kernels).

Mechanism (plan Sec. 2.2, 4.3)
------------------------------
A stack of encoder frames  x̂ ∈ R[B,L,D]  is compressed to a shorter sequence by
*learning where the boundaries are*, then expanded back for the decoder:

    chunk:   x̂ ──router──> p,b ──downsample──> z   (B, M<=L, D)   [M = #boundaries]
    (main network / Mamba runs on z)
    dechunk: ẑ ──EMA smooth (chunk rate, P=downsampled p)──> upsample(gather)
             ──confidence-STE──> x̃ (B, L, D)          [paper Eq. 5 → 8 → 9]

Router (cosine-dissimilarity boundary predictor), per position t:
    q_t = W_q x̂_t,   k_t = W_k x̂_t
    p_t = ½ (1 − cos(q_t, k_{t-1}))  ∈ [0,1]          # low similarity ⇒ boundary
    b_t = 1[p_t ≥ 0.5]                                # hard boundary indicator
    p_1 ≡ 1.0  (the first frame is always a boundary / always kept)
Causal: p_t depends only on t and t-1 (no future leak) — offline ASR is fine with
this, and it keeps the door open for a streaming variant (out of current scope).

Ratio loss (targets an average keep-fraction of 1/N; MoE-load-balancing style):
    L_ratio = N/(N-1) · [ (N-1)·F·G + (1-F)(1-G) ],
    F = mean_t b_t  (hard, non-diff),   G = mean_t p_t  (soft, the training signal)
Undefined at N=1 → N=1 is a special-cased **identity passthrough** (no compression,
no ratio loss): this is DC-ASR's no-chunk control, so the encoder reduces to pure
Mamba. See configs/typeA_small_N1_ctc.yaml (the first go/no-go experiment).

Confidence-weighted straight-through estimator (dechunk):
    c_t = p_t^{b_t} (1-p_t)^{1-b_t}   =  p_t if kept else (1-p_t)
Forward pass is exact (full-magnitude expanded vector); gradient flows as if the
vector were scaled by c_t, so confident/accurate boundaries are rewarded.

DC-ASR compression levels (plan Sec. 4.3)
-----------------------------------------
N is the overall downsampling factor; keep-fraction = 1/N (N=1→100%, 2→50%,
3→33%, 4→25%). Type A (1-stage) uses one chunker at factor N. Type B (2-stage)
uses two chunkers each at per-block factor sqrt(N), so the two blocks multiply to
1/N — matched overall compression ("iso-compression") with Type A.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Aux container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChunkOutput:
    """Everything the dechunk step and the losses/interpretability need."""
    z: torch.Tensor            # [B, M, D]  compressed (boundary) vectors
    z_mask: torch.Tensor       # [B, M]     True where z is a real (non-pad) chunk
    p: torch.Tensor            # [B, L]     soft boundary probabilities
    b: torch.Tensor            # [B, L]     hard boundary indicators {0,1}
    membership: torch.Tensor   # [B, L]     chunk index each fine frame belongs to
    ratio_loss: torch.Tensor   # scalar     0.0 when N==1
    kept_fraction: torch.Tensor  # scalar    realised mean keep-fraction (for logging)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
class RoutingModule(nn.Module):
    """Cosine-dissimilarity boundary router (H-Net Eq. routing).

    p_t = ½(1 − cos(W_q x̂_t, W_k x̂_{t-1})),  b_t = 1[p_t ≥ 0.5],  p_1 = 1.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        # init near-identity so that at start q≈k≈x̂ and cos is meaningful
        nn.init.eye_(self.W_q.weight)
        nn.init.eye_(self.W_k.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: [B, L, D]; mask: [B, L] True=valid. Returns (p [B,L], b [B,L])."""
        B, L, D = x.shape
        q = self.W_q(x)                              # [B,L,D]
        k = self.W_k(x)                              # [B,L,D]
        # cos(q_t, k_{t-1}): shift k by one position (k_{t-1})
        k_prev = torch.roll(k, shifts=1, dims=1)     # k_prev[:,0] is wrapped (unused)
        cos = F.cosine_similarity(q, k_prev, dim=-1, eps=self.eps)  # [B,L]
        p = 0.5 * (1.0 - cos)                         # [B,L] ∈ [0,1]
        # first valid frame is always a boundary: p_1 = 1
        p = p.clone()
        p[:, 0] = 1.0
        p = p.clamp(0.0, 1.0)
        b = (p >= 0.5).to(p.dtype)                    # hard, non-differentiable
        if mask is not None:
            p = p * mask.to(p.dtype)
            b = b * mask.to(p.dtype)
            # ensure the first *valid* position per row is a boundary even if padded-left
            # (our data is right-padded, so index 0 is always valid; kept simple.)
        return p, b


# ─────────────────────────────────────────────────────────────────────────────
# Ratio loss
# ─────────────────────────────────────────────────────────────────────────────
def ratio_loss(p: torch.Tensor, b: torch.Tensor, N: int,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    """H-Net ratio loss steering the mean keep-fraction toward 1/N.

    L = N/(N-1) [ (N-1) F G + (1-F)(1-G) ],  F=mean b (hard), G=mean p (soft).
    Returns a scalar 0-dim tensor. N==1 ⇒ 0 (identity mode, no regulariser).
    """
    if N == 1:
        return p.new_zeros(())
    p, b = p.float(), b.float()      # bf16 sums drift at speech lengths
    if mask is None:
        F_ = b.mean()
        G_ = p.mean()
    else:
        m = mask.to(p.dtype)
        denom = m.sum().clamp_min(1.0)
        F_ = (b * m).sum() / denom
        G_ = (p * m).sum() / denom
    coef = N / (N - 1.0)
    return coef * ((N - 1.0) * F_ * G_ + (1.0 - F_) * (1.0 - G_))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic chunker (chunk + dechunk)
# ─────────────────────────────────────────────────────────────────────────────
class DynamicChunker(nn.Module):
    """One H-Net dynamic-chunking block: chunk() downsamples, dechunk() restores.

    N=1 is an exact identity passthrough (no-chunk control). For N>=2 the router
    predicts boundaries, downsample() gathers boundary frames into a shorter
    (padded) sequence, and dechunk() expands the main-network output back to the
    fine resolution with a confidence-weighted STE and EMA smoothing.
    """

    def __init__(self, d_model: int, N: int = 1, ema_smoothing: bool = True):
        super().__init__()
        assert N >= 1
        self.d_model = d_model
        self.N = N
        self.ema_smoothing = ema_smoothing
        self.identity = (N == 1)
        # router only needed when actually chunking
        self.router = None if self.identity else RoutingModule(d_model)
        logger.debug("DynamicChunker(d_model=%d, N=%d, ema_smoothing=%s) identity=%s",
                     d_model, N, ema_smoothing, self.identity)

    # ---- chunk (downsample) -------------------------------------------------
    def chunk(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> ChunkOutput:
        B, L, D = x.shape
        if self.identity:
            # N=1: keep 100% of frames, every frame is a boundary, no ratio loss.
            ones = x.new_ones(B, L)
            memb = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L).clone()
            if mask is not None:
                ones = ones * mask.to(x.dtype)
            return ChunkOutput(z=x, z_mask=(mask if mask is not None
                                            else x.new_ones(B, L, dtype=torch.bool)),
                               p=ones, b=ones, membership=memb,
                               ratio_loss=x.new_zeros(()),
                               kept_fraction=x.new_ones(()))

        p, b = self.router(x, mask)                    # [B,L], [B,L]
        rl = ratio_loss(p, b, self.N, mask)
        # membership: chunk index of each fine frame = cumsum(b)-1  (b_1=1 ⇒ starts at 0);
        # integer cumsum — float cumsum corrupts ranks past 256 kept frames in bf16
        keep = b > 0.5
        memb = (keep.long().cumsum(dim=1) - 1).clamp_min(0)       # [B,L]
        counts = keep.sum(dim=1)                       # [B] number of chunks per row
        M = int(counts.max().item()) if counts.numel() else 0
        M = max(M, 1)
        # memb[i,t] is each kept frame's destination slot — one collision-free scatter
        bi, ti = keep.nonzero(as_tuple=True)           # [K] batch/time idx of kept frames
        z = x.new_zeros(B, M, D)
        z_mask = torch.zeros(B, M, dtype=torch.bool, device=x.device)
        z[bi, memb[bi, ti]] = x[bi, ti]
        z_mask[bi, memb[bi, ti]] = True
        valid = mask.sum() if mask is not None else torch.tensor(B * L, device=x.device)
        kept = keep.sum().float() / valid.float().clamp_min(1.0)
        return ChunkOutput(z=z, z_mask=z_mask, p=p, b=b, membership=memb,
                           ratio_loss=rl, kept_fraction=kept)

    # ---- dechunk (upsample) -------------------------------------------------
    def dechunk(self, z_proc: torch.Tensor, co: ChunkOutput) -> torch.Tensor:
        """Expand processed coarse vectors z_proc [B,M,D] back to fine [B,L,D].

        Paper order (Eq. 5 → 8 → 9): EMA-smooth over the COMPRESSED sequence
        using the downsampled boundary probabilities P (p at kept frames), THEN
        upsample via the membership gather, THEN scale by the confidence-STE
        weight (exact in forward, grad ∝ c_t). Identity (N=1) is a passthrough.
        """
        if self.identity:
            return z_proc
        B, L = co.membership.shape
        M, D = z_proc.shape[1], z_proc.shape[-1]
        if self.ema_smoothing:
            # P [B,M]: downsample p exactly like z — p at each kept frame,
            # scattered to its chunk slot (pad slots 0; causal EMA never reads
            # them into real slots since padding is on the right)
            keep = co.b > 0.5
            bi, ti = keep.nonzero(as_tuple=True)
            P = co.p.new_zeros(B, M)
            P[bi, co.membership[bi, ti]] = co.p[bi, ti]
            z_proc = self._ema(z_proc, P)                          # Eq. 5, chunk rate
        idx = co.membership.unsqueeze(-1).expand(B, L, D)
        x_up = torch.gather(z_proc, dim=1, index=idx)              # Eq. 8: upsample
        c = torch.where(co.b > 0.5, co.p, 1.0 - co.p)              # [B,L]
        ste = (c + (1.0 - c).detach()).unsqueeze(-1)               # [B,L,1] ==1.0 fwd
        return x_up * ste                                          # Eq. 9: STE last

    @staticmethod
    def _ema(x: torch.Tensor, p: torch.Tensor, p_clamp: float = 1e-4) -> torch.Tensor:
        """EMA smoother  z̄_t = P_t x_t + (1-P_t) z̄_{t-1}, vectorised as one causal
        matmul:  z̄_t = Σ_{j≤t} exp(S_t−S_j)·s_j,  S = cumsum(log(1-P)), s_0 = x_0,
        s_j = P_j x_j. O(L²) — cheap at 25 Hz lengths.

        p is HARD-clamped to [p_clamp, 1-p_clamp] with ZERO gradient at
        saturation (official H-Net DeChunkLayer parity). A backward-identity
        clamp here passes 1/(1-p) gradients up to 1e6 per saturated boundary —
        the gradient amplifier in the N=2 divergence (runlog 2026-07-20).
        """
        B, L, D = x.shape
        if L == 1:
            return x.clone()
        pc = p.clamp(p_clamp, 1.0 - p_clamp)          # grads die outside the band
        src = torch.cat([x[:, :1], pc[:, 1:, None] * x[:, 1:]], dim=1)  # [B,L,D]
        wdtype = torch.promote_types(p.dtype, torch.float32)
        a = 1.0 - pc[:, 1:].to(wdtype)
        S = F.pad(torch.log(a).cumsum(dim=1), (1, 0))                   # [B,L], S_0 = 0
        logw = S.unsqueeze(-1) - S.unsqueeze(1)                         # [B,L,L]: S_t - S_j
        future = torch.ones(L, L, dtype=torch.bool, device=x.device).triu(1)
        W = torch.exp(logw.masked_fill(future, float("-inf"))).to(x.dtype)
        return torch.bmm(W, src)

    def forward(self, x, mask=None):
        """Convenience: returns the ChunkOutput of the downsampling half only."""
        return self.chunk(x, mask)
