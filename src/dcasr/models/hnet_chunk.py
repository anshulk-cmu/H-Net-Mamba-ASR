"""H-Net dynamic chunking: router, downsampler, EMA smoothing, STE, ratio loss, dechunk.

Plan refs: §2.2. Boundary prob p_t = ½(1 − cos(q_t, k_{t−1})); keep p_t>0.5; EMA smoothing
z̄_t = P_t ẑ̂_t + (1−P_t) z̄_{t−1}; straight-through estimator [18]; ratio loss to target 1/N.

KEY KNOB: compression level N (overall downsampling). Per-block factor = N (Type A) or
√N (Type B, so two blocks multiply to 1/N — iso-compression, plan §4.3).
N=1 => no-op (keeps 100%, pure-Mamba passthrough).

TODO:
  - RoutingModule: cosine-dissimilarity boundary prob
  - Downsampler / Upsampler(dechunk)
  - ratio_loss(p, target_ratio)
  - HNetChunk(...) forward returning (compressed, boundaries, ratio_loss)
"""
