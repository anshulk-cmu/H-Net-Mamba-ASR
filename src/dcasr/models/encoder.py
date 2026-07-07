"""DC-ASR encoder assembly: the Mamba–H-Net sandwich.

Plan refs: §4.1–4.2. Type A (1-stage) = Mamba→H-Net→Mamba; Type B (2-stage) =
Mamba→H-Net→Mamba→H-Net→Mamba. Conv subsampling ×4 frontend -> 25 Hz -> stages.
Sizes (§4.5): Small ~25–30M, Large ~90–120M (params-matched to Zipformer-S/M/L).

TODO: DCASREncoder(type={A,B}, size={small,large}, N=1) with returns for the
interpretability hooks (per-stage boundaries + chunk embeddings).
"""
