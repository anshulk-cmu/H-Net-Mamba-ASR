"""Mamba-2 block wrapper (selective SSM backbone).

Plan refs: §2.1, Mamba [5] / Mamba-2 SSD [6]. Backs both the encoder stacks and the
main (coarsest-rate) network. Prefer the official `mamba_ssm` CUDA kernels.

TODO: MambaBlock(d_model, ...) -> residual Mamba-2 layer; MambaStack(n_layers, ...).
"""
