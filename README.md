# H-Mamba ASR — Can Speech Recognition Learn to Skip?

**Anshul Kumar, Carnegie Mellon University**
**Advisor: Prof. Shinji Watanabe**

This project applies H-Net Dynamic Chunking to ConMamba ASR encoders, enabling
learned compression of speech features at acoustic boundaries. The system learns
where to place chunk boundaries mid-network, compressing the sequence before the
second half of the encoder processes it, then reconstructing full resolution for
decoding.

Research question: *Can we reduce computation in Mamba-based ASR by learning to
skip redundant frames, without sacrificing recognition accuracy?*

---

## 1. Project Overview

### Phase 1: Baseline Reproduction (Complete)
Reproduce all 8 experiments from the ConMamba paper
([arXiv:2407.09732](https://arxiv.org/abs/2407.09732)) on LibriSpeech 960h.
This establishes the baselines against which H-Mamba is compared.

### Phase 2: H-Mamba Training (In Progress)
Train H-Mamba models (ConMamba + Dynamic Chunking) at 8 configurations:
- Small model (14.1M params): N=1 (control), N=2, N=3, N=4
- Large model (115.2M params): N=1 (control), N=2, N=3, N=4

Where N is the target compression factor (N=2 keeps 50% of frames, N=4 keeps 25%).

---

## 2. Prerequisites

- Linux with SLURM job scheduler
- CUDA 11.8 available at `/usr/local/cuda-11.8`
- Miniconda / Anaconda
- 2x NVIDIA A6000 GPUs (48 GB each)
- ~120 GB free disk space (LibriSpeech extracted + model checkpoints)

---

## 3. Data

### LibriSpeech 960h Location
```
/data/user_data/anshulk/hnet_asr/LibriSpeech
```

Contains all 7 splits:
```
LibriSpeech/
  train-clean-100/   (~12 GB)
  train-clean-360/   (~44 GB)
  train-other-500/   (~57 GB)
  dev-clean/
  dev-other/
  test-clean/
  test-other/
```

---

## 4. Environment Setup

### 4.1 Create Conda Environment
```bash
conda create -n hnetasr python=3.9 -y
conda activate hnetasr
```

### 4.2 Install PyTorch (CUDA 11.8)
```bash
pip install torch==2.1.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 4.3 Install CUDA Extension Packages

These require `--no-build-isolation` and explicit `CUDA_HOME`:

```bash
CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH \
    pip install causal-conv1d==1.4.0 mamba-ssm==2.0.3 \
    --no-build-isolation
```

> **causal-conv1d 1.4.0** changed the CUDA kernel API (added `initial_states`,
> `dfinal_states` args). Our `selective_scan_interface.py` is patched for this.
> Do NOT use causal-conv1d < 1.4.0 with this codebase.

### 4.4 Install Remaining Packages
```bash
pip install -r requirements.txt
```

> **Pinned versions (do not upgrade):**
> - `numpy==1.26.4` — compiled modules break on NumPy 2.x
> - `transformers==4.40.0` — mamba-ssm uses API removed in 4.42+

### 4.5 Verify Installation
```bash
python -c "
import torch; print(f'torch: {torch.__version__}')
import torchaudio; print(f'torchaudio: {torchaudio.__version__}')
import speechbrain as sb; print(f'speechbrain: {sb.__version__}')
from mamba_ssm import Mamba; print('mamba_ssm: OK')
import causal_conv1d; print(f'causal_conv1d: {causal_conv1d.__version__}')
import tensorboard; print('tensorboard: OK')
print('All OK!')
"
```

### 4.6 Known Issues

| Issue | Affected Component | Workaround | Fix |
|-------|-------------------|------------|-----|
| `mamba_chunk_scan_combined` Triton JIT assertion on Ampere GPUs (A6000 sm_86, L40S sm_89) | DeChunk EMA expansion | PyTorch fallback in `HMambaEncoder.py` (`MAMBA_KERNEL_AVAILABLE = False`) | Requires triton >= 2.2.0 (needs PyTorch >= 2.2) |
| `causal_conv1d_fwd` API changed in v1.4.0 | BiMamba encoder layers (forward) | `selective_scan_interface.py` patched: 5-arg → 7-arg (added `initial_states`, `final_states_out`) | Already applied |
| `causal_conv1d_bwd` API changed in v1.4.0 | BiMamba encoder layers (backward) | `selective_scan_interface.py` patched: 7-arg → 10-arg, returns 4 values instead of 3 | Already applied |
| NumPy 2.x breaks DDP | `torch.distributed.broadcast_object_list` → `tensor.numpy()` fails | Pin `numpy==1.26.4` (< 2.0) | Already applied |

The Triton issue only affects the DeChunk expansion step (Mamba-2 SSD kernel).
The main encoder Mamba-1 layers use optimized CUDA kernels and are unaffected.

### 4.7 Verified Package Stack

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.9 | |
| torch | 2.1.1+cu118 | Bundles triton 2.1.0 |
| torchaudio | 2.1.1+cu118 | |
| mamba-ssm | 2.0.3 | Mamba-1 + Mamba-2 ops |
| causal-conv1d | 1.4.0 | New fwd/bwd API |
| speechbrain | 1.0.0 | |
| numpy | 1.26.4 | Must be < 2.0 |
| transformers | 4.40.0 | Must be <= 4.41 |
| CUDA | 11.8 | `/usr/local/cuda-11.8` |
| GPU | 2x A6000 (48GB) | Ampere, sm_86 |

---

## 5. Phase 1: Baseline Results (960h)

LM decoding: beam=66, CTC weight=0.40, LM weight=0.60 with
`speechbrain/asr-transformer-transformerlm-librispeech`.

### 5.1 Results (WER%)

| Model | Encoder | Decoder | #Params | With LM (clean / other) | Without LM (clean / other) | Status |
|-------|---------|---------|---------|------------------------|---------------------------|--------|
| conformer_small | Conformer | Transformer | 13.3M | 2.52 / 5.97 | 4.13 / 10.13 | Done |
| conmamba_small | ConMamba | Transformer | 14.1M | **2.22** / **5.56** | **3.34** / **8.47** | Done |
| conmambamamba_small | ConMamba | Mamba | 14.6M | 2.52 / 5.98 | 3.64 / 8.70 | Done |
| conformer_large | Conformer | Transformer | 109.1M | **2.03** / **4.70** | **2.57** / **5.94** | Done |
| conmamba_large | ConMamba | Transformer | 115.2M | 2.27 / 5.12 | 2.82 / 6.60 | Done |
| conmambamamba_large | ConMamba | Mamba | 122.9M | 2.41 / 5.72 | 2.93 / 6.99 | Done |
| conformer_large (CTC) | Conformer | CTC only | 28.8M | — | In progress | fp32, running (job 6907548) |
| conmamba_large (CTC) | ConMamba | CTC only | 31.6M | — | 3.93 / 10.40 | Done |

### 5.2 Paper Reference Results (WER%)

| Model | #Params | With LM (clean / other) | Without LM (clean / other) |
|-------|---------|------------------------|---------------------------|
| Conformer (S) | 13.3M | 2.5 / 6.1 | 4.1 / 10.0 |
| ConMamba (S) | 14.1M | 2.4 / 5.8 | 4.0 / 9.5 |
| ConMambaMamba (S) | 15.0M | 2.5 / 6.5 | 4.0 / 9.7 |
| Conformer (L) | 109.1M | 2.0 / 4.5 | 2.6 / 6.7 |
| ConMamba (L) | 115.2M | 2.1 / 4.9 | 2.8 / 6.7 |
| ConMambaMamba (L) | 122.9M | 2.4 / 5.7 | 3.0 / 7.0 |
| Conformer (CTC) | 28.8M | — | 4.3 / 11.3 |
| ConMamba (CTC) | 31.6M | — | 3.9 / 10.3 |

---

## 6. Phase 2: H-Mamba Experiments

### 6.1 Architecture

```
Input (B, L, D)
    |
Stage 0: ConMamba layers 0-5 (frame-level)
    |
DC: RoutingModule -> ChunkLayer (compress L -> M)
    |
Stage 1: ConMamba layers 6-11 (chunk-level, on M frames)
    |
DeChunk: EMA-based expansion (M -> L)
    |
Residual + LayerNorm
    |
Output (B, L, D)
```

### 6.2 Configurations

| Config | d_model | target_N | DC Loss Weight | Warmup Epochs | Gumbel End | Epochs |
|--------|---------|----------|---------------|---------------|------------|--------|
| hmamba_small_N1 | 144 | 1.0 (control) | 0.0 | 0 | 1.0 | 300 |
| hmamba_small_N2 | 144 | 2.0 | 5.0 | 15 | 0.3 | 300 |
| hmamba_small_N3 | 144 | 3.0 | 6.5 | 20 | 0.3 | 300 |
| hmamba_small_N4 | 144 | 4.0 | 7.5 | 20 | 0.3 | 300 |
| hmamba_large_N1 | 512 | 1.0 (control) | 0.0 | 0 | 1.0 | 300 |
| hmamba_large_N2 | 512 | 2.0 | 5.0 | 15 | 0.3 | 300 |
| hmamba_large_N3 | 512 | 3.0 | 6.5 | 20 | 0.3 | 300 |
| hmamba_large_N4 | 512 | 4.0 | 7.5 | 20 | 0.3 | 300 |

All use early stopping (patience=30, warmup=50 epochs) and offline WandB logging.

### 6.3 Running H-Mamba Training

```bash
cd /home/anshulk/h-mamba_asr/Mamba-ASR

# Example: Small N=2
torchrun --nproc_per_node=2 train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --precision bf16

# Via SLURM:
sbatch ../slurm/hmamba_small_N2.sh
```

### 6.4 Smoke Test Status

All smoke tests passed. Ready for full 960h training.

| Test | Job | Result |
|------|-----|--------|
| Single GPU, 3 epochs + eval | v2 (6921845) | Passed. Loss 597→395, compression converges. |
| DDP (2 GPUs), 1 epoch | v3 (6928765) | Passed. NumPy 1.26.4 fix confirmed, no DDP crash. |
| Checkpoint resume (DDP) | 6928816 | Passed. Epoch 2 resumed correctly — optimizer, scheduler, DC state restored. |
| Final validation (DDP + grad fix) | 6928979 | Passed. Grad norm=1195 (was 0), bias_grad=4.9→13.5, VRAM 55%/GPU. |

Previous issues resolved:
- NumPy 2.0.2 broke DDP (`broadcast_object_list` → `tensor.numpy()`). Fixed: numpy 1.26.4.
- Grad norm logging always showed 0.0 (computed after `zero_grad`). Fixed: captured pre-step.
- Smoke test v1 (6912668) timed out during beam search eval (2h limit too short).

### 6.5 H-Mamba Results

| Model | target_N | Compression | With LM (clean / other) | Without LM (clean / other) | Status |
|-------|----------|-------------|------------------------|---------------------------|--------|
| hmamba_small_N1 | 1.0 | None | — | — | Ready to submit |
| hmamba_small_N2 | 2.0 | 50% | — | — | Ready to submit |
| hmamba_small_N3 | 3.0 | 33% | — | — | Ready to submit |
| hmamba_small_N4 | 4.0 | 25% | — | — | Ready to submit |
| hmamba_large_N1 | 1.0 | None | — | — | Ready to submit |
| hmamba_large_N2 | 2.0 | 50% | — | — | Ready to submit |
| hmamba_large_N3 | 3.0 | 33% | — | — | Ready to submit |
| hmamba_large_N4 | 4.0 | 25% | — | — | Ready to submit |

### 6.6 100-Hour Pilot Results (Small model, pre-bug-fix)

| Config | Compression | test-clean WER | test-other WER |
|--------|-------------|---------------|---------------|
| H-Mamba Small N=2 | 50% | 5.96 | 16.35 |
| H-Mamba Small N=3 | 67% | 7.80 | 21.36 |
| H-Mamba Small N=4 | 75% | 7.35 | 19.71 |

---

## 7. Project Structure

```
h-mamba_asr/
  README.md                       # This file
  plan.md                         # 8-week EMNLP 2026 submission plan
  requirements.txt                # Pinned dependencies
  docs/                           # Detailed documentation
    baseline_reproduction.md      # Phase 1: complete analysis of all baseline runs
    hmamba_dynamic_chunking.md    # Phase 2: DC architecture, math, losses, all decisions
  Mamba-ASR/
    train_S2S.py                  # Seq2Seq training (baselines)
    train_CTC.py                  # CTC training (baselines)
    train_s2s_hmamba.py           # H-Mamba training script
    librispeech_prepare.py        # Data preparation
    hparams/
      S2S/
        conformer_{large,small}.yaml
        conmamba_{large,small}.yaml
        conmambamamba_{large,small}.yaml
        hmamba_{small,large}_N{1,2,3,4}.yaml
      CTC/
        conformer_large.yaml
        conmamba_large.yaml
    modules/
      Conformer.py                # Conformer encoder
      Conmamba.py                 # ConMamba encoder + Mamba decoder
      Transformer.py              # Transformer utilities
      TransformerASR.py           # Top-level ASR model
      HMambaEncoder.py            # DC components: Routing, Chunk, DeChunk, loss
      HMambaEncoderWrapper.py     # Wraps ConMamba with DC
      hmamba_logger.py            # H-Mamba metrics logger + TensorBoard
      mamba/
        bimamba.py                # Bidirectional Mamba
        mamba_blocks.py           # Mamba decoder blocks
        selective_scan_interface.py  # Patched for causal-conv1d 1.4.0
  slurm/                          # SLURM job scripts (8 hmamba + smoke test)
  logs/                           # SLURM job output logs
```

---

## 8. Citation

```bibtex
@misc{jiang2024speechslytherin,
    title={Speech Slytherin: Examining the Performance and Efficiency of Mamba
           for Speech Separation, Recognition, and Synthesis},
    author={Xilin Jiang and Yinghao Aaron Li and Adrian Nicolas Florea
            and Cong Han and Nima Mesgarani},
    year={2024},
    eprint={2407.09732},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
}
```
