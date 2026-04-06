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
- Small model (14.1M params): N=1, N=2, N=3, N=4 — **S_N2/S_N3/S_N4 completed, S_N1 timed out ep 200, resubmitted (Apr 6)**
- Large model (115.2M params): N=1, N=2, N=3, N=4 — **L_N1 (ep 108) / L_N2 (ep 112) running (general), L_N3 with-LM done (5.21/10.10, ep 164 running), L_N4 pending preempt (Apr 6)**

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
- Grad norm showed 0.0 on large models (grad_accum=8 misaligned with log interval). Fixed: persist last real grad_norm across non-step batches.
- Smoke test v1 (6912668) timed out during beam search eval (2h limit too short).

### 6.5 H-Mamba Results (960h, updated April 6 evening)

#### Completed runs — test-set WER (beam=66, CTC weight=0.40)

| Model | target_N | Compression | Epochs | Best Epoch | With LM (clean / other) | Without LM (clean / other) | Status |
|-------|----------|-------------|--------|------------|------------------------|---------------------------|--------|
| hmamba_small_N2 | 2.0 | 0.501 | 234 (patience) | 230 | — (eval running, ~96%) | **2.42 / 5.98** | **No-LM done, with-LM running** |
| hmamba_small_N3 | 3.0 | 0.335 | 205 (patience) | 160 | 5.31 / 10.29 | 10.62 / 18.66 | **Done** |
| hmamba_small_N4 | 4.0 | 0.251 | 193 (patience) | 160 | 5.21 / 11.06 | 9.24 / 17.38 | **Done** |
| hmamba_large_N3 | 3.0 | 0.334 | 164+ (still running) | 90 | 5.21 / 10.10 | — (needs eval) | **With-LM done, training continues** |

#### In-progress runs — dev-set WER (valid_search, beam=10, greedy, no LM)

| Model | target_N | Compression | Epoch | ACC | WER (dev) | Status |
|-------|----------|-------------|-------|-----|-----------|--------|
| hmamba_small_N1 | 1.0 | 0.796 | 199 (timed out in 200) | 97.3% | **3.32%** (ep 190) | Resubmitted (job 6987430, general) |
| hmamba_large_N1 | 1.0 | 0.834 | 108 | 97.6% | **2.76%** (ep 80) | Running (job 6965857, general) |
| hmamba_large_N2 | 2.0 | 0.501 | 112 | 97.5% | **2.87%** (ep 110) | Running (job 6968230, general) |
| hmamba_large_N3 | 3.0 | 0.334 | 164 | 79.5% | **7.89%** (ep 90) | Running (job 6965501, preempt) |
| hmamba_large_N4 | 4.0 | 0.251 | 93 | 87.3% | **6.46%** (ep 70) | Pending (job 6965502, preempt) |

**Highlights (April 6):**
- **S_N2 completed** (234 epochs, patience exhausted). Best epoch 230. No-LM test: **2.42 / 5.98**. With-LM eval running (~96%). **Key result: 2.42% test-clean beats ConMamba Small no-LM baseline (3.34%) at 50% compression.**
- **S_N3 completed** (205 epochs, patience exhausted). Best epoch 160. With-LM: **5.31 / 10.29**. No-LM: **10.62 / 18.66**.
- **S_N4 completed** (193 epochs, patience exhausted). Best epoch 160. With-LM: **5.21 / 11.06**. No-LM: **9.24 / 17.38**.
- **S_N1 timed out** in epoch 200 (batch 479/1603) after 2-day wall. Best dev WER **3.32%** (ep 190), still improving. Resubmitted as job 6987430.
- **L_N3 with-LM eval completed**: **5.21 / 10.10** (best epoch 90). Training continues past eval (now ep 164), but best WER was 7.89% at ep 90 — no improvement since.
- **L_N1** (job 6965857), epoch 108. Best WER **2.76%** (ep 80), then 2.77 (ep 90), 2.85 (ep 100) — slight regression. Beats ConMamba Large no-LM baseline (2.82%).
- **L_N2** (job 6968230), epoch 112. Best WER **2.87%** (ep 110), improving from 3.15 (ep 80) → 2.94 (ep 90) → 2.87 (ep 110).
- **L_N4** pending preempt (job 6965502), epoch 93. Best WER **6.46%** (ep 70). ACC regressed from 0.908 (early) to ~0.87.
- S_N4 WER recovered from degradation: 14.61% (ep 70) → 9.70% (ep 160).
- N=3 anomaly confirmed structural: S_N3 plateaued ~10%, L_N3 ~8%, at both model scales.

**Known issues:**
- S_N2 with-LM eval running (job 6986666, ~96% on test-clean) — results expected shortly.
- S_N1 resubmitted (job 6987430) — should finish remaining epoch 200 quickly.
- L_N3 needs no-LM eval submitted (with-LM results already on disk).
- L_N4 stuck pending on preempt — consider moving to general when GPU slots open.

**Evaluation pipeline:** Interim dev WERs use beam=10, CTC only, no LM (valid_search). Final evaluation uses beam=66, CTC(0.40) + TransformerLM(0.60) (test_search). No-LM eval uses beam=66, no LM rescoring. LM decoding typically improves WER by 1.0–1.5% absolute on test-clean. Eval scripts: `slurm/eval_nolm_hmamba_small_N*.sh`, `slurm/eval_withlm_hmamba_small_N2.sh`.

### 6.6 Competitive Landscape (LibriSpeech 960h)

| Model | #Params | WER clean/other | LM | Source |
|-------|---------|----------------|----|--------|
| Conformer (Google) | ~118M | 1.9 / 3.9 | yes | Gulati et al., 2020 |
| E-Branchformer (ESPnet) | ~120M | 1.81 / 3.65 | yes | Kim et al., 2022 |
| Zipformer-S | 23.2M | 2.42 / 5.73 | no | Yao et al., ICLR 2024 |
| Zipformer-L | 148.4M | 2.00 / 4.38 | yes | Yao et al., ICLR 2024 |
| ConMamba Small (ours) | 14.1M | 2.22 / 5.56 | yes | Phase 1 baseline |
| ConMamba Large (ours) | 115.2M | 2.27 / 5.12 | yes | Phase 1 baseline |
| SAMBA-ASR | large | 1.17 / 2.48 | — | Jiang et al., 2025 (multi-dataset, not 960h-only) |

ConMamba Small with LM (2.22/5.56) beats Zipformer-S (2.42/5.73) at 40% fewer parameters. H-Mamba's goal is not SOTA WER but 50% frame compression with negligible degradation from these baselines. See [hmamba_dynamic_chunking.md](docs/hmamba_dynamic_chunking.md) for detailed analysis.

### 6.7 100-Hour Pilot Results (Small model, pre-bug-fix)

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
  slurm/                          # SLURM job scripts (8 hmamba + eval + smoke test)
  logs/                           # SLURM job output logs
```

---

## 8. Citations

ConMamba baseline (Speech Slytherin):
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

H-Net Dynamic Chunking (the "H" in H-Mamba):
```bibtex
@misc{hwang2025dynamicchunkingendtoendhierarchical,
    title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
    author={Sukjun Hwang and Brandon Wang and Albert Gu},
    year={2025},
    eprint={2507.07955},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
}
```
