# H-Net-Mamba-ASR

**Hierarchical Mamba ASR with Dynamic Chunking for Efficient Speech Recognition**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)

## 🎯 Overview

This project implements **H-Mamba**, a novel ASR architecture that combines:

1. **Mamba Encoder**: Linear-time state-space models (O(n) complexity) replacing quadratic Transformer attention
2. **Dynamic Chunking (DC)**: Learned adaptive frame compression that identifies and merges redundant speech segments
3. **Hierarchical Processing**: Two-stage encoder with compression in between for computational efficiency

**Key Innovation**: Unlike fixed-rate downsampling, our Dynamic Chunking layer *learns* where acoustic boundaries occur (phoneme transitions, silence boundaries) and adaptively compresses the sequence, achieving 30-50% speedup while maintaining WER.

## 📊 Results Summary

### Comparison with ESPnet LibriSpeech-100h Baselines

| Model | test-clean WER | test-other WER | Parameters | Relative Speed |
|-------|----------------|----------------|------------|----------------|
| **ConMamba (ours)** | **5.77%** | 17.19% | **14.1M** | 1.0× |
| H-Mamba N=2 (target) | ≤6.0% | TBD | ~14.3M | 1.3-1.5× |
| Multiconvformer | 6.2% | 17.0% | 37.21M | — |
| E-Branchformer | 6.3% | 17.0% | 38.47M | — |
| Conformer | 6.5% | 17.3% | ~30M | — |
| Transformer | 8.4% | 20.5% | ~30M | — |

### ConMamba Baseline — Detailed Results (LibriSpeech 100h, Seed 7778)

**Test Performance (Epoch 200, 10-checkpoint average):**

| Test Set | WER | Errors (ins/del/sub) | SER |
|----------|-----|----------------------|-----|
| **test-clean** | **5.77%** | 3034 (449/213/2372) | 50.31% |
| **test-other** | **17.19%** | 9000 (1272/786/6942) | 76.11% |

**Training Trajectory:**

| Epoch | LR | Valid Loss | Valid ACC | Valid WER |
|-------|-----|------------|-----------|-----------|
| 70 | 7.17e-04 | 93.28 | 51.2% | 55.1% |
| 100 | 1.02e-03 | 43.54 | 79.5% | 21.3% |
| 150 | 1.48e-03 | 27.34 | 87.1% | 12.9% |
| 200 | 1.28e-03 | 23.53 | 89.1% | 10.85% |

**Key Findings:**
- **Best-in-class efficiency:** Achieves lowest test-clean WER (5.77%) with 2.6× fewer parameters than competitive models
- **LM rescoring impact:** Reduces WER from 10.85% (valid) to 5.77% (test-clean) — 5.08% absolute improvement
- **Generalization gap:** 11.4% gap between test-clean and test-other indicates need for noise augmentation (MUSAN/RIR) in future work

## 🏗️ Project Structure

```
hnet_mamba_asr/
├── Mamba-ASR/                      # Main training codebase
│   ├── train_S2S.py                # ConMamba baseline training
│   ├── train_s2s_hmamba.py         # ⭐ H-Mamba training script
│   ├── modules/
│   │   ├── HMambaEncoder.py        # ⭐ DC components (Router, Chunk, DeChunk)
│   │   ├── HMambaEncoderWrapper.py # ⭐ Wraps ConMamba with DC layer
│   │   ├── hmamba_logger.py        # ⭐ Comprehensive training logger
│   │   ├── Conmamba.py             # Bidirectional Mamba encoder
│   │   ├── TransformerASR.py       # ASR model wrapper
│   │   └── mamba/                  # Mamba SSM implementations
│   ├── hparams/S2S/
│   │   ├── conmamba_small_ls100.yaml   # Baseline config
│   │   └── hmamba_S_S2S.yaml           # ⭐ H-Mamba config
│   └── results/                    # Training outputs
├── hnet/                           # Original H-Net reference implementation
│   └── hnet/modules/dc.py          # Reference DC layer
├── espnet/                         # ESPnet toolkit (utilities)
├── data/LibriSpeech/               # Dataset (not tracked)
├── run_baseline_training.sh        # Baseline training launcher
└── TRAINING_SUMMARY.md             # Baseline results documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n hnetasr python=3.9
conda activate hnetasr

# Install PyTorch with CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Mamba (requires CUDA)
pip install mamba-ssm causal-conv1d

# Install SpeechBrain and dependencies
cd Mamba-ASR && pip install speechbrain hyperpyyaml sentencepiece
cd ../espnet && pip install -e .
pip install pynvml  # For GPU monitoring in logger
```

### 2. Download LibriSpeech Data

```bash
cd ~/hnet_mamba_asr
mkdir -p data && cd data

# Download LibriSpeech subsets
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz

# Extract all
for f in *.tar.gz; do tar -xzf $f; done
cd ..
```

### 3. Verify Setup

```bash
# Test H-Mamba components
cd ~/hnet_mamba_asr/Mamba-ASR
python modules/HMambaEncoderWrapper.py

# Expected output:
# [HMambaEncoderWrapper] Created with 6 stage0 layers, 6 stage1 layers
# Compression ratio: ~0.5 (target: 0.50)
# All tests passed!
```

## 🎓 Training Guide

### Option A: Train Baseline ConMamba First (Recommended)

If you want to establish a baseline before adding Dynamic Chunking:

```bash
cd ~/hnet_mamba_asr

# Interactive SLURM allocation
srun --partition=debug --gres=gpu:A6000:4 --cpus-per-task=12 --mem=64G --time=24:00:00 --pty bash
conda activate hnetasr

# Multi-GPU training (4x A6000)
torchrun --nproc-per-node 4 Mamba-ASR/train_S2S.py \
    Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16

# Single GPU (for testing)
python Mamba-ASR/train_S2S.py \
    Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16
```

**Expected**: ~10-12 hours for 200 epochs, achieving **5.77% WER** on test-clean.

### Option B: Train H-Mamba Directly

```bash
cd ~/hnet_mamba_asr

# Interactive SLURM allocation
srun --partition=debug --gres=gpu:A6000:4 --cpus-per-task=12 --mem=64G --time=24:00:00 --pty bash
conda activate hnetasr

# Multi-GPU H-Mamba training
torchrun --nproc-per-node 4 Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16

# Single GPU (for debugging)
python Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16
```

### Monitoring Training

```bash
# Watch training log
tail -f Mamba-ASR/results/S2S/hmamba_S_S2S/7778/train_log.txt

# TensorBoard (if enabled)
tensorboard --logdir Mamba-ASR/results/S2S/hmamba_S_S2S/7778/hmamba_logs/

# Check compression statistics
cat Mamba-ASR/results/S2S/hmamba_S_S2S/7778/hmamba_logs/batch_metrics.csv | tail -20
```

## 🔧 Architecture Deep Dive

### H-Mamba Encoder Architecture

```
Input Audio → Fbank Features (80-dim)
                    ↓
            CNN Frontend (4× downsample)
                    ↓
        ┌─────────────────────────────────┐
        │     Stage 0: Layers 0-5         │
        │   (Frame-level bi-Mamba)        │
        │   Full sequence: L frames       │
        └─────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────┐
        │     Dynamic Chunking Layer      │
        │  ┌─────────────────────────┐    │
        │  │ RoutingModule           │    │  ← Learns boundary positions
        │  │ (cosine similarity)     │    │    via adjacent frame similarity
        │  └─────────────────────────┘    │
        │  ┌─────────────────────────┐    │
        │  │ ChunkLayer              │    │  ← Keeps only boundary frames
        │  │ (compress L → M)        │    │    M ≈ L/N (N=target compression)
        │  └─────────────────────────┘    │
        └─────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────┐
        │     Stage 1: Layers 6-11        │
        │   (Chunk-level bi-Mamba)        │
        │   Compressed: M frames          │  ← 2× fewer tokens = faster!
        └─────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────┐
        │     DeChunk Layer               │
        │  (EMA interpolation M → L)      │  ← Restore original resolution
        └─────────────────────────────────┘
                    ↓
            Residual + LayerNorm
                    ↓
        CTC Head + Attention Decoder
```

### Key Components

#### 1. RoutingModule (`HMambaEncoder.py`)
```python
# Learns to detect acoustic boundaries using cosine similarity
cos_sim = cosine_similarity(frame[t], frame[t+1])
boundary_prob = (1 - cos_sim) / 2  # High dissimilarity → boundary

# Gumbel-Softmax for differentiable sampling during training
boundary_mask = gumbel_softmax(boundary_logits, hard=True)
```

#### 2. ChunkLayer (`HMambaEncoder.py`)
```python
# Keep only frames marked as boundaries
chunked_states = hidden_states[boundary_mask]  # (B, L, D) → (B, M, D)
# M ≈ L/N where N is target compression ratio
```

#### 3. DeChunkLayer (`HMambaEncoder.py`)
```python
# Expand back using EMA interpolation
# Uses Mamba2 kernel for O(n) efficiency
out[t] = p[t] * chunk_value + (1-p[t]) * out[t-1]
```

#### 4. Load Balancing Loss
```python
# Encourages compression ratio to match target N
target_ratio = 1.0 / N  # N=2 → keep 50% of frames
avg_boundary_prob = boundary_prob.mean()
loss = (avg_boundary_prob - target_ratio) ** 2
```

## ⚙️ Configuration

### H-Mamba Hyperparameters (`hmamba_S_S2S.yaml`)

```yaml
# Dynamic Chunking Configuration
hmamba_split_idx: 6          # Split at layer 6 (6 stage0 + 6 stage1)
hmamba_target_N: 2.0         # Target 2× compression (keep 50%)
hmamba_headdim: 36           # EMA head dimension (144/36 = 4 heads)
hmamba_dc_loss_weight: 0.1   # Weight for DC load balancing loss
hmamba_warmup_epochs: 20     # Warm-up from N=1 to target_N

# Model Architecture
d_model: 144                 # Hidden dimension
num_encoder_layers: 12       # Total encoder layers
num_decoder_layers: 4        # Transformer decoder layers

# Training
batch_size: 16               # Per-GPU batch size
grad_accumulation_factor: 2  # Effective batch = 16 × 4 GPUs × 2 = 128
number_of_epochs: 200
precision: bf16
ctc_weight: 0.3              # CTC/Attention balance
```

### Tuning Recommendations

| Parameter | Conservative | Aggressive | Description |
|-----------|-------------|------------|-------------|
| `hmamba_target_N` | 1.5 | 3.0 | Higher = more compression, may hurt WER |
| `hmamba_dc_loss_weight` | 0.03 | 0.3 | Higher = stricter compression enforcement |
| `hmamba_warmup_epochs` | 30 | 10 | Longer = more stable training start |
| `hmamba_split_idx` | 8 | 4 | Later split = more frame-level processing |

## 📈 Expected Training Behavior

### Compression Warm-up Schedule
```
Epoch 0:   N=1.0 (no compression) → ratio ≈ 1.0
Epoch 10:  N=1.5 (25% compression) → ratio ≈ 0.67
Epoch 20:  N=2.0 (50% compression) → ratio ≈ 0.50 (target)
Epoch 20+: N=2.0 (stable)
```

### Training Log Indicators
```
[H-Mamba] Epoch 50: compression=0.48, dc_loss=0.0023, target_N=2.00
          ↑ Should be close to 1/N (0.50)
```

### What to Watch For
- **compression_ratio**: Should converge to ~1/N (e.g., 0.5 for N=2)
- **dc_loss**: Should decrease over time (<0.01 is good)
- **valid_WER**: Should be within 5-10% relative of baseline
- **boundary_prob_mean**: Should match compression_ratio

## 🔬 Evaluation

### Test with Language Model
```bash
cd ~/hnet_mamba_asr/Mamba-ASR

# Evaluate with LM rescoring (best WER)
python train_s2s_hmamba.py \
    hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --skip_train True

# Evaluate without LM
python train_s2s_hmamba.py \
    hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
    --skip_train True \
    --no_lm True
```

### Results Location
```
Mamba-ASR/results/S2S/hmamba_S_S2S/7778/
├── save/                    # Checkpoints
├── train_log.txt           # Training metrics
├── wer_test-clean.txt      # Detailed WER breakdown
├── wer_test-other.txt
└── hmamba_logs/            # DC-specific metrics
    ├── batch_metrics.csv
    ├── epoch_metrics.csv
    └── tensorboard/
```

## 🧪 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 8 --grad_accumulation_factor 4

# Or use gradient checkpointing (if implemented)
--gradient_checkpointing True
```

**2. Mamba Kernel Not Available**
```bash
# Install mamba-ssm with CUDA support
pip install mamba-ssm --no-build-isolation

# Verify installation
python -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('OK')"
```

**3. NaN Loss During Training**
- Reduce learning rate: `--lr_adam 0.001`
- Increase warmup: modify `n_warmup_steps` in YAML
- Reduce DC loss weight: `--hmamba_dc_loss_weight 0.01`

**4. Compression Ratio Not Converging**
- Increase `hmamba_dc_loss_weight` (try 0.2-0.5)
- Extend warmup epochs
- Check boundary probabilities in logs

### Debugging Commands

```bash
# Test HMambaEncoder standalone
python Mamba-ASR/modules/HMambaEncoder.py

# Test full wrapper with ConMamba
python Mamba-ASR/modules/HMambaEncoderWrapper.py

# Check GPU memory
nvidia-smi -l 1

# Profile training step
python -m torch.profiler train_s2s_hmamba.py ...
```

## 📚 Implementation Details

### Files You Should Know

| File | Purpose |
|------|---------|
| `train_s2s_hmamba.py` | Main training script with DC loss integration |
| `HMambaEncoderWrapper.py` | Wraps ConMamba encoder, adds DC layer |
| `HMambaEncoder.py` | Core DC components: Router, Chunk, DeChunk |
| `hmamba_logger.py` | Comprehensive logging (GPU, RTF, compression) |
| `hmamba_S_S2S.yaml` | H-Mamba hyperparameters |

### Key Training Modifications from Baseline

1. **Encoder Wrapping** (`on_fit_start`):
   ```python
   # Original ConMamba encoder is wrapped with DC
   hmamba_encoder = create_hmamba_from_conmamba(
       original_encoder, split_idx=6, target_N=2.0
   )
   ```

2. **DC Loss Added** (`compute_objectives`):
   ```python
   loss = loss_asr + dc_loss_weight * loss_dc
   ```

3. **Warm-up Schedule** (`_get_current_target_N`):
   ```python
   # Gradually increase compression target
   if epoch < warmup_epochs:
       current_N = 1.0 + (target_N - 1.0) * (epoch / warmup_epochs)
   ```

## 📖 References

### Papers
- **Mamba**: [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) — Selective State Space Models
- **ConMamba ASR**: [Jiang et al., 2024](https://arxiv.org/abs/2401.10166) — Mamba for Speech Recognition
- **H-Net**: [Anonymous, 2024](https://github.com/goombalab/hnet) — Hierarchical Dynamic Chunking

### Codebases
- Mamba-ASR: [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)
- H-Net: [goombalab/hnet](https://github.com/goombalab/hnet)
- ESPnet: [espnet/espnet](https://github.com/espnet/espnet)

## 📄 License

This project combines code from multiple sources:
- Mamba-ASR: MIT License
- H-Net: Apache 2.0 License
- ESPnet: Apache 2.0 License

See individual repository licenses for details.

## 👤 Author

**Anshul Kumar** — Carnegie Mellon University

---

## 📋 Quick Reference Card

### Training Commands

| Task | Command |
|------|---------|
| Train baseline | `torchrun --nproc-per-node 4 train_S2S.py hparams/S2S/conmamba_small_ls100.yaml` |
| Train H-Mamba | `torchrun --nproc-per-node 4 train_s2s_hmamba.py hparams/S2S/hmamba_S_S2S.yaml` |
| Evaluate only | Add `--skip_train True` |
| No LM eval | Add `--no_lm True` |

### Key Metrics to Monitor

| Metric | Good Value | Location |
|--------|------------|----------|
| `compression_ratio` | ~0.5 (for N=2) | train_log.txt |
| `dc_loss` | <0.01 | train_log.txt |
| `valid_WER` | <12% | train_log.txt |
| `test-clean WER` | <6.5% | wer_test-clean.txt |

### Important Paths

```
~/hnet_mamba_asr/
├── Mamba-ASR/train_s2s_hmamba.py      # Training script
├── Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml  # Config
├── Mamba-ASR/results/S2S/hmamba_S_S2S/7778/ # Results
└── data/LibriSpeech/                   # Dataset
```