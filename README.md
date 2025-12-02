# H-Net-Mamba-ASR

Hierarchical Mamba ASR with Dynamic Chunking for Efficient Speech Recognition

## Overview

This project combines two innovations for efficient automatic speech recognition:
1. **Mamba Encoder**: Linear-time state-space models replacing Transformer attention
2. **Dynamic Chunking (H-Net)**: Learned adaptive frame compression for redundant speech segments

## Project Structure

```
hnet_mamba_asr/
├── Mamba-ASR/           # ConMamba baseline (modified)
├── hnet/                # H-Net dynamic chunking modules
├── espnet/              # ESPnet toolkit (for utilities)
├── data/                # LibriSpeech dataset (not tracked)
├── run_baseline_training.sh
└── TRAINING_SUMMARY.md
```

## Setup

### 1. Environment Setup

```bash
conda create -n hnetasr python=3.11
conda activate hnetasr

# Install PyTorch with CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Mamba
pip install mamba-ssm causal-conv1d

# Install dependencies
cd Mamba-ASR && pip install speechbrain hyperpyyaml
cd ../espnet && pip install -e .
```

### 2. Download LibriSpeech Data

```bash
cd ~/hnet_mamba_asr
mkdir -p data
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz -P data/
wget https://www.openslr.org/resources/12/dev-clean.tar.gz -P data/
wget https://www.openslr.org/resources/12/test-clean.tar.gz -P data/
wget https://www.openslr.org/resources/12/test-other.tar.gz -P data/
cd data && for f in *.tar.gz; do tar -xzf $f; done && cd ..
```

## Training

### Baseline ConMamba (100h LibriSpeech)

**Configuration:**
- Dataset: train-clean-100 (100 hours)
- GPUs: 4x A6000 48GB
- Global batch: 128 (16 per GPU × 4 GPUs × 2 accumulation)
- Precision: BF16
- Epochs: 70
- Expected WER: 6-8% (test-clean with LM)

**Run training:**
```bash
srun --partition=debug --gres=gpu:A6000:4 --cpus-per-task=12 --mem=64G --time=12:00:00 --pty bash
conda activate hnetasr
cd ~/hnet_mamba_asr
./run_baseline_training.sh
```

**Training time:** ~4-5 hours on 4x A6000

## Model Architecture

### ConMamba Baseline
- **Encoder:** 12 bidirectional Mamba blocks (d_model=144)
- **Decoder:** 4 Transformer layers
- **Training:** Hybrid CTC/Attention (ctc_weight=0.3)
- **Decoding:** Beam search with external Transformer LM

### H-Mamba (Planned)
- **Stage 0:** 8 bi-Mamba blocks (frame-level)
- **Dynamic Chunking:** Learned boundary detection + compression
- **Stage 1:** 4 bi-Mamba blocks (chunk-level, 2-3x fewer tokens)
- **Target:** Maintain WER while achieving 40-60% speedup

## Configuration

Key settings in `Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml`:

```yaml
# Data
data_folder: /home/anshulk/hnet_mamba_asr/data/LibriSpeech
train_splits: ["train-clean-100"]

# Training
batch_size: 16
grad_accumulation_factor: 2
number_of_epochs: 70
precision: bf16

# Regularization
transformer_dropout: 0.15
label_smoothing: 0.1

# Model
d_model: 144
num_encoder_layers: 12
num_decoder_layers: 4
```

## Results

### ConMamba Baseline - LibriSpeech 100h (Seed 7778)

**Architecture:** 12-layer bidirectional Mamba encoder (d_model=144, d_state=16, expand=2, d_conv=4) + 4-layer Transformer decoder. CNN frontend: 2 blocks (64→32 channels, 4× subsampling). **Total parameters: 14.1M**. Vocabulary: 5000 BPE tokens.

**Training Configuration:**
- Dataset: LibriSpeech train-clean-100 (100h), 200 epochs
- Optimization: Adam (lr=0.0015, β=(0.9, 0.98)), Noam scheduler (warmup=6000 steps, peak LR at epoch 146)
- Batch: size=16, grad_accumulation=2 (effective=128), BF16 precision, 4× L40S GPUs
- Regularization: Label smoothing=0.1, SpecAugment (time/freq drop=4), speed perturbation (95-105%)
- Loss: CTC weight=0.3, attention weight=0.7

**Training Trajectory:**

| Epoch | Learning Rate | Valid Loss | Valid ACC | Valid WER |
|-------|---------------|------------|-----------|-----------|
| 10 | 1.02e-04 | 205 | 12.8% | 101% |
| 70 | 7.17e-04 | 93.28 | 51.2% | 55.1% |
| 100 | 1.02e-03 | 43.54 | 79.5% | 21.34% |
| 146 | **1.50e-03** | 28.44 | 86.6% | — |
| 200 | 1.28e-03 | 23.53 | 89.1% | **10.85%** |

**Test Performance (Epoch 200):**
- Decoding: Beam size=66, TransformerLM weight=0.6, CTC weight=0.4, checkpoint averaging=10 epochs

| Test Set | WER | Errors | Total Words | SER |
|----------|-----|--------|-------------|-----|
| **test-clean** | **5.77%** | 3034 (449 ins, 213 del, 2372 sub) | 52,576 | 50.31% |
| **test-other** | **17.19%** | 9000 (1272 ins, 786 del, 6942 sub) | 52,343 | 76.11% |

**Comparison with ESPnet LibriSpeech-100h Baselines:**

| Model | test-clean WER | test-other WER | Parameters |
|-------|----------------|----------------|------------|
| **ConMamba (ours)** | **5.77%** | 17.19% | **14.1M** |
| ESPnet Multiconvformer | 6.2% | 17.0% | 37.21M |
| ESPnet E-Branchformer | 6.3% | 17.0% | 38.47M |
| ESPnet Conformer | 6.5% | 17.3% | ~30M |
| ESPnet Transformer | 8.4% | 20.5% | ~30M |

**Key Insights:**
- **State-of-the-art efficiency:** ConMamba achieves the best test-clean WER (5.77% vs. 6.2%) with **2.6× fewer parameters** than competitive models
- **LM rescoring impact:** Valid WER (10.85%) → test-clean (5.77%) demonstrates 5.08% absolute improvement from TransformerLM integration during decoding
- **Generalization:** Test-other WER (17.19%) reflects challenging acoustic conditions (accents, noise, reverberation). The 11.42% gap from test-clean is expected without noise augmentation (MUSAN/RIR)
- **Training dynamics:** Classic sequence-to-sequence learning: slow warmup (epochs 1-60, WER>80%), rapid acquisition (epochs 60-100, WER 55%→21%), fine-tuning (epochs 100-200, WER 21%→10.85%)

**Reproducibility:** Configuration in [Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml](Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml) (seed=7778).

### H-Mamba Integration (Planned)

| Model | test-clean WER | test-other WER | RTF | Speedup |
|-------|----------------|----------------|-----|---------|
| H-Mamba | TBD | TBD | TBD | Target: 40-60% |

## References

- **Mamba:** [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **ConMamba ASR:** [Jiang et al., 2024](https://github.com/xi-j/Mamba-ASR)
- **H-Net:** [Dynamic Chunking Paper](https://github.com/goombalab/hnet)

## License

This project combines code from multiple sources. See individual repositories for licenses:
- Mamba-ASR: [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)
- H-Net: [goombalab/hnet](https://github.com/goombalab/hnet)
- ESPnet: [espnet/espnet](https://github.com/espnet/espnet)
