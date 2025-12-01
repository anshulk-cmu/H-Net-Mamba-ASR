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

| Model | test-clean WER | test-other WER | RTF |
|-------|----------------|----------------|-----|
| ConMamba Baseline | TBD | TBD | TBD |
| H-Mamba (planned) | TBD | TBD | TBD |

## References

- **Mamba:** [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)
- **ConMamba ASR:** [Jiang et al., 2024](https://github.com/xi-j/Mamba-ASR)
- **H-Net:** [Dynamic Chunking Paper](https://github.com/goombalab/hnet)

## License

This project combines code from multiple sources. See individual repositories for licenses:
- Mamba-ASR: [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)
- H-Net: [goombalab/hnet](https://github.com/goombalab/hnet)
- ESPnet: [espnet/espnet](https://github.com/espnet/espnet)
