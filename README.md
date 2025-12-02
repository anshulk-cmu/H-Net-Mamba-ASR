# H-Net-Mamba-ASR

Hierarchical Mamba ASR with Dynamic Chunking for Efficient Speech Recognition

## Overview

This project combines two innovations for efficient automatic speech recognition:

1. **Mamba Encoder**: Linear-time state-space models replacing Transformer attention
2. **Dynamic Chunking (H-Net)**: Learned adaptive frame compression for redundant speech segments

## Project Structure

```
hnet_mamba_asr/
├── Mamba-ASR/           # ConMamba baseline
├── hnet/                # H-Net dynamic chunking modules
├── espnet/              # ESPnet toolkit (for utilities)
├── data/                # LibriSpeech dataset (not tracked)
├── run_baseline_training.sh
└── TRAINING_SUMMARY.md
```

## Setup

### 1. Environment Setup

```bash
conda create -n hnetasr python=3.9
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
- Epochs: 200
- Achieved WER: **5.77%** (test-clean with LM)

**Run training:**

```bash
srun --partition=debug --gres=gpu:A6000:4 --cpus-per-task=12 --mem=64G --time=24:00:00 --pty bash
conda activate hnetasr
cd ~/hnet_mamba_asr

torchrun --nproc-per-node 4 Mamba-ASR/train_S2S.py \
  Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml \
  --data_folder ~/hnet_mamba_asr/data/LibriSpeech \
  --precision bf16
```

**Training time:** ~10-12 hours on 4x A6000 (200 epochs)

## Model Architecture

### ConMamba Baseline

- **Encoder:** 12 bidirectional Mamba blocks (d_model=144, d_state=16, expand=2)
- **Decoder:** 4 Transformer layers (nhead=4, d_ffn=1024)
- **Frontend:** 2-layer CNN (64→32 channels, 4× subsampling)
- **Training:** Hybrid CTC/Attention (ctc_weight=0.3)
- **Decoding:** Beam search (size=66) with external TransformerLM (weight=0.6)
- **Parameters:** 14.1M

### H-Mamba (Planned)

- **Stage 0:** 6 bi-Mamba blocks (frame-level processing)
- **Dynamic Chunking Layer:** Learned boundary detection + downsampling (~50% compression)
- **Stage 1:** 6 bi-Mamba blocks (chunk-level, 2× fewer tokens)
- **Upsample + CTC head:** Restore sequence length for CTC alignment
- **Target:** Maintain ≤6% WER while achieving 30-50% RTF reduction

## Configuration

Key settings in `Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml`:

```yaml
# Data
data_folder: /home/anshulk/hnet_mamba_asr/data/LibriSpeech
train_splits: ["train-clean-100"]

# Training
batch_size: 16
grad_accumulation_factor: 2
number_of_epochs: 200
precision: bf16
lr_adam: 0.0015
n_warmup_steps: 6000

# Regularization
transformer_dropout: 0.15
label_smoothing: 0.1

# Model
d_model: 144
num_encoder_layers: 12
num_decoder_layers: 4
d_state: 16
expand: 2
d_conv: 4
bidirectional: true

# Decoding
test_beam_size: 66
lm_weight: 0.60
ctc_weight_decode: 0.40
avg_checkpoints: 10
```

## Results

### ConMamba Baseline — LibriSpeech 100h (Seed 7778)

**Test Performance (Epoch 200, 10-checkpoint average):**

| Test Set | WER | Errors (ins/del/sub) | SER |
|----------|-----|----------------------|-----|
| **test-clean** | **5.77%** | 3034 (449/213/2372) | 50.31% |
| **test-other** | **17.19%** | 9000 (1272/786/6942) | 76.11% |

**Comparison with ESPnet LibriSpeech-100h Baselines:**

| Model | test-clean | test-other | Parameters |
|-------|------------|------------|------------|
| **ConMamba (ours)** | **5.77%** | 17.19% | **14.1M** |
| Multiconvformer | 6.2% | 17.0% | 37.21M |
| E-Branchformer | 6.3% | 17.0% | 38.47M |
| Conformer | 6.5% | 17.3% | ~30M |
| Transformer | 8.4% | 20.5% | ~30M |

**Training Trajectory:**

| Epoch | LR | Valid Loss | Valid ACC | Valid WER |
|-------|-----|------------|-----------|-----------|
| 70 | 7.17e-04 | 93.28 | 51.2% | 55.1% |
| 100 | 1.02e-03 | 43.54 | 79.5% | 21.3% |
| 150 | 1.48e-03 | 27.34 | 87.1% | 12.9% |
| 200 | 1.28e-03 | 23.53 | 89.1% | 10.85% |

**Key Findings:**

- **Best-in-class efficiency:** Achieves lowest test-clean WER (5.77%) with 2.6× fewer parameters than competitive models
- **LM rescoring:** Reduces WER from 10.85% (valid) to 5.77% (test-clean) — 5.08% absolute improvement
- **Generalization gap:** 11.4% gap between test-clean and test-other indicates need for noise augmentation (MUSAN/RIR) in future work

### H-Mamba Integration (Planned)

| Model | test-clean | test-other | RTF | Speedup |
|-------|------------|------------|-----|---------|
| ConMamba (baseline) | 5.77% | 17.19% | 1.0× | — |
| H-Mamba (target) | ≤6% | TBD | TBD | 30-50% |

## Next Steps

1. **Analyze H-Net DC layer:** Study `hnet/hnet/modules/dc.py` for router and downsampler implementation
2. **Integrate DC into ConMamba:** Insert Dynamic Chunking layer after encoder layer 6
3. **Train H-Mamba:** Use same hyperparameters, add ratio loss for compression regularization
4. **Evaluate:** Compare WER, RTF, and VRAM usage; analyze learned boundaries vs phoneme boundaries

## References

- **Mamba:** [Gu & Dao, 2023](https://arxiv.org/abs/2312.00752) — Efficient sequence modeling with selective state spaces
- **ConMamba ASR:** [Jiang et al., 2024](https://github.com/xi-j/Mamba-ASR) — Mamba encoders for speech recognition
- **H-Net:** [GoombaLab](https://github.com/goombalab/hnet) — Dynamic chunking for hierarchical sequence processing

## License

This project combines code from multiple sources. See individual repositories for licenses:
- Mamba-ASR: [xi-j/Mamba-ASR](https://github.com/xi-j/Mamba-ASR)
- H-Net: [goombalab/hnet](https://github.com/goombalab/hnet)
- ESPnet: [espnet/espnet](https://github.com/espnet/espnet)