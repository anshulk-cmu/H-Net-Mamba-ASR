# H-Mamba ASR

**Hierarchical Mamba with Dynamic Chunking for Efficient Speech Recognition**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.4+-red.svg)](https://pytorch.org/)

## Overview

H-Mamba combines Mamba-based ASR with learned dynamic chunking. The encoder learns acoustic boundaries (phoneme transitions, silence) and adaptively compresses sequences mid-network, achieving 30-50% speedup while maintaining WER.

**Architecture:** Mamba (6 layers) → Dynamic Chunking → Mamba (6 layers) → Decoder

Built on [Mamba-ASR](https://github.com/xi-j/Mamba-ASR) and [H-Net](https://github.com/goombalab/hnet).

## Results

### LibriSpeech 100h

| Model | test-clean WER | test-other WER | Parameters | Speed |
|-------|----------------|----------------|------------|-------|
| **ConMamba (baseline)** | **5.77%** | 17.19% | 14.1M | 1.0× |
| H-Mamba N=2 | 5.96% | 16.35% | ~14.3M | 1.3-1.5× |
| H-Mamba N=3 | Running... | Running... | ~14.3M | TBD |
| H-Mamba N=4 | 7.35% | 19.71% | ~14.3M | 1.3-1.5× |
| Conformer | 6.5% | 17.3% | ~30M | — |

### Baseline Training (Seed 7778, 200 epochs)

| Test Set | WER | SER |
|----------|-----|-----|
| test-clean | 5.77% | 50.31% |
| test-other | 17.19% | 76.11% |

## Installation

```bash
# Clone repository
git clone https://github.com/anshulk-cmu/H-Net-Mamba-ASR.git
cd H-Net-Mamba-ASR

# Create environment
conda create -n hnetasr python=3.9
conda activate hnetasr

# Install dependencies
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install mamba-ssm causal-conv1d --no-build-isolation
pip install speechbrain hyperpyyaml sentencepiece pynvml
```

## Data

Download LibriSpeech 100h:

```bash
mkdir -p data && cd data
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz
for f in *.tar.gz; do tar -xzf $f; done
cd ..
```

## Training

### Baseline (ConMamba)

```bash
# Multi-GPU
torchrun --nproc-per-node 4 Mamba-ASR/train_S2S.py \
    Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml \
    --data_folder ./data/LibriSpeech --precision bf16

# Single GPU
python Mamba-ASR/train_S2S.py \
    Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml \
    --data_folder ./data/LibriSpeech --precision bf16
```

### H-Mamba (with Dynamic Chunking)

```bash
# Multi-GPU
torchrun --nproc-per-node 4 Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ./data/LibriSpeech --precision bf16

# Single GPU
python Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ./data/LibriSpeech --precision bf16
```

### Monitor Training

```bash
tail -f Mamba-ASR/results/S2S/hmamba_S_S2S/7778/train_log.txt
```

## Architecture

```
Audio → CNN (4× downsample) → Mamba Stage 0 (layers 0-5)
                                      ↓
                              Dynamic Chunking
                              ├── RoutingModule: learns boundaries via cosine similarity
                              └── ChunkLayer: compresses L → M frames (M ≈ L/N)
                                      ↓
                              Mamba Stage 1 (layers 6-11, compressed)
                                      ↓
                              DeChunk (EMA interpolation M → L)
                                      ↓
                              CTC Head + Attention Decoder
```

### Dynamic Chunking

The RoutingModule computes boundary probabilities using cosine similarity between adjacent frames:

```python
cos_sim = cosine_similarity(frame[t], frame[t+1])
boundary_prob = sigmoid((1 - cos_sim + bias) / temp)
```

High dissimilarity → acoustic boundary → keep frame.

## Configuration

Key parameters in `hmamba_S_S2S.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hmamba_split_idx` | 6 | Insert DC after layer 6 |
| `hmamba_target_N` | 2.0 | 2× compression (keep 50%) |
| `hmamba_dc_loss_weight` | 5.0 | Weight for DC loss |
| `hmamba_warmup_epochs` | 20 | Warm-up from N=1 to target |

### DC Loss Components

```python
loss = bce×1.0 + mean×5.0 + variance×0.5 + entropy×0.5 + ratio×10.0
```

## Evaluation

```bash
# With LM rescoring
python Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ./data/LibriSpeech --skip_train True

# Without LM
python Mamba-ASR/train_s2s_hmamba.py \
    Mamba-ASR/hparams/S2S/hmamba_S_S2S.yaml \
    --data_folder ./data/LibriSpeech --skip_train True --no_lm True
```

## Project Structure

```
H-Net-Mamba-ASR/
├── Mamba-ASR/
│   ├── train_S2S.py                    # Baseline training
│   ├── train_s2s_hmamba.py             # H-Mamba training
│   ├── modules/
│   │   ├── HMambaEncoder.py            # DC: RoutingModule, ChunkLayer, DeChunkLayer
│   │   ├── HMambaEncoderWrapper.py     # Wraps ConMamba with DC
│   │   ├── hmamba_logger.py            # Training logger
│   │   ├── Conmamba.py                 # Bidirectional Mamba encoder
│   │   └── mamba/                      # Mamba SSM implementations
│   ├── hparams/S2S/
│   │   ├── conmamba_small_ls100.yaml   # Baseline config
│   │   ├── hmamba_S_S2S.yaml           # H-Mamba N=2 config
│   │   └── hmamba_S_S2S_N4.yaml        # H-Mamba N=4 config
│   └── results/                        # Training outputs
├── hnet/                               # H-Net reference implementation
├── data/LibriSpeech/                   # Dataset (not tracked)
├── run_baseline_training.sh
├── run_hmamba_training_N2.sh
└── run_hmamba_training_N4.sh
```

## Key Files

| File | Purpose |
|------|---------|
| `train_s2s_hmamba.py` | Training script with DC loss integration |
| `HMambaEncoder.py` | Core DC components: Router, Chunk, DeChunk |
| `HMambaEncoderWrapper.py` | Wraps ConMamba, adds DC layer at split_idx |
| `hmamba_logger.py` | Logs compression ratio, GPU memory, RTF |
| `hmamba_S_S2S.yaml` | H-Mamba hyperparameters |

## Troubleshooting

**CUDA OOM:** Reduce batch size or enable gradient accumulation
```bash
--batch_size 8 --grad_accumulation_factor 4
```

**Mamba kernel error:**
```bash
pip install mamba-ssm --no-build-isolation
python -c "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; print('OK')"
```

**Compression not converging:** Increase `hmamba_dc_loss_weight` or extend warmup

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023)
- [ConMamba for ASR](https://arxiv.org/abs/2401.10166) (Jiang et al., 2024)
- [H-Net](https://github.com/goombalab/hnet) (GoombaLab)

## License

- Mamba-ASR: MIT License
- H-Net: Apache 2.0 License

## Author

**Anshul Kumar** — Carnegie Mellon University
