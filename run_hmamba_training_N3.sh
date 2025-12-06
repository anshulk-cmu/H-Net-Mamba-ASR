#!/bin/bash
#
# H-Mamba S2S Training Script
# Single GPU (A6000), LibriSpeech-100h
# Expected runtime: ~14-15 hours
#
set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hnetasr

echo "=== H-Mamba S2S ASR Training ==="
echo "Configuration:"
echo "  - Dataset: LibriSpeech train-clean-100 (100h)"
echo "  - GPU: 1x A6000 (48GB)"
echo "  - Batch size: 16 × 2 (grad accum) = 32"
echo "  - Precision: BF16"
echo "  - Epochs: 200"
echo "  - H-Mamba split_idx: 6"
echo "  - H-Mamba target_N: 3.0"
echo ""

# Navigate to Mamba-ASR directory
cd ~/hnet_mamba_asr/Mamba-ASR

# Run single-GPU training
python train_s2s_hmamba.py hparams/S2S/hmamba_S_S2S_N3.yaml \
    --data_folder /home/anshulk/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16

echo ""
echo "=== Training Complete ==="
