#!/bin/bash
#
# ConMamba Baseline Training Script
# 4x A6000 GPUs, LibriSpeech-100h
# Expected runtime: 4-5 hours
#

set -e

echo "=== ConMamba ASR Training ==="
echo "Configuration:"
echo "  - Dataset: LibriSpeech train-clean-100 (100h)"
echo "  - GPUs: 4x A6000 (48GB each)"
echo "  - Global batch size: 16 × 4 × 2 = 128"
echo "  - Precision: BF16"
echo "  - Epochs: 70"
echo ""

# Navigate to Mamba-ASR directory
cd ~/hnet_mamba_asr/Mamba-ASR

# Run distributed training
torchrun --nproc-per-node 4 \
    train_S2S.py hparams/S2S/conmamba_small_ls100.yaml \
    --data_folder /home/anshulk/hnet_mamba_asr/data/LibriSpeech \
    --precision bf16

echo ""
echo "=== Training Complete ==="
echo "Check results in: Mamba-ASR/results/S2S/conmamba_S_S2S/7775/"
