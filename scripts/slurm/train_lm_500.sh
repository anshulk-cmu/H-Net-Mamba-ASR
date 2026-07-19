#!/bin/bash
# =============================================================================
# External Transformer LM, 500-vocab primary arm (plan #6).
# 16L/512d, 2x A6000 DDP (torchrun), 32k tokens/GPU (exact-length packing =
# hard memory bound), epoch-resumable (--resume auto survives preemption).
# =============================================================================

#SBATCH --job-name=dcasr_lm500
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-lm500-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-lm500-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

echo "[$(date)] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnet-asr
cd /home/anshulk/H-Net_Mamba_ASR

ENVBIN=/data/user_data/anshulk/envs/hnet-asr/bin
export PYTHONDONTWRITEBYTECODE=1
echo "[$(date)] TORCHRUN=$ENVBIN/torchrun"

$ENVBIN/torchrun --standalone --nproc_per_node=2 scripts/train_lm.py \
  --config configs/lm_transformer_500.yaml --resume auto

echo "[$(date)] lm500 training done"
