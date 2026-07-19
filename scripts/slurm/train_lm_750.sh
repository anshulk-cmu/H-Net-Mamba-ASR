#!/bin/bash
# =============================================================================
# External Transformer LM, 750-vocab ablation arm (plan #6).
# 16L/512d, 2x A6000 DDP (torchrun), 32k tokens/GPU (exact-length packing =
# hard memory bound), epoch-resumable (--resume auto survives preemption).
# =============================================================================

#SBATCH --job-name=dcasr_lm750
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-lm750-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-lm750-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:2
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

# any-GPU safety: on <40GB cards halve the per-GPU batch and double accumulation,
# keeping the global 64k tokens/step identical to the 500 arm (controlled ablation)
MINMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
OV=""
if [ "$MINMEM" -lt 40000 ]; then
  OV="batch_tokens=16000 accum_grad=2"
  echo "[$(date)] small-VRAM GPUs (${MINMEM} MiB): using $OV (global batch unchanged)"
fi

$ENVBIN/torchrun --standalone --nproc_per_node=2 scripts/train_lm.py \
  --config configs/lm_transformer_750.yaml --resume auto $OV

echo "[$(date)] lm750 training done"
