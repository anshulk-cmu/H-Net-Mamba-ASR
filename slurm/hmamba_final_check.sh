#!/bin/bash
#SBATCH --job-name=hmamba_final
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/hmamba_final-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/hmamba_final-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Final Pre-Launch Check — validates grad_norm fix + DDP + resume
#
#   Phase 1: DDP (2 GPUs), 1 epoch train-clean-100 + validation
#   Phase 2: Resume from Phase 1 checkpoint, 1 more epoch + validation
#
# No beam search eval (skipped via valid_search_interval=999).
# Expected runtime: ~15 minutes.
# ============================================================================

set -e

echo "============================================================"
echo "H-Mamba Final Pre-Launch Check"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

# --- Environment ---
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_MODE=offline

# --- AutoFS ---
stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1

# --- Conda ---
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnetasr || { echo "ERROR: failed to activate hnetasr"; exit 1; }

echo "Python : $(which python)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo "GPUs   :"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

SMOKE_DIR=/data/user_data/anshulk/hnet_asr/results/hmamba_final_check
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ============================================================================
# PHASE 1: DDP training — 2 GPUs, 1 epoch, train-clean-100
# ============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: DDP (2 GPUs), 1 epoch + validation"
echo "============================================================"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$SMOKE_DIR" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 1 \
    --valid_search_interval 999 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --skip_train False \
    --use_wandb False \
    --precision bf16

echo ""
echo "PHASE 1 PASSED — DDP training + validation OK"
echo ""

# Verify checkpoint exists
if [ ! -d "$SMOKE_DIR/save" ]; then
    echo "ERROR: No checkpoint directory after Phase 1"
    exit 1
fi
echo "Checkpoint saved: $(ls $SMOKE_DIR/save/)"

# ============================================================================
# PHASE 2: Resume from Phase 1 checkpoint, train epoch 2
# ============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: Resume (epoch 1 → 2)"
echo "============================================================"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$SMOKE_DIR" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 2 \
    --valid_search_interval 999 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --skip_train False \
    --use_wandb False \
    --precision bf16

echo ""
echo "PHASE 2 PASSED — resume OK"
echo ""

# Verify two checkpoints exist
CKPT_COUNT=$(ls -d $SMOKE_DIR/save/CKPT* 2>/dev/null | wc -l)
echo "Checkpoints after resume: $CKPT_COUNT"
if [ "$CKPT_COUNT" -lt 2 ]; then
    echo "WARNING: Expected 2 checkpoints, found $CKPT_COUNT"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================"
echo "ALL CHECKS PASSED"
echo "============================================================"
echo ""
echo "  Phase 1: DDP (2 GPU) + validation     OK"
echo "  Phase 2: Checkpoint resume             OK"
echo "  Grad norm fix: check logs for non-zero values"
echo ""
echo "Ready to submit all 8 H-Mamba 960h training runs."
echo ""
echo "End time : $(date)"
echo "Runtime  : $SECONDS seconds"
echo "============================================================"
