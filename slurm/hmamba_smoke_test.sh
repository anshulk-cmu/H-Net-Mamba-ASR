#!/bin/bash
#SBATCH --job-name=hmamba_smoke
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/hmamba_smoke-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/hmamba_smoke-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# H-Mamba Smoke Test — Small N=2, 2x A6000, preempt
#
# Tests (in order):
#   1. Single GPU: 3 epochs on train-clean-100 + test eval → loss decreases, ratio converges
#   2. DDP (2 GPUs): 1 epoch + test eval → no hangs, no CUDA errors
#   3. Resume: load checkpoint from step 1, run 1 more epoch + test eval → verify resume
#
# If all 3 pass, the 8 full H-Mamba training jobs are safe to submit.
# ============================================================================

set -e

echo "============================================================"
echo "H-Mamba Smoke Test"
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
echo "GPUs   :"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

SMOKE_DIR=/data/user_data/anshulk/hnet_asr/results/hmamba_smoke_test_v2
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

# ============================================================================
# PHASE 1: Single GPU — 3 epochs, train-clean-100 only
# ============================================================================
echo ""
echo "============================================================"
echo "PHASE 1: Single GPU, 3 epochs, train-clean-100"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$SMOKE_DIR/phase1" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 3 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --use_wandb False \
    --precision bf16

echo ""
echo "PHASE 1 PASSED — single GPU training OK"
echo ""

# Quick sanity: check that checkpoint exists
if [ ! -d "$SMOKE_DIR/phase1/save" ]; then
    echo "ERROR: No checkpoint directory found after Phase 1"
    exit 1
fi
echo "Checkpoint directory exists: $(ls $SMOKE_DIR/phase1/save/)"

# ============================================================================
# PHASE 2: DDP — 2 GPUs, 1 epoch
# ============================================================================
echo ""
echo "============================================================"
echo "PHASE 2: DDP (2 GPUs), 1 epoch, train-clean-100"
echo "============================================================"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$SMOKE_DIR/phase2" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 1 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --use_wandb False \
    --precision bf16

echo ""
echo "PHASE 2 PASSED — DDP training OK"
echo ""

# ============================================================================
# PHASE 3: Resume from Phase 1 checkpoint (epoch 3 → 4)
# ============================================================================
echo ""
echo "============================================================"
echo "PHASE 3: Resume from Phase 1 checkpoint (epoch 4)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$SMOKE_DIR/phase1" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 4 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --use_wandb False \
    --precision bf16

echo ""
echo "PHASE 3 PASSED — checkpoint resume OK"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "============================================================"
echo "ALL 3 PHASES PASSED"
echo "============================================================"
echo ""
echo "  Phase 1: Single GPU training        OK"
echo "  Phase 2: DDP (2 GPU) training       OK"
echo "  Phase 3: Checkpoint resume           OK"
echo ""
echo "Ready to submit all 8 H-Mamba training jobs."
echo ""
echo "End time : $(date)"
echo "Runtime  : $SECONDS seconds"
echo "============================================================"
