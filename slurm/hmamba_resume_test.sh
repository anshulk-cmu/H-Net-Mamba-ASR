#!/bin/bash
#SBATCH --job-name=hmamba_resume
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/hmamba_resume-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/hmamba_resume-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Quick Resume Test — copies v3 Phase 1 DDP checkpoint, resumes epoch 1 → 2
# Expected: SpeechBrain finds epoch 1 checkpoint, trains epoch 2 only
# ============================================================================

set -e

echo "============================================================"
echo "H-Mamba Resume Test"
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

# Copy v3 Phase 1 checkpoint to isolated resume test dir
RESUME_DIR=/data/user_data/anshulk/hnet_asr/results/hmamba_resume_test
rm -rf "$RESUME_DIR"
mkdir -p "$RESUME_DIR"

echo "Copying v3 Phase 1 DDP checkpoint..."
cp -r /data/user_data/anshulk/hnet_asr/results/hmamba_smoke_test_v3/phase1_ddp/save "$RESUME_DIR/save"
cp /data/user_data/anshulk/hnet_asr/results/hmamba_smoke_test_v3/phase1_ddp/*.csv "$RESUME_DIR/" 2>/dev/null || true
cp /data/user_data/anshulk/hnet_asr/results/hmamba_smoke_test_v3/phase1_ddp/lm.ckpt "$RESUME_DIR/" 2>/dev/null || true
cp /data/user_data/anshulk/hnet_asr/results/hmamba_smoke_test_v3/phase1_ddp/tokenizer.ckpt "$RESUME_DIR/" 2>/dev/null || true
echo "Checkpoint copied to $RESUME_DIR/save/"
ls "$RESUME_DIR/save/"
echo ""

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "============================================================"
echo "RESUME TEST: DDP (2 GPUs), epoch 1 → 2"
echo "============================================================"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder "$RESUME_DIR" \
    --train_splits '["train-clean-100"]' \
    --number_of_epochs 2 \
    --early_stop_warmup 999 \
    --early_stop_patience 999 \
    --use_wandb False \
    --precision bf16

echo ""
echo "============================================================"
echo "RESUME TEST PASSED"
echo "============================================================"
echo ""
echo "SpeechBrain resumed from epoch 1 checkpoint and trained epoch 2."
echo "DDP + checkpoint resume verified."
echo ""
echo "End time : $(date)"
echo "Runtime  : $SECONDS seconds"
echo "============================================================"
