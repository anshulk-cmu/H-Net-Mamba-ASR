#!/bin/bash
#SBATCH --job-name=hmamba_L_N3
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/hmamba_L_N3-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/hmamba_L_N3-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "H-Mamba Large N3 — 2x A6000, general"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB__SERVICE_WAIT=300
export WANDB_MODE=offline

stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1

source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnetasr || { echo "ERROR: failed to activate hnetasr"; exit 1; }

# Pin numpy to avoid 2.0 DDP crash (broadcast_object_list → tensor.numpy().tobytes())
pip install numpy==1.26.4 --quiet 2>/dev/null
echo "NumPy  : $(python -c 'import numpy; print(numpy.__version__)')"

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

DATA_FOLDER=/data/user_data/anshulk/hnet_asr/LibriSpeech
OUTPUT_FOLDER=/data/user_data/anshulk/hnet_asr/results/hmamba_large_N3
HPARAMS=hparams/S2S/hmamba_large_N3.yaml
TRAIN_ARGS="--batch_size 16 --max_batch_length_train 600 --max_batch_length_val 100 --precision bf16"
EVAL_ARGS="--batch_size 16 --max_batch_length_train 600 --max_batch_length_val 100 --precision bf16 --skip_train True"

# --- Signal trap: if SLURM timeout approaches, kill training so eval can run ---
TRAIN_PID=""
handle_timeout() {
    echo ""
    echo "============================================================"
    echo "SLURM timeout approaching — killing training to run evals — $(date)"
    echo "============================================================"
    if [ -n "$TRAIN_PID" ]; then
        kill -TERM "$TRAIN_PID" 2>/dev/null
        wait "$TRAIN_PID" 2>/dev/null
    fi
}
trap 'handle_timeout' USR1

# --- Phase 1: Training (resumes from checkpoint) ---
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Master port: $MASTER_PORT"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py $HPARAMS \
    --data_folder $DATA_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    $TRAIN_ARGS \
    --use_wandb True &
TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT=$?
TRAIN_PID=""

echo ""
echo "============================================================"
echo "Training finished — exit code: $TRAIN_EXIT — $(date)"
echo "============================================================"

# --- Phase 2: With-LM eval (always runs) ---
echo ""
echo "============================================================"
echo "Starting with-LM eval (single GPU) — $(date)"
echo "============================================================"

python train_s2s_hmamba.py $HPARAMS \
    --data_folder $DATA_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    $EVAL_ARGS

LM_EXIT=$?
echo "With-LM eval finished — exit code: $LM_EXIT — $(date)"

# --- Phase 3: No-LM eval (always runs) ---
echo ""
echo "============================================================"
echo "Starting no-LM eval (single GPU) — $(date)"
echo "============================================================"

python train_s2s_hmamba.py $HPARAMS \
    --data_folder $DATA_FOLDER \
    --output_folder $OUTPUT_FOLDER \
    $EVAL_ARGS --no_lm True

NOLM_EXIT=$?
echo "No-LM eval finished — exit code: $NOLM_EXIT — $(date)"

echo ""
echo "============================================================"
echo "All done — train=$TRAIN_EXIT, with-LM=$LM_EXIT, no-LM=$NOLM_EXIT — $(date)"
echo "============================================================"
