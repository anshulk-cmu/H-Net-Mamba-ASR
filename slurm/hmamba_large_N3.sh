#!/bin/bash
#SBATCH --job-name=hmamba_L_N3
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/hmamba_L_N3-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/hmamba_L_N3-%j.err
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "H-Mamba Large N3 — 2x A6000, preempt"
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

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Master port: $MASTER_PORT"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_s2s_hmamba.py hparams/S2S/hmamba_large_N3.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_large_N3 \
    --batch_size 16 \
    --max_batch_length_train 600 \
    --max_batch_length_val 100 \
    --use_wandb True \
    --precision bf16

TRAIN_EXIT=$?

echo ""
echo "============================================================"
echo "Training + with-LM eval finished — exit code: $TRAIN_EXIT — $(date)"
echo "============================================================"

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Starting no-LM eval (single GPU) — $(date)"
    echo "============================================================"

    python train_s2s_hmamba.py hparams/S2S/hmamba_large_N3.yaml \
        --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_large_N3 \
        --batch_size 16 \
        --max_batch_length_train 600 \
        --max_batch_length_val 100 \
        --precision bf16 \
        --skip_train True \
        --no_lm True

    NOLM_EXIT=$?
    echo ""
    echo "============================================================"
    echo "No-LM eval finished — exit code: $NOLM_EXIT — $(date)"
    echo "============================================================"
fi

exit $TRAIN_EXIT
