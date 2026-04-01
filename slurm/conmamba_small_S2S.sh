#!/bin/bash
#SBATCH --job-name=conmamba_S_S2S
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/conmamba_S_S2S-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/conmamba_S_S2S-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "conmamba_small S2S — 2x A6000, preempt"
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

stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1

eval "$(conda shell.bash hook)"
conda activate hnetasr || { echo "ERROR: failed to activate hnetasr"; exit 1; }

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

# Pick a free port dynamically to avoid conflicts when jobs share a node
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Master port: $MASTER_PORT"

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_S2S.py hparams/S2S/conmamba_small.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder /data/user_data/anshulk/hnet_asr/results/conmamba_small_S2S \
    --batch_size 24 \
    --max_batch_length_train 1200 \
    --max_batch_length_val 120 \
    --use_wandb True \
    --precision bf16

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job finished — exit code: $EXIT_CODE — $(date)"
echo "============================================================"
exit $EXIT_CODE
