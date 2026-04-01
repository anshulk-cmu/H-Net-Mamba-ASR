#!/bin/bash
#SBATCH --job-name=conformer_L_CTC
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/conformer_L_CTC-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/conformer_L_CTC-%j.err
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "conformer_large CTC — 4x A6000, general, fp32 restart"
echo "============================================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Start    : $(date)"
echo "============================================================"

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB__SERVICE_WAIT=300

stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1

source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnetasr || { echo "ERROR: failed to activate hnetasr"; exit 1; }

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Master port: $MASTER_PORT"

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT train_CTC.py hparams/CTC/conformer_large.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder /data/user_data/anshulk/hnet_asr/results/conformer_large_CTC \
    --batch_size 32 \
    --max_batch_length_train 600 \
    --max_batch_len_val 60 \
    --use_wandb True \
    --precision fp32

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "Job finished — exit code: $EXIT_CODE — $(date)"
echo "============================================================"
exit $EXIT_CODE
