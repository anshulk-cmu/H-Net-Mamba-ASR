#!/bin/bash
#SBATCH --job-name=eval_S_N1
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/eval_hmamba_S_N1-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/eval_hmamba_S_N1-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "H-Mamba Small N=1 — no-LM + with-LM eval — 1x A6000, general"
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
export WANDB_MODE=offline

stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1

source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnetasr || { echo "ERROR: failed to activate hnetasr"; exit 1; }

pip install numpy==1.26.4 --quiet 2>/dev/null

echo "Python : $(which python)"
echo "GPUs   : $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
echo ""

cd /home/anshulk/h-mamba_asr/Mamba-ASR

DATA_FOLDER=/data/user_data/anshulk/hnet_asr/LibriSpeech
COMMON_ARGS="--batch_size 24 --max_batch_length_train 1200 --max_batch_length_val 120 --precision bf16 --skip_train True"

echo "[S_N1] no-LM eval starting — $(date)"
python train_s2s_hmamba.py hparams/S2S/hmamba_small_N1.yaml \
    --data_folder $DATA_FOLDER \
    --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N1 \
    $COMMON_ARGS --no_lm True
NOLM_EXIT=$?
echo "[S_N1] no-LM eval done — exit code: $NOLM_EXIT — $(date)"

echo "[S_N1] with-LM eval starting — $(date)"
python train_s2s_hmamba.py hparams/S2S/hmamba_small_N1.yaml \
    --data_folder $DATA_FOLDER \
    --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N1 \
    $COMMON_ARGS
LM_EXIT=$?
echo "[S_N1] with-LM eval done — exit code: $LM_EXIT — $(date)"

echo ""
echo "============================================================"
echo "S_N1 eval finished — $(date)"
echo "============================================================"
