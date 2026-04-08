#!/bin/bash
#SBATCH --job-name=eval_S_all
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/eval_hmamba_S_all-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/eval_hmamba_S_all-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=anshulk@andrew.cmu.edu

echo "============================================================"
echo "H-Mamba Small — eval ALL (N2, N3, N4) — 3x A6000, preempt"
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

# ---------- GPU 0: S_N2 ----------
(
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU 0] S_N2 no-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N2 \
        $COMMON_ARGS --no_lm True
    NOLM_EXIT=$?
    echo "[GPU 0] S_N2 no-LM eval done — exit code: $NOLM_EXIT — $(date)"

    echo "[GPU 0] S_N2 with-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N2 \
        $COMMON_ARGS
    LM_EXIT=$?
    echo "[GPU 0] S_N2 with-LM eval done — exit code: $LM_EXIT — $(date)"
) &
PID_N2=$!

# ---------- GPU 1: S_N3 ----------
(
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU 1] S_N3 no-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N3.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N3 \
        $COMMON_ARGS --no_lm True
    NOLM_EXIT=$?
    echo "[GPU 1] S_N3 no-LM eval done — exit code: $NOLM_EXIT — $(date)"

    echo "[GPU 1] S_N3 with-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N3.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N3 \
        $COMMON_ARGS
    LM_EXIT=$?
    echo "[GPU 1] S_N3 with-LM eval done — exit code: $LM_EXIT — $(date)"
) &
PID_N3=$!

# ---------- GPU 2: S_N4 ----------
(
    export CUDA_VISIBLE_DEVICES=2
    echo "[GPU 2] S_N4 no-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N4.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N4 \
        $COMMON_ARGS --no_lm True
    NOLM_EXIT=$?
    echo "[GPU 2] S_N4 no-LM eval done — exit code: $NOLM_EXIT — $(date)"

    echo "[GPU 2] S_N4 with-LM eval starting — $(date)"
    python train_s2s_hmamba.py hparams/S2S/hmamba_small_N4.yaml \
        --data_folder $DATA_FOLDER \
        --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N4 \
        $COMMON_ARGS
    LM_EXIT=$?
    echo "[GPU 2] S_N4 with-LM eval done — exit code: $LM_EXIT — $(date)"
) &
PID_N4=$!

echo "Waiting for all evals: N2=$PID_N2, N3=$PID_N3, N4=$PID_N4"
wait $PID_N2 $PID_N3 $PID_N4

echo ""
echo "============================================================"
echo "All evals finished — $(date)"
echo "============================================================"
