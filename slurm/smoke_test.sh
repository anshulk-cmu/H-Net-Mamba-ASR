#!/bin/bash
#SBATCH --job-name=hnetasr_smoke
#SBATCH --output=/home/anshulk/h-mamba_asr/logs/smoke-%j.out
#SBATCH --error=/home/anshulk/h-mamba_asr/logs/smoke-%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# ============================================================================
# Smoke Test — ConMamba (Small, S2S), 2 epochs, 1x L40S, debug partition
#
# Purpose: Verify end-to-end pipeline works on a Babel compute node:
#   - AutoFS mounts are accessible
#   - conda env + mamba/causal-conv1d CUDA kernels load correctly
#   - LibriSpeech data prep (CSV generation + tokenizer download) completes
#   - Model forward + backward pass runs without OOM or CUDA errors
#   - Checkpoint saving works to user_data
#
# After this passes, all 8 full training jobs can be safely submitted.
# ============================================================================

echo "============================================================"
echo "Smoke Test — ConMamba ASR (hnetasr)"
echo "============================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "============================================================"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Set CUDA 11.8 explicitly — required for mamba-ssm and causal-conv1d kernels
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# WandB online — compute nodes have internet access
export WANDB__SERVICE_WAIT=300

# Match num threads to allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ============================================================================
# AUTOMOUNT TRIGGER
# AutoFS mounts /data/user_data on-demand. Must stat the full path first
# or the directory will appear empty and all data paths will fail.
# ============================================================================
echo ""
echo "Triggering AutoFS mounts..."
stat /data/user_data/anshulk > /dev/null 2>&1
stat /data/user_data/anshulk/hnet_asr > /dev/null 2>&1
stat /data/user_data/anshulk/hnet_asr/LibriSpeech > /dev/null 2>&1
echo "  Mounts triggered."

# ============================================================================
# ACTIVATE CONDA ENVIRONMENT
# ============================================================================
echo ""
echo "Activating conda environment: hnetasr..."
eval "$(conda shell.bash hook)"
conda activate hnetasr || { echo "ERROR: Failed to activate hnetasr env"; exit 1; }
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
echo ""
echo "Running pre-flight checks..."

# 1. GPU check
echo "  [1/5] Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "  ERROR: nvidia-smi not available"
    exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "  GPU OK"

# 2. CUDA kernel imports (mamba-ssm + causal-conv1d compile on first use)
echo "  [2/5] Checking mamba-ssm and causal-conv1d CUDA kernels..."
python -c "
from mamba_ssm import Mamba
import causal_conv1d
import torch
print(f'    torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'    mamba_ssm: OK')
print(f'    causal_conv1d: OK')
" || { echo "  ERROR: mamba/causal_conv1d import failed"; exit 1; }
echo "  CUDA kernels OK"

# 3. SpeechBrain import
echo "  [3/5] Checking SpeechBrain..."
python -c "import speechbrain as sb; print(f'    speechbrain: {sb.__version__}')" \
    || { echo "  ERROR: speechbrain import failed"; exit 1; }
echo "  SpeechBrain OK"

# 4. LibriSpeech data
echo "  [4/5] Checking LibriSpeech data..."
for split in train-clean-100 dev-clean test-clean; do
    if [ ! -d "/data/user_data/anshulk/hnet_asr/LibriSpeech/$split" ]; then
        echo "  ERROR: Missing split: $split"
        exit 1
    fi
done
echo "  LibriSpeech splits OK"

# 5. Results output directory
echo "  [5/5] Checking results directory..."
mkdir -p /data/user_data/anshulk/hnet_asr/results/smoke_test
echo "  Output dir OK: /data/user_data/anshulk/hnet_asr/results/smoke_test"

echo ""
echo "All pre-flight checks passed!"
echo "============================================================"

# ============================================================================
# RUN SMOKE TEST
# Using: conmamba_small S2S (smallest + fastest model)
# Overrides:
#   - number_of_epochs 2          (just enough to verify train+valid loop)
#   - output_folder               (redirect checkpoints to user_data, not /home)
#   - skip_prep False             (run full data prep on first pass)
# ============================================================================

cd /home/anshulk/h-mamba_asr/Mamba-ASR

echo ""
echo "Launching smoke test: ConMamba Small S2S, 2 epochs..."
echo ""

python train_S2S.py hparams/S2S/conmamba_small.yaml \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder /data/user_data/anshulk/hnet_asr/results/smoke_test/conmamba_small_S2S \
    --number_of_epochs 2 \
    --use_wandb True \
    --precision bf16

EXIT_CODE=$?

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================"
echo "Smoke test finished"
echo "Exit code  : $EXIT_CODE"
echo "End time   : $(date)"
echo "Runtime    : $SECONDS seconds"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Smoke test passed!"
    echo ""
    echo "Verified:"
    echo "  - AutoFS mounts accessible"
    echo "  - mamba-ssm / causal-conv1d CUDA kernels loaded"
    echo "  - Data prep (CSV manifests + tokenizer) completed"
    echo "  - 2 full train+valid epochs ran without error"
    echo "  - Checkpoints saved to user_data"
    echo ""
    echo "You are ready to submit all 8 full training jobs."
    echo ""
    echo "Results saved to:"
    ls -lh /data/user_data/anshulk/hnet_asr/results/smoke_test/ 2>/dev/null
else
    echo ""
    echo "FAILURE: Smoke test failed (exit code $EXIT_CODE)"
    echo ""
    echo "Check logs for details:"
    echo "  $SLURM_SUBMIT_DIR/../logs/smoke-${SLURM_JOB_ID}.err"
fi

echo "============================================================"
exit $EXIT_CODE
