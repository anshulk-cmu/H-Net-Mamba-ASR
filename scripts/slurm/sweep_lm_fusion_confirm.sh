#!/bin/bash
# =============================================================================
# Confirmation run for the LM-fusion weights: does the round-2 optimum survive
# a DIFFERENT random sample and a DIFFERENT split?
#
# Round 2 (250 random dev-clean utts, seed 1234) left two candidates:
#   lambda=0.05 alpha=0.034  - flat plateau, alpha insensitive, gain ~0
#   lambda=0.15 alpha=0.134  - best WER, but knife-edge (lambda 0.2 -> 4.6/5.5)
# The N1 gap between them is about one word in 5,000, so it is inside noise.
# Choosing the better number from a single sample is how lm_length_bonus=1.0
# shipped. This re-measures both on seed 999 and on dev-other, where the model
# is less confident and therefore has more room for a language prior.
#
# The 2x2 grid also covers the off-diagonal points, mapping the ridge rather
# than just testing two isolated settings.
# =============================================================================

#SBATCH --job-name=dcasr_lm_confirm
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-confirm-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-confirm-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -uo pipefail
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnet-asr
cd /home/anshulk/H-Net_Mamba_ASR
ENVBIN=/data/user_data/anshulk/envs/hnet-asr/bin
export PYTHONDONTWRITEBYTECODE=1

N=${SWEEP_N:-250}
SEED=999                     # NOT 1234: an independent draw from round 2
run() {   # <gpu> <run_name> <split> <tag>
  CUDA_VISIBLE_DEVICES=$1 $ENVBIN/python scripts/analysis/sweep_lm_fusion.py \
    --config configs/$2.yaml --checkpoint checkpoints/$2/valid.wer.ave.pt \
    --split "$3" --n "$N" --seed "$SEED" --tag "$4" \
    --lm-weights 0.05,0.15 --ilm-weights 0.034,0.134 \
    > "logs/confirm_$2_$4.txt" 2>&1
}

run 0 typeA_small_N1 dev-clean dc &
run 1 typeA_small_N1 dev-other do &
run 2 typeA_small_N2 dev-clean dc &
run 3 typeA_small_N2 dev-other do &
wait

echo "[$(date)] CONFIRM COMPLETE"
for f in logs/confirm_typeA_small_N*_*.txt; do echo "── $f"; grep -E "^(no-LM|lam=|best)" "$f"; done
