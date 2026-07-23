#!/bin/bash
# =============================================================================
# Dev sweep of the SECOND-PASS rescoring weights for aed_beam_lm / joint_beam_lm.
#
# The +LM cells on the AED and joint read-outs now integrate the external LM by
# rescoring the completed n-best, not by first-pass shallow fusion:
#     S(h) = (1-ctc_w)*AED(h) + ctc_w*CTC(h) + lambda*logP_LM(h) + gamma*len(h)
# The acoustic beam is LM-free, so the n-best and every component score are
# INDEPENDENT of (lambda, gamma). Each utterance is therefore decoded ONCE and the
# whole grid is evaluated by pure re-ranking: one decode pass buys the entire sweep,
# and every grid point is compared on an identical hypothesis set.
#
# gamma (ESPnet's `penalty`) is swept because logP_LM(h) is a sum of negative
# per-token terms, so lambda*logP_LM systematically favours SHORTER hypotheses.
# The lambda=0 row is the built-in CONTROL: length correction with no LM consulted,
# so any reported gain must beat it to be attributable to the LM.
#
# FULL dev splits (n>=2703) x both models x both splits — no subsampling, so there
# is no sampling seed to get lucky on. Nothing here touches a test split.
# =============================================================================

#SBATCH --job-name=dcasr_rescore_sweep
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-rescore-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-rescore-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -uo pipefail
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-?}"
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnet-asr
cd /home/anshulk/H-Net_Mamba_ASR
ENVBIN=/data/user_data/anshulk/envs/hnet-asr/bin
export PYTHONDONTWRITEBYTECODE=1        # no __pycache__ left behind

run() {   # <gpu> <run_name> <split> <tag>
  CUDA_VISIBLE_DEVICES=$1 $ENVBIN/python -u scripts/analysis/sweep_rescore_weight.py \
    --config configs/$2.yaml --checkpoint checkpoints/$2/valid.wer.ave.pt \
    --split "$3" --n 0 --read-outs aed,joint \
    > "logs/rescore_sweep_$2_$4.txt" 2>&1
}

run 0 typeA_small_N1 dev-clean dc &
run 1 typeA_small_N1 dev-other do &
run 2 typeA_small_N2 dev-clean dc &
run 3 typeA_small_N2 dev-other do &
wait

echo "[$(date)] RESCORE SWEEP COMPLETE"
for f in logs/rescore_sweep_typeA_small_N*_*.txt; do
  echo "── $f"
  grep -E "^(=====|no-LM|CONTROL|best |  LM-attributable|n-best ORACLE|  WARNING)" "$f"
done
