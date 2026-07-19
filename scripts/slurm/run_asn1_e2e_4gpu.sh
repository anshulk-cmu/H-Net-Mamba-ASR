#!/bin/bash
# =============================================================================
# DC-ASR A·S·N1 END-TO-END pipeline — ONE submission runs the whole experiment
# from configs/typeA_small_N1.yaml: train (4-GPU DDP, 200 epochs, early stop)
# → decode (7 cells × 4 splits ± LM, one split per GPU in parallel) → score
# (WER/CER/RTF, bootstrap significance, ctc_greedy gate < 12) → efficiency →
# interp (boundaries/probes/robustness/emergence, uncapped).
#
# Requeue-safe end to end: training resumes via --resume auto; every finished
# stage skips via marker files under experiments/typeA_small_N1/pipeline/, so
# preemption or the 48 h limit just chains — the pipeline always completes.
# Note: parallel per-split decodes each rewrite decode/summary.json (last one
# wins); score/scores.json is the canonical combined record of every cell.
# =============================================================================

#SBATCH --job-name=dcasr_asn1_e2e
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-asn1-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-asn1-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -uo pipefail
echo "[$(date)] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnet-asr
cd /home/anshulk/H-Net_Mamba_ASR
ENVBIN=/data/user_data/anshulk/envs/hnet-asr/bin
export PYTHONDONTWRITEBYTECODE=1

CFG=configs/typeA_small_N1.yaml
RUN=typeA_small_N1
CKPT=checkpoints/$RUN/valid.wer.ave.pt
MARK=experiments/$RUN/pipeline
mkdir -p "$MARK"

requeue() {
  echo "[$(date)] time-limit signal — requeueing ${SLURM_JOB_ID} (pipeline resumes at the incomplete stage)"
  for _try in 1 2 3; do
    scontrol requeue "${SLURM_JOB_ID}" && return
    echo "[$(date)] scontrol requeue failed (attempt $_try), retrying"
    sleep 20
  done
  echo "[$(date)] WARNING: requeue failed 3x — chain broken, resubmit manually"
}
trap requeue USR1

plog() { echo "[$(date)] $*" | tee -a "$MARK/pipeline.log"; }
stage_done() { [ -f "$MARK/$1.done" ]; }
mark_done()  { touch "$MARK/$1.done"; plog "STAGE $1 DONE"; }
plog "attempt start: job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0} host=$(hostname)"

# ── 1) TRAIN: 4-GPU DDP; exit 0 == finished (max_epoch or early stop) ────────
if ! stage_done train; then
  plog "STAGE train: torchrun x4"
  $ENVBIN/torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --config $CFG --resume auto &
  wait $!
  E=$?
  if [ "$E" -ne 0 ]; then plog "TRAIN_EXIT=$E"; exit "$E"; fi
  mark_done train
fi

# ── 2) DECODE: 7 cells per split, one split per GPU, in parallel ─────────────
PIDS=() NAMES=()
i=0
for SPLIT in dev-clean dev-other test-clean test-other; do
  if ! stage_done "decode_$SPLIT"; then
    plog "STAGE decode_$SPLIT on GPU $i"
    {
      CUDA_VISIBLE_DEVICES=$i $ENVBIN/python scripts/decode.py --config $CFG \
        --checkpoint "$CKPT" "decode.splits=[$SPLIT]" \
        >> "$MARK/decode_$SPLIT.log" 2>&1 && touch "$MARK/decode_$SPLIT.done"
    } &
    PIDS+=($!) NAMES+=("$SPLIT")
  fi
  i=$((i + 1))
done
FAIL=0
for j in "${!PIDS[@]}"; do
  wait "${PIDS[$j]}" || { plog "DECODE FAIL: ${NAMES[$j]} (see $MARK/decode_${NAMES[$j]}.log)"; FAIL=1; }
done
[ "$FAIL" -ne 0 ] && exit 1

# stages below run backgrounded + wait so the USR1 requeue trap stays live
# ── 3) SCORE: WER/CER/RTF + significance + the ctc_greedy test-clean gate ────
if ! stage_done score; then
  plog "STAGE score"
  $ENVBIN/python scripts/score_wer.py --config $CFG --checkpoint "$CKPT" &
  wait $! || exit 1
  mark_done score
fi

# ── 4) EFFICIENCY: params + analytic GFLOPs into the run dir ─────────────────
if ! stage_done efficiency; then
  plog "STAGE efficiency"
  $ENVBIN/python scripts/efficiency.py --config $CFG &
  wait $! || exit 1
  mark_done efficiency
fi

# ── 5) INTERP: full uncapped suite + emergence over every retained epoch ─────
if ! stage_done interp; then
  plog "STAGE interp (boundaries,probes,robustness,emergence)"
  CUDA_VISIBLE_DEVICES=0 $ENVBIN/python scripts/run_interp.py --config $CFG \
    --checkpoint "$CKPT" --modes boundaries,probes,robustness,emergence &
  wait $! || exit 1
  mark_done interp
fi

plog "PIPELINE_COMPLETE"
$ENVBIN/python - << 'EOF'
import json
g = json.load(open("experiments/typeA_small_N1/decode/valid.wer.ave/score/scores.json"))["gate"]
print(f"GATE: {'PASS' if g.get('passed') else 'FAIL'} — {g}")
EOF
