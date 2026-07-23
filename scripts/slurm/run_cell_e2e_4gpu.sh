#!/bin/bash
# =============================================================================
# DC-ASR grid-cell END-TO-END pipeline (parameterized): args = <config.yaml> <run_name>.
# Same five stages + requeue chain as the A·S·N1 launcher, for any grid cell.
# → decode (7 cells × 4 splits ± LM, one split per GPU in parallel) → score
# (WER/CER/RTF, bootstrap significance, ctc_greedy gate < 12) → efficiency →
# interp (boundaries/probes/robustness/emergence, uncapped).
#
# Requeue-safe end to end: training resumes via --resume auto; every finished
# stage skips via marker files under experiments/<run_name>/pipeline/, so
# preemption or the 48 h limit just chains — the pipeline always completes.
# Note: parallel per-split decodes each rewrite decode/summary.json (last one
# wins); score/scores.json is the canonical combined record of every cell.
# =============================================================================

#SBATCH --job-name=dcasr_cell_e2e
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-cell-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-cell-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=general
# GPU TYPE IS PINNED, and it is load-bearing for the efficiency stage: the
# 2026-07-20 audit found N1 landed on L40S and N2 on A6000 (~1.34x slower), so
# the apparent "N=2 is 30% slower" was mostly hardware — same-hardware profiling
# showed the N=2 encoder is actually 0.8% FASTER. RTF/efficiency numbers are only
# comparable across grid cells if every cell decodes on the SAME GPU type. L40S
# also subsumes the old --exclude=babel-w9-32 (an A6000 node). 49 L40S nodes x 8.
#SBATCH --gres=gpu:L40S:4
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
# NCCL over shared memory: job 9370311 deadlocked in DDP's init broadcast with
# all GPUs spinning at 100%/~1GB (P2P transport wedge on PCIe L40S). SHM costs
# a few % on a 79M model's allreduces and is robust across node topologies.
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

CFG=${1:?usage: sbatch [-J name] run_cell_e2e_4gpu.sh <config.yaml> <run_name>}
RUN=${2:?usage: sbatch [-J name] run_cell_e2e_4gpu.sh <config.yaml> <run_name>}
CKPT=checkpoints/$RUN/valid.wer.ave.pt
MARK=experiments/$RUN/pipeline
mkdir -p "$MARK"

requeue() {
  plog "time-limit signal — draining then requeueing ${SLURM_JOB_ID}"
  # MUST kill the training processes BEFORE requeueing. A requeued job starts a
  # SECOND writer while the old one is still alive: on 2026-07-21 a preemption
  # requeue left an orphan that co-wrote metrics.jsonl for ~18 min (231 duplicate
  # steps, 6 backward jumps) and came within ~5 min of clobbering latest.pt with
  # stale state. Checkpoints are written at epoch boundaries, so a clean SIGTERM
  # here loses at most the in-flight epoch, which --resume auto redoes anyway.
  if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM -- "-$(ps -o pgid= -p "$TRAIN_PID" | tr -d ' ')" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null
    for _i in $(seq 1 20); do
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      sleep 3
    done
    kill -KILL -- "-$(ps -o pgid= -p "$TRAIN_PID" | tr -d ' ')" 2>/dev/null || kill -KILL "$TRAIN_PID" 2>/dev/null
  fi
  pkill -KILL -f "train.py --config $CFG" 2>/dev/null   # backstop for orphaned workers
  sleep 2
  plog "training processes drained; requeueing"
  for _try in 1 2 3; do
    scontrol requeue "${SLURM_JOB_ID}" && return
    plog "scontrol requeue failed (attempt $_try), retrying"
    sleep 20
  done
  plog "WARNING: requeue failed 3x — chain broken, resubmit manually"
}
trap requeue USR1

plog() { echo "[$(date)] $*" | tee -a "$MARK/pipeline.log"; }
stage_done() { [ -f "$MARK/$1.done" ]; }
mark_done()  { touch "$MARK/$1.done"; plog "STAGE $1 DONE"; }
plog "attempt start: job=${SLURM_JOB_ID:-?} restarts=${SLURM_RESTART_COUNT:-0} host=$(hostname)"

# ── preflight: torch must actually initialize CUDA (nvidia-smi alone can lie —
# job 9370297 died on babel-w9-32 with 'CUDA unknown error' on a node whose
# nvidia-smi looked healthy). Bad node => requeue (bounded by MaxBatchRequeue=5).
if ! $ENVBIN/python -c "import sys, torch; n = torch.cuda.device_count(); print(f'preflight: torch sees {n} GPUs'); sys.exit(0 if n >= 4 else 1)"; then
  plog "PREFLIGHT FAILED on $(hostname): torch cannot initialize CUDA — requeueing to another node"
  if [ "${SLURM_RESTART_COUNT:-0}" -lt 4 ]; then
    scontrol requeue "${SLURM_JOB_ID}" && exit 0
  fi
  plog "PREFLIGHT FAILED ${SLURM_RESTART_COUNT}x — giving up"
  exit 1
fi

# ── 1) TRAIN: 4-GPU DDP; exit 0 == finished (max_epoch or early stop) ────────
if ! stage_done train; then
  plog "STAGE train: torchrun x4"
  $ENVBIN/torchrun --standalone --nproc_per_node=4 scripts/train.py \
    --config $CFG --resume auto &
  TRAIN_PID=$!            # the trap kills this group before requeueing
  wait $TRAIN_PID
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
$ENVBIN/python - << EOF
import json
g = json.load(open("experiments/$RUN/decode/valid.wer.ave/score/scores.json"))["gate"]
print(f"GATE: {'PASS' if g.get('passed') else 'FAIL'} — {g}")
EOF
