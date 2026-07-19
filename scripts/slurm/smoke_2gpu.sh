#!/bin/bash
# =============================================================================
# DC-ASR 2-GPU smoke — proves the config-only multi-GPU contract after the
# Phase-0 DDP fixes (device-type, unsharded dev, rank logs, rank-0 provenance).
# Tiny real-audio manifests; torchrun x2 + resume; asserts then SMOKE2GPU_OK.
# =============================================================================

#SBATCH --job-name=dcasr_smoke2gpu
#SBATCH --output=/data/user_data/anshulk/hnet-asr/logs/slurm-smoke2gpu-%j.out
#SBATCH --error=/data/user_data/anshulk/hnet-asr/logs/slurm-smoke2gpu-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -uo pipefail

echo "[$(date)] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?}"
source /home/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate hnet-asr
cd /home/anshulk/H-Net_Mamba_ASR

ENVBIN=/data/user_data/anshulk/envs/hnet-asr/bin
SMOKE=/data/user_data/anshulk/hnet-asr/smoke_phase0
export DCASR_LOG_DIR=$SMOKE/logs DCASR_METRICS_DIR=$SMOKE/experiments PYTHONDONTWRITEBYTECODE=1
echo "[$(date)] PYTHON=$ENVBIN/python"

RUN=smoke_p0_2gpu
OV="experiment.name=$RUN data.manifests_dir=$SMOKE/manifests batch_bins=6000 num_workers=2 \
train.log_interval=1 eval.valid_interval_epoch=1 scheduler_conf.warmup_steps=20 \
keep_nbest_models=1 keep_all_checkpoints=true"
rm -rf checkpoints/$RUN $SMOKE/experiments/$RUN $SMOKE/logs/train_${RUN}*.log

echo "[$(date)] == RUN 1: fresh, world=2, 2 epochs =="
$ENVBIN/torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/typeA_small_N1_ctc.yaml $OV train.max_epoch=2
E1=$?
echo "[$(date)] RUN1_EXIT=$E1"

echo "[$(date)] == RUN 2: --resume auto, world=2, 1 more epoch =="
$ENVBIN/torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/typeA_small_N1_ctc.yaml --resume auto $OV train.max_epoch=3
E2=$?
echo "[$(date)] RUN2_EXIT=$E2"

echo "[$(date)] == VERIFY =="
$ENVBIN/python - << 'EOF'
import json, pathlib, sys
S = pathlib.Path("/data/user_data/anshulk/hnet-asr/smoke_phase0")
CK = pathlib.Path("/home/anshulk/H-Net_Mamba_ASR/checkpoints/smoke_p0_2gpu")
run = S / "experiments" / "smoke_p0_2gpu"
ok = True
def check(name, cond):
    global ok
    print(("PASS" if cond else "FAIL"), name)
    ok &= bool(cond)

check("F3 rank-1 log file exists", (S / "logs" / "train_smoke_p0_2gpu.rank1.log").exists())
check("F3 rank-0 log file exists", (S / "logs" / "train_smoke_p0_2gpu.log").exists())
recs = [json.loads(l) for l in open(run / "metrics.jsonl") if l.strip()]
keys = {r["key"] for r in recs}
check("F1 sys/gpu_mem_gb logged under cuda:N", "sys/gpu_mem_gb" in keys)
check("F8 monitors train/loss + valid/wer in metrics", {"train/loss", "valid/wer"} <= keys)
check("dev metrics per split present", {"dev_dev-clean/wer", "dev_dev-other/wer"} <= keys)
summ = json.loads((run / "summary.json").read_text())
prov = summ.get("provenance", [])
check("provenance appended across resume (2 entries)", len(prov) == 2)
check("provenance world_size == 2",
      all(p.get("process", {}).get("env_vars", {}).get("WORLD_SIZE") == "2"
          and p.get("batch", {}).get("world_size") == 2 for p in prov)
      and summ.get("world_size") == 2)
egb = prov[0].get("batch", {}).get("effective_global_batch_frames")
check("effective global batch == 6000*1*2", egb == 12000)
eps = {p.name for p in CK.glob("epoch*.pt")}
check("F5 all 3 epoch ckpts retained (keep_all, keep_nbest=1)",
      {"epoch0000.pt", "epoch0001.pt", "epoch0002.pt"} <= eps)
link = CK / "valid.wer.best.pt"
check("F11 best symlink resolves", link.is_symlink() and link.resolve().exists())
sys.exit(0 if ok else 1)
EOF
EV=$?
echo "[$(date)] VERIFY_EXIT=$EV"
if [ "$E1" -eq 0 ] && [ "$E2" -eq 0 ] && [ "$EV" -eq 0 ]; then
  echo "SMOKE2GPU_OK"
else
  echo "SMOKE2GPU_FAIL"
fi
