"""Dev-only sweep of the LM-fusion weights for the label-synchronous +LM beam cells.

Shallow fusion on an AED decoder double-counts the language prior, because the decoder
already carries its own internal LM (ILM). That surplus per-token cost truncates
hypotheses. Two knobs can offset it, and this script measures both on real audio:

  lm_weight    (lambda) external-LM weight
  ilm_weight   (alpha)  density-ratio subtraction of the AED's own internal LM
  length_bonus (gamma)  flat insertion bonus (the blunt alternative to alpha)

Measured LM entropies (dev-clean, 110,132 tokens): H_ext = 2.38 nats, H_ILM = 3.55 nats.
So the per-token drift of the fused score is  alpha*H_ILM - lambda*H_ext, which vanishes on
the ray  alpha = lambda * H_ext/H_ILM ~ 0.67*lambda. That ray is the sensible place to search.

Runs on a RANDOM subset of a dev split. Sampling matters: tuning on the LONGEST utterances
overstates the truncation problem and picks a bonus that over-generates on typical audio
(this is exactly how lm_length_bonus=1.0 shipped and caused 15% insertions). Never select
these weights on a length-biased sample, and never on a test split.

Usage:
  python scripts/analysis/sweep_lm_fusion.py --config configs/typeA_small_N1.yaml \
      --checkpoint checkpoints/typeA_small_N1/valid.wer.ave.pt --n 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dcasr.data.librispeech import LibriSpeechDataset, make_dataloader   # noqa: E402
from dcasr.data.tokenizer import Tokenizer                               # noqa: E402
from dcasr.eval.metrics import word_error_rate                           # noqa: E402
from dcasr.logging_utils import get_logger, setup_logging                # noqa: E402
from dcasr.tasks.asr_task import build_model                             # noqa: E402
from dcasr.tasks.build import build_cmvn, build_frontend                 # noqa: E402
from dcasr.tasks.decode_task import (decode_batch, load_lm_scorer,       # noqa: E402
                                     load_model_weights)

REPO = Path(__file__).resolve().parents[2]
logger = get_logger(__name__)


def sample_manifest(src: Path, dst: Path, n: int, seed: int) -> int:
    """Write a uniformly random n-line subset of `src` (whole-corpus, NOT length-sorted)."""
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    rng = random.Random(seed)
    keep = lines if n >= len(lines) else rng.sample(lines, n)
    dst.write_text("\n".join(keep) + "\n")
    return len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--cell", default="aed_beam_lm",
                    choices=["aed_beam_lm", "joint_beam_lm"])
    ap.add_argument("--n", type=int, default=200, help="random utterances to decode")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lm-weights", default="0.3,0.5")
    ap.add_argument("--ilm-weights", default="0.0,0.1,0.2,0.3,0.4")
    ap.add_argument("--length-bonus", default="0.0",
                    help="comma list; the blunt alternative to ilm_weight")
    ap.add_argument("--tag", default="", help="suffix for output paths; keeps parallel "
                    "workers on the same run from racing on the same files")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    run_name = str(cfg.experiment.name)
    setup_logging(f"sweep_lm_fusion_{run_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer(REPO / cfg.bpemodel)
    frontend = build_frontend(cfg)
    cmvn = build_cmvn(cfg, REPO)
    model = build_model(cfg, tokenizer.vocab_size).to(device).eval()
    load_model_weights(model, REPO / args.checkpoint)
    dc = OmegaConf.to_container(cfg.decode, resolve=True)
    lm = load_lm_scorer(dc, REPO, tokenizer, device)

    src = REPO / str(cfg.data.manifests_dir) / f"{args.split}.jsonl"
    tag = f".{args.tag}" if args.tag else ""
    sub = (REPO / "experiments" / run_name / "sweep" /
           f"{args.split}.sample{args.n}{tag}.jsonl")
    sub.parent.mkdir(parents=True, exist_ok=True)
    kept = sample_manifest(src, sub, args.n, args.seed)
    logger.info("sweep on %d random %s utts (seed %d)", kept, args.split, args.seed)

    ds = LibriSpeechDataset(sub, frontend, tokenizer, cmvn=cmvn, specaugment=None,
                            augment=False)
    loader, _ = make_dataloader(ds, int(cfg.batch_bins), augment=False, num_workers=2,
                                world_size=1, rank=0)
    batches = list(loader)

    read_out = "aed" if args.cell == "aed_beam_lm" else "joint"
    grid = [(lw, iw, lb)
            for lw in [float(x) for x in args.lm_weights.split(",")]
            for iw in [float(x) for x in args.ilm_weights.split(",")]
            for lb in [float(x) for x in args.length_bonus.split(",")]]

    rows = []
    # reference row: the same beam with NO external LM at all
    for name, cell, over in ([("no-LM", {"name": args.cell.replace("_lm", ""),
                                         "read_out": read_out, "search": "beam", "lm": False},
                               {})]
                             + [(f"lam={lw} alpha={iw} gamma={lb}",
                                 {"name": args.cell, "read_out": read_out,
                                  "search": "beam", "lm": True},
                                 {"lm_weight": lw, "ilm_weight": iw, "lm_length_bonus": lb})
                                for lw, iw, lb in grid]):
        d = dict(dc)
        d.update(over)
        refs, hyps, t0 = [], [], time.perf_counter()
        for b in batches:
            for r in decode_batch(model, tokenizer, b, cell, d, device, lm=lm):
                refs.append(r["ref"])
                hyps.append(r["hyp"])
        st = word_error_rate(refs, hyps)
        nref = sum(len(r.split()) for r in refs)
        nhyp = sum(len(h.split()) for h in hyps)
        row = {"setting": name, **over, "wer": 100 * st.er,
               "sub": 100 * st.sub_rate, "del": 100 * st.del_rate, "ins": 100 * st.ins_rate,
               "len_ratio": nhyp / max(1, nref), "empty": sum(1 for h in hyps if not h.strip()),
               "secs": round(time.perf_counter() - t0, 1)}
        rows.append(row)
        logger.info("%-34s WER %6.2f  (S %5.2f D %5.2f I %5.2f)  len %.3f  empty %d  [%.0fs]",
                    name, row["wer"], row["sub"], row["del"], row["ins"],
                    row["len_ratio"], row["empty"], row["secs"])
        print(f"{name:<34} WER {row['wer']:6.2f}  S {row['sub']:5.2f} D {row['del']:5.2f} "
              f"I {row['ins']:5.2f}  len {row['len_ratio']:.3f}  empty {row['empty']}",
              flush=True)

    out = (Path(args.out) if args.out
           else sub.parent / f"{args.cell}.{args.split}{tag}.sweep.json")
    out.write_text(json.dumps(
        {"run": run_name, "cell": args.cell, "split": args.split, "n_utts": kept,
         "seed": args.seed, "checkpoint": str(args.checkpoint), "rows": rows}, indent=1))
    best = min((r for r in rows if r["setting"] != "no-LM"), key=lambda r: r["wer"])
    print(f"\nbest: {best['setting']}  WER {best['wer']:.2f}  "
          f"(no-LM reference {rows[0]['wer']:.2f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
