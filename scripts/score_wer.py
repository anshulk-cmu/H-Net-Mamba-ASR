"""Entry point: score decode.py's outputs -> WER/CER tables, S/D/I, significance, gate.

Thin orchestrator over tested library code (eval/score.py). Scores every
<cell>/<split>.jsonl under the decode dir, runs the paired bootstrap between cells
per split, checks goal.sane_test_clean_wer_below, and writes
score/{scores.json, report.txt, <cell>/<split>.jsonl (per-utt S/D/I counts)}.

Usage:
    python scripts/score_wer.py --config configs/typeA_small_N1_ctc.yaml \
        --checkpoint checkpoints/<run>/valid.wer.ave.pt
    python scripts/score_wer.py --config <cfg> --decode-dir experiments/<run>/decode/<stem>
    # knobs (optional score: block or dotlist): score.n_bootstrap=10000 score.seed=0
    #   score.normalize=true score.gate_split=test-clean score.gate_cell=ctc_greedy
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.eval.score import discover_cells, format_report, score_decode_dir
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.provenance import collect_provenance
from dcasr.tasks.build import _plain

REPO = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def load_config(config_path, overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def resolve_decode_dir(cfg, checkpoint=None, decode_dir=None, repo_root=REPO):
    if decode_dir:
        return Path(repo_root) / str(decode_dir)
    if checkpoint:
        return (Path(repo_root) / "experiments" / str(cfg.experiment.name) / "decode" /
                Path(str(checkpoint)).stem)
    raise ValueError("need --checkpoint or --decode-dir to locate the decode outputs")


def run(cfg, checkpoint=None, decode_dir=None, repo_root=REPO):
    """Score one decode dir. Returns {splits, gate} (also persisted to score/)."""
    ddir = resolve_decode_dir(cfg, checkpoint, decode_dir, repo_root)
    sc = _plain(cfg.get("score")) or {}
    n_bootstrap = int(sc.get("n_bootstrap", 10000))
    seed = int(sc.get("seed", cfg.experiment.seed))
    results = score_decode_dir(
        ddir, normalize=bool(sc.get("normalize", True)), n_bootstrap=n_bootstrap,
        seed=seed, goal_cfg=_plain(cfg.get("goal")),
        gate_split=str(sc.get("gate_split", "test-clean")),
        gate_cell=sc.get("gate_cell"))

    tree = discover_cells(ddir)
    scored_files = {f"{cell}/{split}": path for split, cells in tree.items()
                    for cell, path in cells.items()}
    summary = ddir / "summary.json"
    if summary.exists():
        scored_files["decode_summary"] = summary
    prov = collect_provenance(
        cfg, repo_root=repo_root, world_size=1, seed=seed, manifests=[],
        extra_files=scored_files,
        extra={"decode_dir": str(ddir), "n_bootstrap": n_bootstrap,
               "splits": sorted(tree), "cells": sorted({c for v in tree.values() for c in v})})

    report = format_report(results["splits"], results["gate"])
    out = ddir / "score"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps({**results, "provenance": prov}, indent=1))
    (out / "report.txt").write_text(report + "\n")
    for line in report.splitlines():
        logger.info("%s", line)
    logger.info("scores -> %s", out)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML of the decoded run")
    ap.add_argument("--checkpoint", help="checkpoint decode.py was run with (locates the dir)")
    ap.add_argument("--decode-dir", help="explicit decode dir (overrides --checkpoint)")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()
    if not args.checkpoint and not args.decode_dir:
        ap.error("need --checkpoint or --decode-dir")

    cfg = load_config(args.config, args.overrides)
    setup_logging(f"score_{cfg.experiment.name}")
    logger.info("config=%s checkpoint=%s decode_dir=%s overrides=%s", args.config,
                args.checkpoint, args.decode_dir, args.overrides)
    run(cfg, args.checkpoint, args.decode_dir)


if __name__ == "__main__":
    main()
