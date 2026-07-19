"""Entry point: decode a trained DC-ASR checkpoint across the decode matrix (plan §6.3).

Thin orchestrator over tested library code (tasks/decode_task.py). Cells =
read_outs × search × ±LM from the config's decode: block; every cell × split writes
per-utterance {id, ref, hyp, decode_s, audio_s} JSONL (consumed by score_wer.py and
the bootstrap significance tests) plus a summary.json with per-cell RTF + provenance.

Usage:
    python scripts/decode.py --config configs/typeA_small_N1_ctc.yaml \
        --checkpoint checkpoints/<run>/valid.wer.ave.pt
    python scripts/decode.py --config <cfg> --checkpoint <ckpt> \
        decode.read_outs='[ctc,aed,joint]' decode.lm=shallow_fusion \
        decode.lm_config=configs/lm_transformer_500.yaml \
        decode.lm_checkpoint=checkpoints/lm_transformer_500/valid.loss.ave.pt \
        decode.lm_weight=0.3
    # beam-width knee sweep: rerun with decode.beam_size=4/8/10/16 on a dev split
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dcasr.data.librispeech import LibriSpeechDataset, make_dataloader
from dcasr.data.tokenizer import Tokenizer
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.provenance import collect_provenance
from dcasr.tasks.asr_task import build_model
from dcasr.tasks.build import build_cmvn, build_frontend
from dcasr.tasks.decode_task import (as_str_list, audio_seconds_from_manifest,
                                     check_heads, decode_split, expand_cells,
                                     load_lm_scorer, load_model_weights)

REPO = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def load_config(config_path, overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run(cfg, checkpoint, repo_root=REPO, device=None):
    """Decode every configured cell × split. Returns {split: {cell: summary}}."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = str(cfg.experiment.name)
    dc = cfg.decode
    cells = expand_cells(dc)

    tokenizer = Tokenizer(repo_root / cfg.bpemodel)
    frontend = build_frontend(cfg)
    cmvn = build_cmvn(cfg, repo_root)
    model = build_model(cfg, tokenizer.vocab_size).to(device).eval()
    meta = load_model_weights(model, repo_root / str(checkpoint))
    check_heads(model, cells)
    lm = (load_lm_scorer(dc, repo_root, tokenizer, device)
          if any(c["lm"] for c in cells) else None)

    data = cfg.data
    splits = as_str_list(dc.get("splits"),
                         list(data.dev_splits) + list(data.get("test_splits", [])))
    mdir = Path(repo_root) / str(data.manifests_dir)

    out_root = (Path(repo_root) / "experiments" / run_name / "decode" /
                Path(str(checkpoint)).stem)
    out_root.mkdir(parents=True, exist_ok=True)
    lm_files = ({"lm_checkpoint": repo_root / str(dc["lm_checkpoint"])}
                if lm is not None else {})
    prov = collect_provenance(
        cfg, repo_root=repo_root, world_size=1, seed=int(cfg.experiment.seed),
        manifests=[mdir / f"{s}.jsonl" for s in splits],
        extra_files={"asr_checkpoint": repo_root / str(checkpoint), **lm_files},
        extra={"run_name": run_name, "checkpoint_meta": meta,
               "cells": [c["name"] for c in cells], "device": device})

    results: dict[str, dict] = {}
    for split in splits:
        manifest = mdir / f"{split}.jsonl"
        ds = LibriSpeechDataset(manifest, frontend, tokenizer, cmvn=cmvn,
                                specaugment=None, augment=False)
        loader, _ = make_dataloader(ds, int(cfg.batch_bins), augment=False,
                                    num_workers=int(cfg.get("num_workers", 4)),
                                    world_size=1, rank=0)
        audio_s = audio_seconds_from_manifest(manifest)
        results[split] = {}
        for cell in cells:
            summary = decode_split(model, tokenizer, loader, cell, dc, device,
                                   audio_seconds=audio_s,
                                   out_path=out_root / cell["name"] / f"{split}.jsonl",
                                   lm=lm)
            results[split][cell["name"]] = summary
    (out_root / "summary.json").write_text(
        json.dumps({"results": results, "provenance": prov}, indent=1))
    logger.info("decode done -> %s", out_root)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML of the trained run")
    ap.add_argument("--checkpoint", required=True,
                    help="weights to decode (e.g. checkpoints/<run>/valid.wer.ave.pt)")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(f"decode_{cfg.experiment.name}")
    logger.info("config=%s checkpoint=%s overrides=%s", args.config, args.checkpoint,
                args.overrides)
    run(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
