"""Entry point: interpretability suite over a trained DC-ASR checkpoint (plan #12 / §6.4).

Thin orchestrator over tested library code (interp/driver.py). Modes:
  boundaries  learned-boundary P/R/F1 + R-value per stage x {words, phones}
              vs MFA ground truth, with the random-baseline floor attached
  probes      phone_id / phone_class / word_id linear probes per level
              (train = MFA train subset, test = the eval split; utterance
              disjointness asserted on the ids actually consumed)
  robustness  boundary shift under noise / speed / silence perturbations
              (needs the clean boundaries, so it also writes boundaries.json)
  emergence   boundaries (+ probes) per retained epoch*.pt beside the
              checkpoint (needs keep_all_checkpoints) -> TB + emergence.json;
              runs last: it overwrites the model weights per epoch

All knobs live in the optional interp: config block (defaults inline here).
Outputs: experiments/<run>/interp/<ckpt_stem>/{boundaries,probes,robustness}.json
+ summary.json (provenance); emergence under experiments/<run>/interp/.

Usage:
    python scripts/run_interp.py --config configs/<run>.yaml \
        --checkpoint checkpoints/<run>/valid.wer.ave.pt
    python scripts/run_interp.py --config <cfg> --checkpoint <ckpt> \
        --modes boundaries,probes interp.max_utts=50
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dcasr.data.librispeech import LibriSpeechDataset, load_manifest, make_dataloader
from dcasr.data.tokenizer import Tokenizer
from dcasr.interp.alignments import load_alignments
from dcasr.interp.boundary_align import collect_boundaries
from dcasr.interp.driver import (PerturbedDataset, boundary_report,
                                 durations_from_entries, emergence_report,
                                 list_epoch_checkpoints, perturbations_from_config,
                                 probe_report, robustness_report)
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.metrics_logger import MetricsLogger
from dcasr.provenance import collect_provenance
from dcasr.tasks.asr_task import build_model
from dcasr.tasks.build import build_cmvn, build_frontend
from dcasr.tasks.decode_task import load_model_weights

REPO = Path(__file__).resolve().parents[1]
MODES = ("boundaries", "probes", "robustness", "emergence")
logger = get_logger(__name__)


def load_config(config_path, overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def load_split(name, alignments_dir, manifests_dir, max_utts=0):
    """Aligned utterances of a split: (entries, {id: alignment record}).
    Manifest entries are restricted to clean-aligned ids (and vice versa)."""
    alignments = {r["id"]: r for r in load_alignments(alignments_dir / f"{name}.jsonl")}
    entries = [e for e in load_manifest(manifests_dir / f"{name}.jsonl")
               if e["id"] in alignments]
    if not entries:
        raise ValueError(f"{name}: no manifest entries with alignments")
    if max_utts:
        entries = entries[:max_utts]
    alignments = {e["id"]: alignments[e["id"]] for e in entries}
    logger.info("%s: %d aligned utterances%s", name, len(entries),
                f" (capped at {max_utts})" if max_utts else "")
    return entries, alignments


def make_eval_loader(entries, frontend, tokenizer, cmvn, batch_bins, num_workers):
    ds = LibriSpeechDataset(entries, frontend, tokenizer, cmvn=cmvn, augment=False)
    loader, _ = make_dataloader(ds, batch_bins, augment=False,
                                num_workers=num_workers, world_size=1, rank=0)
    return loader


def detect_n_stages(encoder, loader, device):
    with torch.no_grad():
        for batch in loader:
            enc = encoder(batch["feats"].to(device), batch["feat_lens"].to(device))
            return len(enc.boundaries)
    raise ValueError("empty loader")


def run(cfg, checkpoint, modes, repo_root=REPO, device=None):
    """Run the selected interp modes for one checkpoint. Returns {mode: report}."""
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"unknown modes {bad}; choose from {MODES}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = str(cfg.experiment.name)
    ic = cfg.get("interp") or {}
    pc = ic.get("probe") or {}
    ec = ic.get("emergence") or {}
    rc = ic.get("robustness") or {}

    tokenizer = Tokenizer(repo_root / cfg.bpemodel)
    frontend = build_frontend(cfg)
    cmvn = build_cmvn(cfg, repo_root)
    model = build_model(cfg, tokenizer.vocab_size).to(device).eval()
    meta = load_model_weights(model, repo_root / str(checkpoint))

    adir = repo_root / str(ic.get("alignments_dir", "alignments"))
    mdir = repo_root / str(cfg.data.manifests_dir)
    split = str(ic.get("eval_split", "dev-clean"))
    max_utts = int(ic.get("max_utts", 0))
    batch_bins = int(ic.get("batch_bins", cfg.batch_bins))
    num_workers = int(cfg.get("num_workers", 4))
    tol = float(ic.get("tol_s", 0.02))
    trials = int(ic.get("baseline_trials", 20))

    entries, alignments = load_split(split, adir, mdir, max_utts)
    durations = durations_from_entries(entries)
    loader = make_eval_loader(entries, frontend, tokenizer, cmvn, batch_bins,
                              num_workers)

    out_root = (repo_root / "experiments" / run_name / "interp" /
                Path(str(checkpoint)).stem / split)      # per-split: a dev-other
    out_root.mkdir(parents=True, exist_ok=True)          # rerun must not clobber

    results: dict = {}
    clean_bounds = None
    if {"boundaries", "robustness"} & set(modes):
        results["boundaries"], clean_bounds = boundary_report(
            model.encoder, loader, alignments, durations, device=device,
            tol=tol, baseline_trials=trials)
        (out_root / "boundaries.json").write_text(
            json.dumps(results["boundaries"], indent=1))

    probe_setup = None
    if {"probes", "emergence"} & set(modes):
        tr_manifest = str(ic.get("probe_train_manifest",
                                 "alignments/train_subset_manifest.jsonl"))
        tr_align_path = str(ic.get("probe_train_alignments",
                                   "alignments/train-subset.jsonl"))
        tr_alignments = {r["id"]: r
                         for r in load_alignments(repo_root / tr_align_path)}
        tr_entries = [e for e in load_manifest(repo_root / tr_manifest)
                      if e["id"] in tr_alignments]
        if max_utts:
            tr_entries = tr_entries[:max_utts]
        tr_alignments = {e["id"]: tr_alignments[e["id"]] for e in tr_entries}
        n_stages = (len(clean_bounds) if clean_bounds is not None
                    else detect_n_stages(model.encoder, loader, device))
        probe_setup = dict(
            train_loader_fn=lambda: make_eval_loader(
                tr_entries, frontend, tokenizer, cmvn, batch_bins, num_workers),
            test_loader_fn=lambda: loader,
            train_alignments=tr_alignments, n_stages=n_stages,
            levels=tuple(pc.get("levels", ("frames", "chunks"))),
            top_k_words=int(pc.get("top_k_words", 500)),
            max_iter=int(pc.get("max_iter", 1000)), C=float(pc.get("C", 1.0)),
            seed=int(pc.get("seed", 1)), backend=str(pc.get("backend", "sklearn")))

    if "probes" in modes:
        results["probes"] = probe_report(
            model.encoder, probe_setup["train_loader_fn"](),
            probe_setup["test_loader_fn"](), probe_setup["train_alignments"],
            alignments, device=device, n_stages=probe_setup["n_stages"],
            levels=probe_setup["levels"], top_k_words=probe_setup["top_k_words"],
            train_cap=int(pc.get("train_cap", 50000)),
            test_cap=int(pc.get("test_cap", 20000)),
            max_iter=probe_setup["max_iter"], C=probe_setup["C"],
            seed=probe_setup["seed"], backend=probe_setup["backend"])
        (out_root / "probes.json").write_text(json.dumps(results["probes"], indent=1))

    if "robustness" in modes:
        perts = perturbations_from_config(rc)
        rseed = int(rc.get("seed", 1))

        def collect_fn(pert):
            ds = PerturbedDataset(entries, frontend, tokenizer, cmvn, pert,
                                  seed=rseed)
            ploader, _ = make_dataloader(ds, batch_bins, augment=False,
                                         num_workers=num_workers,
                                         world_size=1, rank=0)
            return collect_boundaries(model.encoder, ploader, device)

        results["robustness"] = robustness_report(
            perts, collect_fn, clean_bounds, alignments, durations, tol=tol,
            baseline_trials=int(rc.get("baseline_trials", 10)))
        (out_root / "robustness.json").write_text(
            json.dumps(results["robustness"], indent=1))

    if "emergence" in modes:                    # last: reloads weights per epoch
        ckpts = list_epoch_checkpoints((repo_root / str(checkpoint)).parent)
        probe_fn = None
        e_levels = tuple(ec.get("probe_levels", ("chunks",)))
        if e_levels:
            probe_fn = lambda enc: probe_report(
                enc, probe_setup["train_loader_fn"](),
                probe_setup["test_loader_fn"](),
                probe_setup["train_alignments"], alignments, device=device,
                n_stages=probe_setup["n_stages"], levels=e_levels,
                top_k_words=probe_setup["top_k_words"],
                train_cap=int(ec.get("train_cap", 20000)),
                test_cap=int(ec.get("test_cap", 10000)),
                max_iter=probe_setup["max_iter"], C=probe_setup["C"],
                seed=probe_setup["seed"], backend=probe_setup["backend"])
        mlogger = MetricsLogger(run_name="emergence", root=out_root, rank=0)
        try:
            rows = emergence_report(model, ckpts, loader, alignments, durations,
                                    device=device, tol=tol,
                                    baseline_trials=int(ec.get("baseline_trials", 10)),
                                    probe_fn=probe_fn, mlogger=mlogger)
        finally:
            mlogger.close()
        results["emergence"] = rows
        (out_root / "emergence.json").write_text(json.dumps(rows, indent=1))

    prov = collect_provenance(
        cfg, repo_root=repo_root, world_size=1, seed=int(cfg.experiment.seed),
        manifests=[mdir / f"{split}.jsonl"],
        extra_files={"asr_checkpoint": repo_root / str(checkpoint),
                     f"alignments_{split}": adir / f"{split}.jsonl"},
        extra={"run_name": run_name, "checkpoint_meta": meta,
               "modes": list(modes), "eval_split": split, "device": device,
               "n_utts": len(entries)})
    (out_root / "summary.json").write_text(json.dumps(
        {"results": {k: v for k, v in results.items() if k != "emergence"},
         "n_emergence_epochs": len(results.get("emergence", [])),
         "provenance": prov}, indent=1))
    logger.info("interp done -> %s", out_root)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML of the trained run")
    ap.add_argument("--checkpoint", required=True,
                    help="weights to analyse (e.g. checkpoints/<run>/valid.wer.ave.pt)")
    ap.add_argument("--modes", default="boundaries,probes,robustness",
                    help="comma-separated subset of "
                         f"{','.join(MODES)} (emergence is opt-in)")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cfg = load_config(args.config, args.overrides)
    setup_logging(f"interp_{cfg.experiment.name}")
    logger.info("config=%s checkpoint=%s modes=%s overrides=%s", args.config,
                args.checkpoint, modes, args.overrides)
    run(cfg, args.checkpoint, modes)


if __name__ == "__main__":
    main()
