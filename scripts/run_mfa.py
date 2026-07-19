"""Entry point: MFA forced alignment -> phone/word boundary ground truth (plan #9).

Thin orchestrator over tested library code (interp/alignments.py). For each split
(dev sets + a seeded train-960 subset): lay the manifest out as an MFA corpus
(symlinks + .lab), shell out to the MFA CLI (its own conda env — NOT hnet-asr),
parse the TextGrids, integrity-check every record against its transcript and audio
duration, and write alignments/<name>.jsonl + summary.json with provenance.

Usage:
    python scripts/run_mfa.py                          # dev-clean, dev-other, 10h train subset
    python scripts/run_mfa.py --splits dev-clean --train-subset-hours 0
    python scripts/run_mfa.py --only-parse             # re-parse existing TextGrids
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.interp.alignments import (SAMPLE_RATE, alignment_record, check_alignment,
                                     load_manifest, parse_textgrid, prepare_corpus,
                                     select_subset, speaker_of, write_alignments)
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.provenance import collect_provenance

REPO = Path(__file__).resolve().parents[1]
MFA_BIN = "/data/user_data/anshulk/envs/mfa/bin/mfa"
MFA_ROOT = "/data/user_data/anshulk/hnet-asr/data/mfa"
logger = get_logger(__name__)


def run_mfa_align(corpus_dir, out_dir, *, mfa_bin=MFA_BIN, mfa_root=MFA_ROOT,
                  model="english_us_arpa", num_jobs=8) -> None:
    """Invoke the MFA CLI (external env) on a prepared corpus."""
    env = {**os.environ, "MFA_ROOT_DIR": str(mfa_root),
           "PATH": f"{Path(mfa_bin).parent}:{os.environ.get('PATH', '')}"}
    # no --final_clean: a mid-run crash with it leaves the extracted model cache
    # partial (missing final.mdl) and poisons every later run until manual repair
    cmd = [str(mfa_bin), "align", str(corpus_dir), model, model, str(out_dir),
           "--clean", "--overwrite", "-j", str(num_jobs),
           "--output_format", "long_textgrid"]
    logger.info("mfa: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"mfa binary not found: {mfa_bin}") from e
    if proc.returncode != 0:
        logger.error("mfa align failed:\n%s", proc.stderr[-3000:])
        raise RuntimeError(f"mfa align exited {proc.returncode} for {corpus_dir}")


def parse_split(entries, tg_dir):
    """TextGrids -> checked records. Only CLEAN records enter the ground truth;
    corrupt grids and integrity failures are listed, never silently included.
    Returns (records, rejected_records, missing_ids, problem_lines)."""
    records, rejected, missing, problems = [], [], [], []
    for e in entries:
        tg = Path(tg_dir) / speaker_of(e["id"]) / f"{e['id']}.TextGrid"
        if not tg.is_file():
            missing.append(e["id"])
            continue
        try:
            rec = alignment_record(e["id"],
                                   parse_textgrid(tg.read_text(encoding="utf-8")))
        except ValueError as err:                     # one corrupt grid must not kill
            problems.append(f"{e['id']}: unparseable TextGrid: {err}")
            missing.append(e["id"])
            continue
        utt_problems = check_alignment(rec, e["text"], e["frames"] / SAMPLE_RATE)
        if utt_problems:
            problems.extend(f"{e['id']}: {p}" for p in utt_problems)
            rejected.append(rec)
        else:
            records.append(rec)
    return records, rejected, missing, problems


def process_split(name, entries, out_root, args) -> dict:
    """Prepare -> align -> parse -> check -> persist one split. Returns its summary."""
    corpus = out_root / "corpus" / name
    tg_dir = out_root / "textgrids" / name
    if not args.only_parse:
        if corpus.exists():
            shutil.rmtree(corpus)
        prepare_corpus(entries, corpus)
        run_mfa_align(corpus, tg_dir, mfa_bin=args.mfa_bin, mfa_root=args.mfa_root,
                      model=args.model, num_jobs=args.num_jobs)
    records, rejected, missing, problems = parse_split(entries, tg_dir)
    if not records:
        raise RuntimeError(f"{name}: 0 clean alignments out of {len(entries)} utts "
                           f"(textgrids under {tg_dir}) — refusing to write empty "
                           "ground truth")
    write_alignments(records, out_root / f"{name}.jsonl")
    if rejected:                                      # inspectable, never ground truth
        write_alignments(rejected, out_root / f"{name}.rejected.jsonl")
    for line in problems[:20]:
        logger.warning("integrity: %s", line)
    summary = {"split": name, "n_utts": len(entries), "aligned": len(records),
               "rejected": [r["id"] for r in rejected], "missing": missing,
               "n_problems": len(problems), "problems": problems[:200]}
    logger.info("%s: %d/%d clean, %d rejected, %d missing, %d integrity problems",
                name, len(records), len(entries), len(rejected), len(missing),
                len(problems))
    return summary


def mfa_version(mfa_bin, mfa_root) -> str:
    proc = subprocess.run([str(mfa_bin), "version"], capture_output=True, text=True,
                          env={**os.environ, "MFA_ROOT_DIR": str(mfa_root)})
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "unknown"


def run(args) -> dict:
    out_root = REPO / args.out
    summaries = {}
    for split in args.splits:
        entries = load_manifest(REPO / args.manifests_dir / f"{split}.jsonl")
        summaries[split] = process_split(split, entries, out_root, args)
    if args.train_subset_hours > 0:
        train = load_manifest(REPO / args.manifests_dir / "train-960.jsonl")
        subset = select_subset(train, args.train_subset_hours, seed=args.seed)
        write_alignments(subset, out_root / "train_subset_manifest.jsonl")
        summaries["train-subset"] = process_split("train-subset", subset, out_root, args)

    prov = collect_provenance(
        vars(args), repo_root=REPO, world_size=1, seed=args.seed, manifests=[],
        extra_files={f"alignments/{n}": out_root / f"{n}.jsonl" for n in summaries},
        extra={"mfa_version": mfa_version(args.mfa_bin, args.mfa_root),
               "model": args.model,
               "splits": {n: {k: v for k, v in s.items() if k != "problems"}
                          for n, s in summaries.items()}})
    (out_root / "summary.json").write_text(
        json.dumps({"splits": summaries, "provenance": prov}, indent=1))
    logger.info("alignments -> %s", out_root)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="*", default=["dev-clean", "dev-other"])
    ap.add_argument("--train-subset-hours", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--out", default="alignments")
    ap.add_argument("--mfa-bin", default=MFA_BIN)
    ap.add_argument("--mfa-root", default=MFA_ROOT)
    ap.add_argument("--model", default="english_us_arpa")
    ap.add_argument("--num-jobs", type=int, default=8)
    ap.add_argument("--only-parse", action="store_true",
                    help="re-parse existing TextGrids without re-aligning")
    args = ap.parse_args()

    setup_logging("run_mfa")
    logger.info("args=%s", vars(args))
    run(args)


if __name__ == "__main__":
    main()
