"""Dev-only sweep of the second-pass rescoring weight (lambda) for aed_beam_lm / joint_beam_lm.

The +LM cells on the AED and joint read-outs integrate the external LM by rescoring the
completed n-best rather than by first-pass shallow fusion (runlog 2026-07-23):

    S(h) = (1-ctc_weight)*AED(h) + ctc_weight*CTC(h) + lambda*logP_LM(h)

The acoustic beam is LM-free, so the n-best and every hypothesis's AED/CTC/LM component score
are INDEPENDENT of lambda. This script therefore decodes each utterance ONCE, caches the n-best
with its component scores, and then sweeps lambda as pure re-ranking — the whole grid costs one
decode pass, and every lambda is compared on an identical hypothesis set (no re-decode noise).

Also reports:
  no-LM  : lambda=0, which reproduces the aed_beam / joint_beam cell exactly
  oracle : the lowest WER reachable by ANY re-ranking of this n-best — the ceiling on what
           second-pass rescoring could ever buy.

Run on FULL dev splits (default) — never on a test split, and never on a length-biased sample.

Usage:
  python scripts/analysis/sweep_rescore_weight.py --config configs/typeA_small_N1.yaml \
      --checkpoint checkpoints/typeA_small_N1/valid.wer.ave.pt --split dev-clean
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
from dcasr.decoders.joint import joint_beam_search_nbest                 # noqa: E402
from dcasr.eval.metrics import levenshtein_counts, normalize_text        # noqa: E402
from dcasr.eval.metrics import word_error_rate                           # noqa: E402
from dcasr.logging_utils import get_logger, setup_logging                # noqa: E402
from dcasr.tasks.asr_task import build_model                             # noqa: E402
from dcasr.tasks.build import build_cmvn, build_frontend                 # noqa: E402
from dcasr.tasks.decode_task import load_lm_scorer, load_model_weights   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
logger = get_logger(__name__)


def sample_manifest(src: Path, dst: Path, n: int, seed: int) -> int:
    """Write the manifest subset to decode. n<=0 keeps the FULL split (the default)."""
    lines = [ln for ln in src.read_text().splitlines() if ln.strip()]
    if n and n > 0 and n < len(lines):
        lines = random.Random(seed).sample(lines, n)
    dst.write_text("\n".join(lines) + "\n")
    return len(lines)


@torch.no_grad()
def nbest_for_batch(model, tokenizer, batch, dc, device, lm, read_out, nbest):
    """One batch -> per-utterance {id, ref, hyps:[{text, aed, ctc, lm, ntok}]}.

    The beam is acoustic-only; each hypothesis keeps its AED/CTC components and its full-sequence
    LM log-prob, which is everything the lambda re-ranking needs.
    """
    ctc_w = 0.0 if read_out == "aed" else float(dc.get("ctc_weight", 0.3))
    ctc_head = model.ctc_head if ctc_w > 0.0 else None
    beam_size = int(dc.get("beam_size", 10))
    pre = int(dc["pre_beam"]) if dc.get("pre_beam") else None
    length_bonus = float(dc.get("length_bonus", 0.0))
    tok = tokenizer

    feats = batch["feats"].to(device)
    flens = batch["feat_lens"].to(device)
    enc = model.encoder(feats, flens)
    out = []
    for i in range(feats.shape[0]):
        n = int(enc.lengths[i])
        hyps = joint_beam_search_nbest(
            ctc_head, model.aed_head, enc.features[i : i + 1, :n], enc.lengths[i : i + 1],
            beam_size=beam_size, ctc_weight=ctc_w, bos_id=tok.bos_id, eos_id=tok.eos_id,
            pad_id=tok.pad_id, length_bonus=length_bonus, pre_beam=pre, nbest=nbest)[0]
        lmv = lm.sequence_logprob([h.tokens for h in hyps], device).tolist()
        ref_ids = batch["tokens"][i, : int(batch["token_lens"][i])].tolist()
        out.append({"id": batch["ids"][i], "ref": tok.decode(ref_ids),
                    "hyps": [{"text": tok.decode(h.tokens), "aed": h.aed, "ctc": h.ctc,
                              "lm": lmv[j], "ntok": len(h.tokens)}
                             for j, h in enumerate(hyps)]})
    return out


def rerank(records, lam, ctc_w, gamma=0.0):
    """Pick each utterance's best hypothesis at this (lambda, gamma) -> (refs, hyps).

    gamma is the per-token insertion bonus (ESPnet's `penalty`). It is needed because
    logP_LM(h) is a sum of negative per-token terms, so lambda*logP_LM systematically favours
    SHORTER hypotheses; gamma*len offsets that length bias. gamma=0 is the pure-lambda result.
    """
    refs, hyps = [], []
    for r in records:
        best = max(r["hyps"],
                   key=lambda h: ((1 - ctc_w) * h["aed"] + ctc_w * h["ctc"]
                                  + lam * h["lm"] + gamma * h["ntok"]))
        refs.append(r["ref"])
        hyps.append(best["text"])
    return refs, hyps


def oracle_wer(records):
    """Lowest corpus WER reachable by ANY re-ranking of the cached n-best (per-utt argmin
    of word errors) — the ceiling on second-pass rescoring."""
    err = ref_len = 0
    for r in records:
        ref_w = normalize_text(r["ref"]).split()
        best = min(sum(levenshtein_counts(ref_w, normalize_text(h["text"]).split())[:3])
                   for h in r["hyps"])
        err += best
        ref_len += len(ref_w)
    return 100.0 * err / max(1, ref_len)


def row_for(records, lam, ctc_w, gamma=0.0):
    refs, hyps = rerank(records, lam, ctc_w, gamma)
    st = word_error_rate(refs, hyps)
    nref = sum(len(r.split()) for r in refs)
    nhyp = sum(len(h.split()) for h in hyps)
    return {"lambda": lam, "gamma": gamma, "wer": 100 * st.er, "sub": 100 * st.sub_rate,
            "del": 100 * st.del_rate, "ins": 100 * st.ins_rate,
            "len_ratio": nhyp / max(1, nref),
            "empty": sum(1 for h in hyps if not h.strip())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="dev-clean")
    ap.add_argument("--read-outs", default="aed,joint")
    ap.add_argument("--n", type=int, default=0, help="0 = the FULL split (default)")
    ap.add_argument("--seed", type=int, default=1234, help="only used when --n subsets")
    ap.add_argument("--nbest", type=int, default=0, help="0 = decode.beam_size")
    ap.add_argument("--lambdas",
                    default="0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.25,1.5,2.0")
    ap.add_argument("--gammas",
                    default="0.0,0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0",
                    help="per-token insertion bonus offsetting the LM's length bias "
                         "(ESPnet `penalty`); gamma=0 is the pure-lambda result and the "
                         "lambda=0 row is the gamma-only control (no LM consulted)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cache", default=None, help="n-best cache JSON; reused if present")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    run_name = str(cfg.experiment.name)
    setup_logging(f"sweep_rescore_{run_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lambdas = [float(x) for x in args.lambdas.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]
    tag = f".{args.tag}" if args.tag else ""
    outdir = REPO / "experiments" / run_name / "rescore_sweep"
    outdir.mkdir(parents=True, exist_ok=True)

    model = tokenizer = lm = None
    for read_out in args.read_outs.split(","):
        cache = (Path(args.cache) if args.cache
                 else outdir / f"nbest.{read_out}.{args.split}{tag}.json")
        if cache.exists():
            blob = json.loads(cache.read_text())
            records, kept = blob["records"], blob["n_utts"]
            logger.info("reusing cached n-best %s (%d utts)", cache.name, kept)
        else:
            if model is None:                       # build once, reuse across read-outs
                tokenizer = Tokenizer(REPO / cfg.bpemodel)
                frontend = build_frontend(cfg)
                cmvn = build_cmvn(cfg, REPO)
                model = build_model(cfg, tokenizer.vocab_size).to(device).eval()
                load_model_weights(model, REPO / args.checkpoint)
                dc_all = OmegaConf.to_container(cfg.decode, resolve=True)
                lm = load_lm_scorer(dc_all, REPO, tokenizer, device)
            dc = OmegaConf.to_container(cfg.decode, resolve=True)
            nbest = args.nbest or int(dc.get("beam_size", 10))
            src = REPO / str(cfg.data.manifests_dir) / f"{args.split}.jsonl"
            sub = outdir / f"{args.split}{tag}.manifest.jsonl"
            kept = sample_manifest(src, sub, args.n, args.seed)
            ds = LibriSpeechDataset(sub, frontend, tokenizer, cmvn=cmvn, specaugment=None,
                                    augment=False)
            loader, _ = make_dataloader(ds, int(cfg.batch_bins), augment=False, num_workers=2,
                                        world_size=1, rank=0)
            logger.info("decoding %d %s utts, read_out=%s, nbest=%d", kept, args.split,
                        read_out, nbest)
            records, t0, done = [], time.perf_counter(), 0
            for b in loader:
                records += nbest_for_batch(model, tokenizer, b, dc, device, lm, read_out, nbest)
                done += len(b["ids"])
                if done % 200 < len(b["ids"]):
                    el = time.perf_counter() - t0
                    logger.info("  %d/%d utts  %.1fs  (%.2f s/utt)", done, kept, el, el / done)
            cache.write_text(json.dumps({"run": run_name, "read_out": read_out,
                                         "split": args.split, "n_utts": kept,
                                         "nbest": nbest, "records": records}))
            logger.info("cached n-best -> %s (%.0fs)", cache, time.perf_counter() - t0)

        ctc_w = 0.0 if read_out == "aed" else float(cfg.decode.get("ctc_weight", 0.3))
        rows = [row_for(records, lam, ctc_w, g) for lam in lambdas for g in gammas]
        orc = oracle_wer(records)
        depth = sum(len(r["hyps"]) for r in records) / max(1, len(records))
        base = row_for(records, 0.0, ctc_w, 0.0)["wer"]        # == the no-LM cell exactly
        by = {(r["lambda"], r["gamma"]): r for r in rows}

        print(f"\n===== {run_name} / {args.split} / {read_out}_beam_lm / n={kept} "
              f"(mean n-best depth {depth:.1f}) =====")
        print("WER over lambda (rows) x gamma (cols); gamma=0 column is pure-lambda rescoring")
        print("lam\\gam".rjust(8) + "".join(f"{g:>7.2f}" for g in gammas))
        for lam in lambdas:
            print(f"{lam:>8.2f}" + "".join(f"{by[(lam, g)]['wer']:>7.2f}" for g in gammas),
                  flush=True)

        pure = min((r for r in rows if r["gamma"] == 0.0), key=lambda r: r["wer"])
        gonly = min((r for r in rows if r["lambda"] == 0.0), key=lambda r: r["wer"])
        best = min(rows, key=lambda r: r["wer"])
        edge = (best["lambda"] in (lambdas[0], lambdas[-1])
                or best["gamma"] in (gammas[0], gammas[-1]))
        print(f"\nno-LM (lambda=0,gamma=0)  WER {base:.2f}")
        print(f"CONTROL gamma-only        WER {gonly['wer']:.2f} ({gonly['wer'] - base:+.2f}) "
              f"at gamma={gonly['gamma']}  <- length correction WITHOUT the LM")
        print(f"best pure-lambda          WER {pure['wer']:.2f} ({pure['wer'] - base:+.2f}) "
              f"at lambda={pure['lambda']}")
        print(f"best (lambda, gamma)      WER {best['wer']:.2f} ({best['wer'] - base:+.2f}) "
              f"at lambda={best['lambda']} gamma={best['gamma']}  "
              f"[S {best['sub']:.2f} D {best['del']:.2f} I {best['ins']:.2f} "
              f"len {best['len_ratio']:.3f}]")
        print(f"  LM-attributable gain    {best['wer'] - gonly['wer']:+.2f} "
              f"(best vs gamma-only control)")
        print(f"n-best ORACLE ceiling     WER {orc:.2f}")
        if edge:
            print("  WARNING: optimum sits on a grid BOUNDARY — widen --lambdas/--gammas")

        out = outdir / f"{read_out}_beam_lm.{args.split}{tag}.sweep.json"
        out.write_text(json.dumps(
            {"run": run_name, "read_out": read_out, "split": args.split, "n_utts": kept,
             "checkpoint": str(args.checkpoint), "ctc_weight": ctc_w,
             "nbest_mean_depth": depth, "no_lm_wer": base, "oracle_wer": orc,
             "best_pure_lambda": pure, "best_overall": best, "rows": rows}, indent=1))
        print(f"-> {out}")


if __name__ == "__main__":
    main()
