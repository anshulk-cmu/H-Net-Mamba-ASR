"""Decode task: a trained checkpoint -> transcripts across the decode matrix (plan §6.3).

Cells are read-outs × search × ±LM per the plan's conventions: greedy is CTC-only
(fast reference / peakiness diagnostic); AED and joint are beam-only; +LM lives on the
beam side only — first-pass shallow fusion for CTC, second-pass n-best rescoring for the
aed/joint beam (the LM never enters their search). Per-utterance hyp/ref records (JSONL) feed
score_wer.py and the paired-bootstrap significance tests; per-cell timing feeds the
RTF/efficiency reporting (#8). CUDA-free imports — works with any model object
exposing .encoder / .ctc_head / .aed_head (duck-typed; scripts/decode.py builds the
real one).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from dcasr.decoders.ctc import ctc_prefix_beam_search
from dcasr.decoders.joint import joint_beam_search, joint_beam_search_nbest, lm_rescore
from dcasr.decoders.lm_fusion import CausalLMScorer
from dcasr.logging_utils import get_logger
from dcasr.tasks.build import _plain
from dcasr.tasks.lm_task import LMModel, build_lm

logger = get_logger(__name__)

SAMPLE_RATE = 16000


def _sync(device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def as_str_list(value, default: list[str]) -> list[str]:
    """Config value -> list of strings; a bare string becomes a one-element list."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def expand_cells(decode_cfg: Mapping[str, Any]) -> list[dict]:
    """decode: block -> ordered matrix cells. Greedy = CTC-only; AED/joint beam-only;
    +LM only on beam cells (and only when lm is enabled)."""
    dc = _plain(decode_cfg) or {}
    read_outs = as_str_list(dc.get("read_outs"), ["ctc"])
    searches = as_str_list(dc.get("search"), ["greedy", "beam"])
    lm_val = dc.get("lm", "none")                        # None/False/"none"/"false" all mean off
    with_lm = str(lm_val).lower() not in ("none", "false", "")
    cells, seen = [], set()
    for ro in read_outs:
        if ro not in ("ctc", "aed", "joint"):
            raise ValueError(f"unknown read_out {ro!r}")
        for s in searches:
            if s not in ("greedy", "beam"):
                raise ValueError(f"unknown search {s!r}")
            if s == "greedy" and ro != "ctc":
                continue                                 # plan: greedy is only meaningful for CTC
            for use_lm in ([False, True] if (s == "beam" and with_lm) else [False]):
                name = f"{ro}_{s}" + ("_lm" if use_lm else "")
                if name in seen:                         # duplicate config entries: decode once
                    continue
                seen.add(name)
                cells.append({"read_out": ro, "search": s, "lm": use_lm, "name": name})
    if not cells:                                        # e.g. read_outs=[aed] with search=greedy
        raise ValueError(f"decode config yields no cells (read_outs={read_outs}, "
                         f"search={searches}); greedy applies to the CTC read-out only")
    return cells


def check_heads(model, cells: list[dict]) -> None:
    """Requested read-outs must have their heads (explicit beats silent skips)."""
    for c in cells:
        if c["read_out"] in ("ctc", "joint") and model.ctc_head is None:
            raise ValueError(f"cell {c['name']}: model has no CTC head")
        if c["read_out"] in ("aed", "joint") and model.aed_head is None:
            raise ValueError(f"cell {c['name']}: model has no AED head")


def load_model_weights(model, ckpt_path: str | Path) -> dict:
    """Load weights from either a full trainer checkpoint or a .ave/.best file."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    meta = {k: state[k] for k in ("epoch", "global_step", "averaged_epochs") if k in state}
    logger.info("loaded weights %s %s", Path(ckpt_path).name, meta or "")
    return meta


def load_lm_scorer(decode_cfg: Mapping[str, Any], repo_root: str | Path, tokenizer,
                   device) -> CausalLMScorer:
    """Build the fusion LM from decode.lm_config + decode.lm_checkpoint (shared vocab)."""
    from omegaconf import OmegaConf
    dc = _plain(decode_cfg)
    if not dc.get("lm_config") or not dc.get("lm_checkpoint"):
        raise ValueError("decode.lm=shallow_fusion requires decode.lm_config and "
                         "decode.lm_checkpoint")
    lm_cfg = OmegaConf.load(Path(repo_root) / str(dc["lm_config"]))
    lmm = LMModel(build_lm(lm_cfg, tokenizer.vocab_size))
    load_model_weights(lmm, Path(repo_root) / str(dc["lm_checkpoint"]))
    if lmm.lm.vocab_size != tokenizer.vocab_size:        # the shared-vocab fusion contract
        raise ValueError(f"LM vocab {lmm.lm.vocab_size} != tokenizer {tokenizer.vocab_size}")
    return CausalLMScorer(lmm.lm.to(device).eval(), bos_id=tokenizer.bos_id,
                          eos_id=tokenizer.eos_id, pad_id=tokenizer.pad_id)


def length_bonus_for(decode_cfg: Mapping[str, Any]) -> float:
    """Per-token insertion bonus for the label-synchronous beam (`decode.length_bonus`).

    Uniform across all aed/joint cells. The +LM cells no longer need a bonus to offset a
    per-token LM cost: the LM enters as SECOND-PASS rescoring of the completed n-best
    (joint_beam_search_nbest -> lm_rescore), never inside the search, so it cannot truncate
    hypotheses. Default 0.0 (a positive value over-generates: bonus 2.0 -> length ratio 1.40).
    See runlog 2026-07-23.
    """
    return float(_plain(decode_cfg).get("length_bonus", 0.0))


@torch.no_grad()
def decode_batch(model, tokenizer, batch: dict, cell: Mapping[str, Any],
                 decode_cfg: Mapping[str, Any], device, lm=None) -> list[dict]:
    """One collated batch through one cell -> per-utterance {id, ref, hyp, decode_s}."""
    dc = _plain(decode_cfg)
    beam_size = int(dc.get("beam_size", 10))
    pre_beam = dc.get("pre_beam")
    # lm_weight -> CTC first-pass shallow fusion; rescore_weight (lambda) -> AED/joint second-
    # pass n-best rescoring. Kept distinct so ctc_beam_lm stays at its settled weight while the
    # AED/joint rescoring lambda is tuned independently; rescore_weight defaults to lm_weight.
    lm_weight = float(dc.get("lm_weight", 0.0)) if cell["lm"] else 0.0
    rescore_weight = float(dc.get("rescore_weight", lm_weight)) if cell["lm"] else 0.0
    # rescore_length_bonus (gamma) offsets the LM's length bias and is applied ONLY at re-rank
    # time, never in the search. It is deliberately NOT `decode.length_bonus`: that one feeds the
    # acoustic beam for every cell, and a positive value there over-generates on the no-LM cells
    # (measured 2.0 -> length ratio 1.40, WER 47%). The dev sweep produced the n-best with the
    # beam at length_bonus=0.0 and applied gamma post-hoc, so this mirrors it exactly.
    rescore_length_bonus = float(dc.get("rescore_length_bonus", 0.0)) if cell["lm"] else 0.0
    use_lm = lm if cell["lm"] else None
    if cell["lm"] and lm is None:
        raise ValueError(f"cell {cell['name']} needs decode.lm_checkpoint and lm_weight")
    length_bonus = length_bonus_for(dc)

    feats = batch["feats"].to(device)
    feat_lens = batch["feat_lens"].to(device)
    t0 = time.perf_counter()
    enc = model.encoder(feats, feat_lens)
    _sync(device)                                        # async CUDA launches must not leak
    enc_s = time.perf_counter() - t0                     # ...into the next timed region
    B = feats.shape[0]
    tok = tokenizer

    hyps: list[list[int]] = []
    times: list[float] = []
    if cell["read_out"] == "ctc" and cell["search"] == "greedy":
        t0 = time.perf_counter()
        hyps = model.ctc_head.greedy_decode(enc.features, enc.lengths)
        dt = time.perf_counter() - t0
        times = [dt / B] * B                             # batched: amortized per-utt time
    elif cell["read_out"] == "ctc":                      # prefix beam (±LM)
        t0 = time.perf_counter()
        logp = model.ctc_head.log_probs(enc.features)    # shared projection: timed + amortized
        _sync(device)
        enc_s += time.perf_counter() - t0
        for i in range(B):
            t0 = time.perf_counter()
            hyp = ctc_prefix_beam_search(
                logp[i : i + 1], enc.lengths[i : i + 1], blank_id=model.ctc_head.blank_id,
                beam_size=beam_size, pre_beam=int(pre_beam or 30),
                lm=use_lm, lm_weight=lm_weight, device=device)[0]
            times.append(time.perf_counter() - t0)
            hyps.append(hyp)
    else:                                                # aed / joint label-synchronous beam
        ctc_w = 0.0 if cell["read_out"] == "aed" else float(dc.get("ctc_weight", 0.3))
        ctc_head = model.ctc_head if ctc_w > 0.0 else None
        pre = int(pre_beam) if pre_beam else None
        for i in range(B):
            n = int(enc.lengths[i])
            t0 = time.perf_counter()
            if cell["lm"]:                               # second-pass n-best LM rescoring
                nbest = joint_beam_search_nbest(
                    ctc_head, model.aed_head, enc.features[i : i + 1, :n],
                    enc.lengths[i : i + 1], beam_size=beam_size, ctc_weight=ctc_w,
                    bos_id=tok.bos_id, eos_id=tok.eos_id, pad_id=tok.pad_id,
                    length_bonus=length_bonus, pre_beam=pre, nbest=beam_size)[0]
                hyp = lm_rescore(nbest, use_lm, rescore_weight, ctc_weight=ctc_w,
                                 device=device, length_bonus=rescore_length_bonus)
            else:
                hyp = joint_beam_search(
                    ctc_head, model.aed_head, enc.features[i : i + 1, :n],
                    enc.lengths[i : i + 1], beam_size=beam_size, ctc_weight=ctc_w,
                    bos_id=tok.bos_id, eos_id=tok.eos_id, pad_id=tok.pad_id,
                    length_bonus=length_bonus, pre_beam=pre)[0]
            times.append(time.perf_counter() - t0)
            hyps.append(hyp)

    per_utt_enc = enc_s / B                              # encoder cost amortized over the batch
    records = []
    for i in range(B):
        ref_ids = batch["tokens"][i, : int(batch["token_lens"][i])].tolist()
        records.append({"id": batch["ids"][i], "ref": tok.decode(ref_ids),
                        "hyp": tok.decode(hyps[i]),
                        "decode_s": round(per_utt_enc + times[i], 6)})
    return records


def decode_split(model, tokenizer, loader, cell, decode_cfg, device, *,
                 audio_seconds: Mapping[str, float], out_path: str | Path, lm=None) -> dict:
    """Decode one split through one cell; write per-utt JSONL; return the cell summary."""
    model_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    n, dec_s, aud_s = 0, 0.0, 0.0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warm = next(iter(loader), None)                      # untimed warmup: CUDA/kernel compile
    if warm is not None:                                 # must not land in the first cell's RTF
        with torch.no_grad():
            model.encoder(warm["feats"].to(device), warm["feat_lens"].to(device))
        _sync(device)
    with open(out_path, "w", encoding="utf-8") as w:
        for batch in loader:
            for r in decode_batch(model, tokenizer, batch, cell, decode_cfg, device, lm=lm):
                r["audio_s"] = round(audio_seconds.get(r["id"], 0.0), 3)
                w.write(json.dumps(r) + "\n")
                n += 1
                dec_s += r["decode_s"]
                aud_s += r["audio_s"]
    if model_training and hasattr(model, "train"):
        model.train()
    summary = {"cell": cell["name"], "n_utts": n, "decode_s": round(dec_s, 3),
               "audio_s": round(aud_s, 3),
               "rtf": round(dec_s / aud_s, 5) if aud_s > 0 else None}
    logger.info("decoded %s: %s", out_path.name, summary)
    return summary


def audio_seconds_from_manifest(manifest_path: str | Path) -> dict[str, float]:
    """{utt id: audio seconds} from the manifest's raw sample counts."""
    out = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                out[e["id"]] = e["frames"] / SAMPLE_RATE
    return out
