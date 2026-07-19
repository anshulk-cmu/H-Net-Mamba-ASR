"""Analytic efficiency accounting: parameter counts and inference GFLOPs (plan #8).

Closed-form params/FLOPs for the DC-ASR encoder as a function of the SAME config keys
the build seam consumes (encoder_conf/hnet/model_conf/aed_conf), so the efficiency
table is derived from the run config, not a profiler. Heads (pure torch) are counted
by CPU instantiation — exact by construction. Encoder formulas mirror the real
modules field-for-field and are pinned against instantiated models by GPU tests.

FLOP policy (recorded in every report): 1 MAC = 2 FLOPs; matmul/conv/SSD-scan terms
only; biases, normalization, activations, gates, residuals, reductions, and
gather/scatter are excluded (sub-percent); the Mamba-2 SSD scan is counted as the
linear recurrence (2·d_inner·d_state MACs/token for state update + readout); the EMA
dechunk smoother is counted as implemented (one causal L×L matmul, 2·L²·d FLOPs).
Compressed lengths use the stage keep-fractions (design target 1/N, or realised
values passed in), as continuous expectations.
"""
from __future__ import annotations

from typing import Any, Mapping

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

MAMBA2_DEFAULTS = dict(d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1)

ASSUMPTIONS = [
    "1 MAC = 2 FLOPs; matmul/conv/scan terms only",
    "biases, norms, activations, gates, residuals, reductions, gathers excluded (<1%)",
    "Mamba-2 SSD scan counted as the linear recurrence (2*d_inner*d_state MACs/token) = "
    "ALGORITHMIC flops; the chunked-SSD kernel's executed matmul work is ~2.1x that term "
    "(~+11-15% of encoder totals) — cross-cell comparisons use the same convention",
    "EMA smoother counted as implemented: one causal matmul, 2*L^2*d FLOPs per utterance",
    "input frames = 100 * audio_seconds (100 Hz frontend), one utterance per report",
    "compressed lengths = keep_fraction * L0 as continuous expectations",
    "AED secondary numbers assume a KV-cached decoder (teacher-forced equivalent); the "
    "implemented cache-less greedy is quadratic and re-projects memory K/V per step -> "
    "measured RTF (decode.py) is the ground truth for decode speed",
]


# ── params (closed forms; pinned against real modules by tests) ───────────────
def _check_headdim(d_model: int, expand: int, headdim: int) -> None:
    if (expand * d_model) % headdim:                      # mirrors MambaBlock's assert
        raise ValueError(f"expand*d_model ({expand * d_model}) not divisible by "
                         f"headdim ({headdim}) — no such Mamba2 exists")


def mamba2_params(d_model: int, *, d_state: int = 128, d_conv: int = 4, expand: int = 2,
                  headdim: int = 64, ngroups: int = 1) -> int:
    _check_headdim(d_model, expand, headdim)
    d_inner = expand * d_model
    nheads = d_inner // headdim
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    conv_dim = d_inner + 2 * ngroups * d_state
    return (d_model * d_in_proj + conv_dim * d_conv + conv_dim + 3 * nheads
            + d_inner + d_inner * d_model)


def mamba_stack_params(n_layers: int, d_model: int, bidirectional: bool = True,
                       **mamba_kw) -> int:
    per_block = 2 * d_model + (2 if bidirectional else 1) * mamba2_params(d_model, **mamba_kw)
    return n_layers * per_block + 2 * d_model             # blocks (pre-norm) + final LN


def conv_subsample_params(n_mels: int, d_model: int) -> int:
    f = ((n_mels - 1) // 2 - 1) // 2
    conv1 = d_model * 1 * 9 + d_model
    conv2 = d_model * d_model * 9 + d_model
    proj = d_model * f * d_model + d_model
    return conv1 + conv2 + proj


def chunker_params(kind: str, d_model: int, N: float) -> int:
    """Dynamic N>1 carries the 2-linear router; N=1 identity and fixed-pool carry none."""
    return 2 * d_model * d_model if (str(kind).lower() == "dynamic" and N != 1) else 0


def _linear_params(d_in: int, d_out: int) -> int:
    return d_in * d_out + d_out


def encoder_params(enc: Mapping[str, Any], n_mels: int = 80) -> dict:
    """Per-stage parameter counts for DCASREncoder from encoder_conf-style keys."""
    a = _arch(enc, n_mels)
    br: dict[str, int] = {"subsample": conv_subsample_params(a["n_mels"], a["d_outer"]),
                          "enc_stack": mamba_stack_params(a["n_enc"], a["d_outer"], a["bidir"]),
                          "dec_stack": mamba_stack_params(a["n_dec"], a["d_outer"], a["bidir"])}
    if a["type"] == "A":
        br["chunker"] = chunker_params(a["chunker"], a["d_outer"], a["N"])
        br["projections"] = (_linear_params(a["d_outer"], a["d_main"])
                             + _linear_params(a["d_main"], a["d_outer"]))
        br["main_stack"] = mamba_stack_params(a["n_main"], a["d_main"], a["bidir"])
    else:
        nb = a["N"] ** 0.5
        br["chunker"] = (chunker_params(a["chunker"], a["d_outer"], nb)
                         + chunker_params(a["chunker"], a["d_main"], nb))
        br["projections"] = (_linear_params(a["d_outer"], a["d_main"])
                             + _linear_params(a["d_main"], a["d_outer"]))
        br["mid_stack"] = mamba_stack_params(a["n_mid"], a["d_main"], a["bidir"])
        br["main_stack"] = mamba_stack_params(a["n_main"], a["d_main"], a["bidir"])
        br["mid_dec_stack"] = mamba_stack_params(a["n_mid"], a["d_main"], a["bidir"])
    return {"breakdown": br, "total": sum(br.values())}


def head_params(config: Mapping[str, Any], vocab_size: int) -> dict:
    """CTC/AED head params by CPU instantiation (pure torch; mirrors build_model gating)."""
    mc = dict(config.get("model_conf", {}) or {})
    ec = config["encoder_conf"]
    d_outer = int(ec["d_outer"])
    out = {"ctc_head": 0, "aed_head": 0}
    if float(mc.get("ctc_weight", 1.0)) <= 0 and float(mc.get("aed_weight", 0.0)) <= 0:
        raise ValueError("model_conf needs ctc_weight > 0 or aed_weight > 0 "
                         "(build_model would refuse this config)")
    if float(mc.get("ctc_weight", 1.0)) > 0:
        from dcasr.decoders.ctc import CTCHead
        out["ctc_head"] = sum(p.numel() for p in CTCHead(d_outer, int(vocab_size)).parameters())
    if float(mc.get("aed_weight", 0.0)) > 0:
        from dcasr.decoders.aed import AEDHead
        ac = dict(config.get("aed_conf", {}) or {})
        head = AEDHead(int(vocab_size), d_outer, n_layers=int(ac.get("n_layers", 6)),
                       n_heads=int(ac.get("n_heads", 4)), d_ff=int(ac.get("d_ff", 2048)))
        out["aed_head"] = sum(p.numel() for p in head.parameters())
    return out


# ── flops ─────────────────────────────────────────────────────────────────────
def mamba2_flops_per_token(d_model: int, *, d_state: int = 128, d_conv: int = 4,
                           expand: int = 2, headdim: int = 64, ngroups: int = 1) -> float:
    _check_headdim(d_model, expand, headdim)
    d_inner = expand * d_model
    nheads = d_inner // headdim
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    conv_dim = d_inner + 2 * ngroups * d_state
    macs = (d_model * d_in_proj + conv_dim * d_conv + 2 * d_inner * d_state
            + d_inner * d_model)
    return 2.0 * macs


def mamba_stack_flops(n_layers: int, d_model: int, n_tokens: float,
                      bidirectional: bool = True, **mamba_kw) -> float:
    return (n_layers * (2 if bidirectional else 1)
            * mamba2_flops_per_token(d_model, **mamba_kw) * n_tokens)


def subsampled_frames(n_frames: int) -> int:
    """Valid 25 Hz length after the two k=3,s=2 convs (encoder._subsampled_length)."""
    return max(((n_frames - 1) // 2 - 1) // 2, 0)


def conv_subsample_flops(n_frames: int, n_mels: int, d_model: int) -> float:
    t1, f1 = (n_frames - 1) // 2, (n_mels - 1) // 2
    t2, f2 = (t1 - 1) // 2, (f1 - 1) // 2
    conv1 = t1 * f1 * d_model * 9 * 1
    conv2 = t2 * f2 * d_model * 9 * d_model
    proj = t2 * (d_model * f2) * d_model
    return 2.0 * (conv1 + conv2 + proj)


def router_flops(d_model: int, n_tokens: float) -> float:
    return 2.0 * 2 * d_model * d_model * n_tokens         # W_q + W_k per frame


def ema_flops(n_tokens: float, d_model: int) -> float:
    return 2.0 * n_tokens * n_tokens * d_model            # causal LxL matmul as implemented


def _stage_active(chunker: str, n: float) -> bool:
    """Router+EMA run only for the dynamic chunker at N>1 (identity/fixed skip both)."""
    return str(chunker).lower() == "dynamic" and n != 1


def encoder_flops(enc: Mapping[str, Any], n_frames: int, n_mels: int = 80,
                  kept_fractions=None, ema: bool | None = None) -> dict:
    """Per-stage FLOPs for one utterance of n_frames 100 Hz input frames."""
    a = _arch(enc, n_mels)
    l0 = float(subsampled_frames(n_frames))
    use_ema = a["ema"] if ema is None else bool(ema)
    if kept_fractions is None:
        kept = a["kept_default"]
    else:
        if not isinstance(kept_fractions, (list, tuple)):
            raise ValueError(f"kept_fractions must be a list, got "
                             f"{type(kept_fractions).__name__} ({kept_fractions!r})")
        kept = [float(k) for k in kept_fractions]
    if len(kept) != a["n_stages"]:
        raise ValueError(f"need {a['n_stages']} kept fraction(s) for type {a['type']}, "
                         f"got {len(kept)}")
    bad = [k for k in kept if not 0.0 < k <= 1.0]
    if bad:                                              # a realised keep is in (0, 1]
        raise ValueError(f"kept fraction(s) outside (0, 1]: {bad}")
    br: dict[str, float] = {"subsample": conv_subsample_flops(n_frames, a["n_mels"], a["d_outer"]),
                            "enc_stack": mamba_stack_flops(a["n_enc"], a["d_outer"], l0, a["bidir"]),
                            "dec_stack": mamba_stack_flops(a["n_dec"], a["d_outer"], l0, a["bidir"])}
    if a["type"] == "A":
        m = kept[0] * l0
        act = _stage_active(a["chunker"], a["N"])
        br["router"] = router_flops(a["d_outer"], l0) if act else 0.0
        br["ema"] = ema_flops(l0, a["d_outer"]) if (act and use_ema) else 0.0
        br["projections"] = 2.0 * (m * a["d_outer"] * a["d_main"]) * 2
        br["main_stack"] = mamba_stack_flops(a["n_main"], a["d_main"], m, a["bidir"])
        compressed = [m]
    else:
        nb = a["N"] ** 0.5
        m1, m2 = kept[0] * l0, kept[0] * kept[1] * l0
        act = _stage_active(a["chunker"], nb)
        br["router"] = ((router_flops(a["d_outer"], l0)
                         + router_flops(a["d_main"], m1)) if act else 0.0)
        br["ema"] = ((ema_flops(m1, a["d_main"]) + ema_flops(l0, a["d_outer"]))
                     if (act and use_ema) else 0.0)
        br["projections"] = 2.0 * (m1 * a["d_outer"] * a["d_main"]) * 2
        br["mid_stack"] = mamba_stack_flops(a["n_mid"], a["d_main"], m1, a["bidir"])
        br["main_stack"] = mamba_stack_flops(a["n_main"], a["d_main"], m2, a["bidir"])
        br["mid_dec_stack"] = mamba_stack_flops(a["n_mid"], a["d_main"], m1, a["bidir"])
        compressed = [m1, m2]
    return {"breakdown": br, "total": sum(br.values()), "frames_25hz": l0,
            "compressed_frames": compressed, "kept_fractions": kept}


def ctc_head_flops(d_model: int, vocab_size: int, n_tokens: float) -> float:
    return 2.0 * n_tokens * d_model * (vocab_size + 1)


def aed_flops_per_token(vocab_size: int, d_model: int, n_layers: int, d_ff: int,
                        memory_len: float, ctx_len: float) -> dict:
    """Secondary, decode-dependent: per generated token (teacher-forced equivalent)
    plus the once-per-utterance memory K/V projections."""
    per_layer = (4 * d_model * d_model + 2 * ctx_len * d_model      # self-attn
                 + 2 * d_model * d_model + 2 * memory_len * d_model  # cross-attn (q/out + scores)
                 + 2 * d_model * d_ff)                               # feed-forward
    per_token = 2.0 * (n_layers * per_layer + d_model * vocab_size)
    kv_per_utt = 2.0 * n_layers * 2 * memory_len * d_model * d_model
    return {"per_token": per_token, "memory_kv_per_utt": kv_per_utt,
            "ctx_len": ctx_len, "memory_len": memory_len}


# ── config adapter + report ──────────────────────────────────────────────────
def _arch(enc: Mapping[str, Any], n_mels: int) -> dict:
    h = dict(enc.get("hnet", {}) or {})
    t = str(enc["arch_type"])
    if t not in ("A", "B"):
        raise ValueError(f"arch_type must be 'A' or 'B', got {t!r}")
    n = int(h.get("compression_N", 1))                    # same cast as the build seam
    chunker = str(h.get("chunker", "dynamic")).lower()
    if t == "B" and chunker == "fixed" and (n ** 0.5) % 1 != 0:
        raise ValueError(f"Type B fixed-pool needs a perfect-square N, got {n} "
                         "(the real FixedPoolChunker would refuse to build)")
    n_stages = 1 if t == "A" else 2
    kept_default = [1.0 / n] if t == "A" else [1.0 / n ** 0.5] * 2
    return {"type": t, "N": n, "n_stages": n_stages, "kept_default": kept_default,
            "n_mels": int(n_mels), "d_outer": int(enc["d_outer"]),
            "d_main": int(enc["d_main"]), "n_enc": int(enc["n_enc"]),
            "n_main": int(enc["n_main"]), "n_dec": int(enc["n_dec"]),
            "n_mid": int(enc.get("n_mid", 4)),
            "bidir": bool(enc.get("bidirectional", True)),
            "chunker": chunker,
            "ema": bool(h.get("ema_smoothing", True))}


def efficiency_report(config: Mapping[str, Any], vocab_size: int,
                      audio_seconds: float = 10.0, kept_fractions=None) -> dict:
    """Params + GFLOPs for one config; the paper's per-cell efficiency numbers."""
    if not audio_seconds > 0:
        raise ValueError(f"audio_seconds must be positive, got {audio_seconds}")
    enc = config["encoder_conf"]
    n_mels = int((config.get("frontend_conf", {}) or {}).get("n_mels", 80))
    n_frames = int(round(100 * audio_seconds))
    a = _arch(enc, n_mels)

    p_enc = encoder_params(enc, n_mels)
    p_heads = head_params(config, vocab_size)
    params = {"encoder": p_enc["total"], **p_heads,
              "total": p_enc["total"] + p_heads["ctc_head"] + p_heads["aed_head"],
              "encoder_breakdown": p_enc["breakdown"]}

    f_enc = encoder_flops(enc, n_frames, n_mels, kept_fractions)
    br = dict(f_enc["breakdown"])
    if p_heads["ctc_head"]:
        br["ctc_head"] = ctc_head_flops(a["d_outer"], int(vocab_size), f_enc["frames_25hz"])
    total = sum(br.values())
    flops = {"audio_seconds": float(audio_seconds), "input_frames": n_frames,
             "frames_25hz": f_enc["frames_25hz"],
             "kept_fractions": f_enc["kept_fractions"],
             "compressed_frames": f_enc["compressed_frames"],
             "gflops_total": total / 1e9,
             "gflops_per_second": total / 1e9 / max(audio_seconds, 1e-9),
             "breakdown_gflops": {k: v / 1e9 for k, v in br.items()}}
    if p_heads["aed_head"]:
        ac = dict(config.get("aed_conf", {}) or {})
        flops["aed_secondary"] = aed_flops_per_token(
            int(vocab_size), a["d_outer"], int(ac.get("n_layers", 6)),
            int(ac.get("d_ff", 2048)), memory_len=f_enc["frames_25hz"], ctx_len=32.0)

    arch = {k: a[k] for k in ("type", "N", "chunker", "d_outer", "d_main", "n_enc",
                              "n_main", "n_dec", "n_mid", "bidir", "ema")}
    arch["vocab"] = int(vocab_size)
    return {"arch": arch, "params": params, "flops": flops, "assumptions": ASSUMPTIONS}


def format_efficiency(report: Mapping[str, Any]) -> str:
    a, p, f = report["arch"], report["params"], report["flops"]
    lines = [f"arch type {a['type']}  N={a['N']:g}  chunker={a['chunker']}  "
             f"d {a['d_outer']}/{a['d_main']}  layers {a['n_enc']}/{a['n_main']}/{a['n_dec']}"
             + (f" (mid {a['n_mid']})" if a["type"] == "B" else "")
             + f"  bidir={a['bidir']}  V={a['vocab']}",
             f"params: total {p['total']/1e6:.2f}M  (encoder {p['encoder']/1e6:.2f}M, "
             f"ctc {p['ctc_head']/1e6:.2f}M, aed {p['aed_head']/1e6:.2f}M)",
             f"flops @ {f['audio_seconds']:g}s: {f['gflops_total']:.2f} GFLOPs "
             f"({f['gflops_per_second']:.2f} GFLOPs/s), kept={f['kept_fractions']}"]
    lines.append(f"{'stage':<16}{'params(M)':>10}{'GFLOPs':>10}{'share':>8}")
    total = f["gflops_total"] or 1.0
    stages = {**p["encoder_breakdown"], "ctc_head": p["ctc_head"],
              "aed_head": p["aed_head"]}
    for k, g in f["breakdown_gflops"].items():
        pm = stages.get(k, 0) / 1e6
        lines.append(f"{k:<16}{pm:>10.2f}{g:>10.2f}{100 * g / total:>7.1f}%")
    return "\n".join(lines)
