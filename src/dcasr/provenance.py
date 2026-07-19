"""Run provenance: everything needed to reproduce and trace a training run.

`collect_provenance(cfg, ...)` returns one nested, JSON-serializable dict capturing the
resolved config, git state (commit / dirty / diff), environment freeze (python / torch /
cuda / mamba + full package list), determinism flags + seed, effective global batch +
world_size, and content fingerprints (sha256) of the tokenizer / CMVN / manifests. The
entry point (scripts/train.py) collects it and hands it to the Trainer, which appends it
to experiments/<run>/summary.json at train start (surviving an early crash). Every
collector is defensive — provenance is metadata and must never crash the run.
"""
from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

# non-deterministic-kernel fact (proven in the runlog): resume restores state exactly,
# but the Mamba/causal-conv1d CUDA kernels are not bit-reproducible run-to-run.
_KERNEL_NOTE = ("mamba_ssm/causal-conv1d CUDA kernels are non-deterministic run-to-run "
                "(~1e-4 over epochs); resume restores all state exactly and adds no excess "
                "divergence, so run-to-run bit-reproducibility is hardware-limited")

# env vars that matter for reproducibility / DDP / SLURM (captured when present)
_ENV_ALLOWLIST = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                  "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "OMP_NUM_THREADS",
                  "DCASR_LOG_DIR", "DCASR_METRICS_DIR",
                  "SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_PROCID", "SLURM_NTASKS")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _resolve(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _safe(label: str, fn):
    """Run a collector; on any failure record the error instead of crashing the run."""
    try:
        return fn()
    except Exception as e:                                    # provenance must never raise
        logger.warning("provenance: %s collector failed: %r", label, e)
        return {"error": repr(e)}


# ── git ───────────────────────────────────────────────────────────────────────
def _run_git(args: Sequence[str], repo_root: Path, timeout: float = 15.0) -> tuple[bool, str]:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                             text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return False, repr(e)
    return (out.returncode == 0), (out.stdout if out.returncode == 0 else out.stderr).strip()


def git_info(repo_root: str | Path | None = None, *, max_diff_chars: int = 200_000) -> dict:
    """Commit / branch / dirty flag / changed files / diff-vs-HEAD (truncated)."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    ok, _ = _run_git(["rev-parse", "--is-inside-work-tree"], root)
    if not ok:
        return {"available": False, "repo_root": str(root)}
    info: dict[str, Any] = {"available": True, "repo_root": str(root)}
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                      ("commit_subject", ["log", "-1", "--pretty=%s"]),
                      ("commit_date", ["log", "-1", "--pretty=%cI"])):
        good, val = _run_git(args, root)
        info[key] = val if good else None
    good, porcelain = _run_git(["status", "--porcelain"], root)
    changed = [ln for ln in porcelain.splitlines() if ln.strip()] if good else []
    info["dirty"] = bool(changed)
    info["changed_files"] = changed[:1000]
    good, stat = _run_git(["diff", "--stat", "HEAD"], root)
    info["diffstat"] = stat if good else None
    good, diff = _run_git(["diff", "HEAD"], root)
    if good and diff:
        info["diff"] = diff[:max_diff_chars]
        info["diff_truncated"] = len(diff) > max_diff_chars
    return info


# ── environment ───────────────────────────────────────────────────────────────
def _installed_packages() -> dict[str, str]:
    import importlib.metadata as im
    pkgs: dict[str, str] = {}
    for dist in im.distributions():
        try:
            name = dist.metadata["Name"]
            if name:
                pkgs[name] = dist.version
        except Exception:
            continue
    return dict(sorted(pkgs.items(), key=lambda kv: kv[0].lower()))


def _gpu_info() -> Any:
    import torch
    if not torch.cuda.is_available():
        return None
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({"name": p.name, "capability": f"{p.major}.{p.minor}",
                     "total_mem_gb": round(p.total_memory / 1e9, 3)})
    return gpus


def env_info() -> dict:
    import torch
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": _gpu_info(),
        "packages": _installed_packages(),
    }


# ── determinism / seed ────────────────────────────────────────────────────────
def determinism_info(seed: int | None = None) -> dict:
    import torch
    return {
        "seed": seed,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "note": _KERNEL_NOTE,
    }


# ── batching / effective global batch ─────────────────────────────────────────
def batch_info(cfg: Mapping[str, Any], world_size: int = 1) -> dict:
    """Effective global batch (frames) = per-GPU frame budget · accum_grad · world_size.

    Batching is dynamic (length-bucketed under a frame budget), so the 'global batch'
    is a frame budget rather than a fixed utterance count — held constant across GPU
    counts by the accum_grad / world_size design.
    """
    bins = cfg.get("batch_bins", cfg.get("max_frames"))
    accum = int(cfg.get("accum_grad", 1) or 1)
    ws = int(world_size)
    eff = int(bins) * accum * ws if bins is not None else None
    return {
        "batch_type": cfg.get("batch_type"),
        "per_gpu_frame_budget": bins,
        "accum_grad": accum,
        "world_size": ws,
        "effective_global_batch_frames": eff,
        "note": "effective_global_batch_frames = per_gpu_frame_budget * accum_grad * world_size",
    }


# ── file fingerprints ─────────────────────────────────────────────────────────
def fingerprint_file(path: str | Path, *, count_lines: bool | None = None) -> dict:
    """sha256 + size + mtime for one file (streamed); n_lines for .jsonl. Never raises."""
    p = Path(path)
    info: dict[str, Any] = {"path": str(path)}
    try:
        info["resolved"] = str(p.resolve())
        if not p.exists():
            info["exists"] = False
            return info
        info["exists"] = True
        st = p.stat()
        info["size_bytes"] = st.st_size
        info["modified"] = _iso(st.st_mtime)
        if count_lines is None:
            count_lines = p.suffix == ".jsonl"
        h, nl = hashlib.sha256(), 0
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                if count_lines:
                    nl += chunk.count(b"\n")
        info["sha256"] = h.hexdigest()
        if count_lines:
            info["n_lines"] = nl
    except OSError as e:
        info["error"] = str(e)
    return info


# ── resolved config ───────────────────────────────────────────────────────────
def resolved_config(cfg: Any) -> Any:
    """Resolve an OmegaConf config (interpolations expanded) to a plain, serializable value."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(cfg, (DictConfig, ListConfig)):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        logger.warning("provenance: OmegaConf resolve failed: %r", e)
    import json
    return json.loads(json.dumps(cfg, default=str))          # force plain + serializable


# ── data fingerprints ─────────────────────────────────────────────────────────
def data_info(cfg: Mapping[str, Any], repo_root: Path, *,
              manifests: Sequence[str | Path] | None = None,
              extra_files: Mapping[str, str | Path] | None = None) -> dict:
    """Fingerprint the tokenizer model, CMVN stats, and the manifests the run consumes."""
    info: dict[str, Any] = {}
    bpemodel = cfg.get("bpemodel")
    if bpemodel:
        info["tokenizer"] = fingerprint_file(_resolve(bpemodel, repo_root))
    cmvn = (cfg.get("frontend_conf", {}) or {}).get("cmvn")
    if cmvn:
        info["cmvn"] = fingerprint_file(_resolve(cmvn, repo_root))

    man: dict[str, Any] = {}
    if manifests is not None:
        for m in manifests:
            man[Path(m).name] = fingerprint_file(_resolve(m, repo_root))
    else:                                                    # derive from cfg.data (build_manifests naming)
        data = cfg.get("data", {}) or {}
        mdir = data.get("manifests_dir", "manifests")
        names = ["train-960", *data.get("dev_splits", []), *data.get("test_splits", [])]
        for name in names:
            man[name] = fingerprint_file(_resolve(f"{mdir}/{name}.jsonl", repo_root))
    if man:
        info["manifests"] = man
    if extra_files:
        info["extra"] = {k: fingerprint_file(_resolve(v, repo_root)) for k, v in extra_files.items()}
    return info


# ── top-level ─────────────────────────────────────────────────────────────────
def collect_provenance(cfg: Any, *, repo_root: str | Path | None = None, world_size: int = 1,
                       seed: int | None = None, manifests: Sequence[str | Path] | None = None,
                       extra_files: Mapping[str, str | Path] | None = None,
                       extra: Mapping[str, Any] | None = None) -> dict:
    """Assemble the full provenance record for one run (see module docstring)."""
    root = Path(repo_root) if repo_root is not None else _repo_root()
    plain_cfg = resolved_config(cfg)
    if seed is None and isinstance(plain_cfg, Mapping):      # default seed from the config
        seed = (plain_cfg.get("experiment", {}) or {}).get("seed")
    now = time.time()
    prov = {
        "generated_at": {"epoch": now, "utc": _iso(now)},
        "config": plain_cfg,
        "git": _safe("git", lambda: git_info(root)),
        "env": _safe("env", env_info),
        "determinism": _safe("determinism", lambda: determinism_info(seed)),
        "batch": _safe("batch", lambda: batch_info(plain_cfg if isinstance(plain_cfg, Mapping) else cfg, world_size)),
        "data": _safe("data", lambda: data_info(plain_cfg if isinstance(plain_cfg, Mapping) else cfg,
                                                root, manifests=manifests, extra_files=extra_files)),
        "process": {"argv": list(sys.argv), "cwd": os.getcwd(), "pid": os.getpid(),
                    "env_vars": {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}},
    }
    if extra:
        prov.update(dict(extra))
    logger.info("provenance collected: commit=%s dirty=%s world=%d seed=%s",
                (prov["git"] or {}).get("commit"), (prov["git"] or {}).get("dirty"),
                world_size, seed)
    return prov
