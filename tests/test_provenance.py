"""Tests for run provenance collection (src/dcasr/provenance.py). CPU-only, no GPU gate."""
import hashlib
import json
import subprocess

import torch
import torch.nn as nn

from dcasr.metrics_logger import MetricsLogger
from dcasr.provenance import (batch_info, collect_provenance, data_info, determinism_info,
                              env_info, fingerprint_file, git_info, resolved_config)
from dcasr.training.trainer import Trainer


# ── fingerprints ──────────────────────────────────────────────────────────────
def test_fingerprint_file_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    data = b"hello dcasr\n" * 1000
    p.write_bytes(data)
    fp = fingerprint_file(p)
    assert fp["exists"] is True
    assert fp["size_bytes"] == len(data)
    assert fp["sha256"] == hashlib.sha256(data).hexdigest()
    assert "modified" in fp and "n_lines" not in fp        # .bin => no line count


def test_fingerprint_jsonl_counts_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n{"a":3}\n')
    fp = fingerprint_file(p)
    assert fp["n_lines"] == 3
    assert fp["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()


def test_fingerprint_missing_file(tmp_path):
    fp = fingerprint_file(tmp_path / "nope.model")
    assert fp["exists"] is False and "sha256" not in fp    # graceful, no raise


# ── git ───────────────────────────────────────────────────────────────────────
def test_git_info_non_repo(tmp_path):
    info = git_info(tmp_path)
    assert info["available"] is False


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_git_info_temp_repo(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    (tmp_path / "f.txt").write_text("one\n")
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-qm", "init"], tmp_path)

    info = git_info(tmp_path)
    assert info["available"] is True
    assert len(info["commit"]) == 40 and all(c in "0123456789abcdef" for c in info["commit"])
    assert isinstance(info["branch"], str) and info["branch"]
    assert info["dirty"] is False and info["changed_files"] == []

    (tmp_path / "f.txt").write_text("one\ntwo\n")            # now dirty
    info2 = git_info(tmp_path)
    assert info2["commit"] == info["commit"]                 # same commit, working tree changed
    assert info2["dirty"] is True and info2["changed_files"]
    assert "two" in (info2["diff"] or "")


def test_git_diff_truncation(tmp_path):
    _git(["init", "-q"], tmp_path)
    _git(["config", "user.email", "t@t"], tmp_path)
    _git(["config", "user.name", "t"], tmp_path)
    (tmp_path / "f.txt").write_text("x\n")
    _git(["add", "-A"], tmp_path)
    _git(["commit", "-qm", "init"], tmp_path)
    (tmp_path / "f.txt").write_text("y\n" * 100000)
    info = git_info(tmp_path, max_diff_chars=500)
    assert len(info["diff"]) == 500 and info["diff_truncated"] is True


# ── batch / effective global batch ───────────────────────────────────────────
def test_batch_info_effective_global_batch():
    cfg = {"batch_type": "length", "batch_bins": 24000, "accum_grad": 2}
    b = batch_info(cfg, world_size=4)
    assert b["per_gpu_frame_budget"] == 24000 and b["accum_grad"] == 2 and b["world_size"] == 4
    assert b["effective_global_batch_frames"] == 24000 * 2 * 4


def test_batch_info_missing_bins():
    b = batch_info({"accum_grad": 1}, world_size=1)
    assert b["effective_global_batch_frames"] is None       # no budget declared -> None, no crash


# ── env / determinism ────────────────────────────────────────────────────────
def test_env_info_has_core_keys():
    e = env_info()
    for k in ("python", "platform", "hostname", "torch", "torch_cuda", "packages"):
        assert k in e
    assert e["torch"] == torch.__version__
    assert isinstance(e["packages"], dict) and "torch" in e["packages"]


def test_determinism_info():
    d = determinism_info(seed=7)
    assert d["seed"] == 7
    for k in ("cudnn_deterministic", "cudnn_benchmark", "deterministic_algorithms"):
        assert isinstance(d[k], bool)
    assert "non-deterministic" in d["note"]                 # records the known Mamba-kernel fact


# ── resolved config ──────────────────────────────────────────────────────────
def test_resolved_config_omegaconf_interpolation():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"a": 3, "b": "${a}", "nest": {"c": [1, 2]}})
    out = resolved_config(cfg)
    assert isinstance(out, dict) and out["b"] == 3          # interpolation resolved to a plain value
    assert out["nest"]["c"] == [1, 2]


def test_resolved_config_plain_dict_is_serializable():
    out = resolved_config({"x": 1, "t": ("a", "b")})
    assert json.dumps(out)                                  # round-trips, no exception
    assert out["x"] == 1


# ── data fingerprints ────────────────────────────────────────────────────────
def test_data_info_fingerprints_artifacts(tmp_path):
    (tmp_path / "data" / "tokenizer").mkdir(parents=True)
    (tmp_path / "features").mkdir()
    (tmp_path / "manifests").mkdir()
    (tmp_path / "data" / "tokenizer" / "spm.model").write_bytes(b"MODEL")
    (tmp_path / "features" / "cmvn.pt").write_bytes(b"CMVN")
    (tmp_path / "manifests" / "train-960.jsonl").write_text('{"id":1}\n')
    (tmp_path / "manifests" / "dev-clean.jsonl").write_text('{"id":2}\n')
    cfg = {"bpemodel": "data/tokenizer/spm.model",
           "frontend_conf": {"cmvn": "features/cmvn.pt"},
           "data": {"manifests_dir": "manifests", "dev_splits": ["dev-clean"], "test_splits": []}}
    info = data_info(cfg, tmp_path)
    assert info["tokenizer"]["exists"] and info["cmvn"]["exists"]
    assert info["manifests"]["train-960"]["exists"] and info["manifests"]["train-960"]["n_lines"] == 1
    assert info["manifests"]["dev-clean"]["exists"]


# ── top-level collect ────────────────────────────────────────────────────────
def _min_cfg():
    return {"experiment": {"seed": 11}, "batch_type": "length", "batch_bins": 1000,
            "accum_grad": 1, "bpemodel": "data/tokenizer/spm_bpe_500.model",
            "frontend_conf": {"cmvn": "features/cmvn_train960.pt"},
            "data": {"manifests_dir": "manifests", "dev_splits": [], "test_splits": []}}


def test_collect_provenance_all_sections_and_serializable(tmp_path):
    prov = collect_provenance(_min_cfg(), repo_root=tmp_path, world_size=2)
    for section in ("generated_at", "config", "git", "env", "determinism", "batch", "data", "process"):
        assert section in prov
    assert prov["batch"]["world_size"] == 2
    assert prov["determinism"]["seed"] == 11               # defaulted from config experiment.seed
    assert json.dumps(prov, default=str)                    # whole record persists to summary.json


def test_collect_provenance_seed_override(tmp_path):
    prov = collect_provenance(_min_cfg(), repo_root=tmp_path, world_size=1, seed=99)
    assert prov["determinism"]["seed"] == 99
    assert prov["process"]["argv"] and "cwd" in prov["process"]


def test_collect_provenance_extra_merged(tmp_path):
    prov = collect_provenance(_min_cfg(), repo_root=tmp_path, extra={"run_name": "r1", "log_file": "/x.log"})
    assert prov["run_name"] == "r1" and prov["log_file"] == "/x.log"


def test_collect_provenance_never_raises_on_bad_repo(tmp_path):
    prov = collect_provenance(_min_cfg(), repo_root=tmp_path / "does_not_exist")
    assert prov["git"]["available"] is False                # non-repo -> recorded, not raised


# ── MetricsLogger.append_summary + trainer wiring ────────────────────────────
def test_append_summary_accumulates_across_resume(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.append_summary("provenance", {"commit": "a"})
    ml.close()
    ml2 = MetricsLogger("run", root=tmp_path, resume=True)
    ml2.append_summary("provenance", {"commit": "b"})
    ml2.close()
    summary = json.loads(ml2.summary_path.read_text())
    assert [p["commit"] for p in summary["provenance"]] == ["a", "b"]


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, feats, feat_lens, targets, target_lens):
        loss = self.lin(feats).pow(2).mean()
        return loss, {"loss/total": loss.detach()}


def test_trainer_writes_provenance_at_start(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    prov = {"commit": "deadbeef", "git": {"dirty": False}}
    tr = Trainer(_FakeModel(), [], {"max_epoch": 0, "precision": "fp32"},
                 metrics=ml, device="cpu", ckpt_dir=tmp_path / "ck", provenance=prov)
    tr.train()
    summary = json.loads(ml.summary_path.read_text())
    assert summary["provenance"] == [prov]                  # persisted before the (empty) epoch loop
