"""Tests for the run-assembly seam (src/dcasr/tasks/build.py). CPU-only, no Mamba/GPU."""
import json

import torch
from omegaconf import OmegaConf

from dcasr.data.features import LogMelFrontend, SpecAugment
from dcasr.tasks.build import (build_cmvn, build_dataloaders, build_frontend,
                               build_specaugment, flatten_config, resolve_manifests)

REAL_CFG = "configs/typeA_small_N1_ctc.yaml"


# ── flatten_config ────────────────────────────────────────────────────────────
def test_flatten_config_real_yaml():
    cfg = OmegaConf.load(REAL_CFG)
    flat = flatten_config(cfg)
    assert flat["max_epoch"] == 120                    # hoisted from train.max_epoch
    assert flat["valid_interval_epoch"] == 10          # hoisted from eval.valid_interval_epoch
    assert flat["precision"] == "bf16"                 # from train.precision
    assert flat["accum_grad"] == 1 and flat["optim"] == "adamw"
    assert flat["scheduler"] == "warmuplr" and flat["scheduler_conf"]["warmup_steps"] == 10000
    assert [list(c) for c in flat["best_model_criterion"]] == [["valid", "wer", "min"],
                                                               ["valid", "loss", "min"]]
    assert flat["keep_nbest_models"] == 5 and isinstance(flat["early_stopping"], dict)
    assert flat["max_steps"] is None                   # absent in YAML -> None


def test_flatten_config_defaults_and_overrides():
    flat = flatten_config({"train": {"max_epoch": 3, "max_steps": 7}, "accum_grad": 4})
    assert flat["max_epoch"] == 3 and flat["max_steps"] == 7 and flat["accum_grad"] == 4
    assert flat["valid_interval_epoch"] == 10 and flat["optim"] == "adamw"   # defaults when absent


# ── frontend / specaug / cmvn ────────────────────────────────────────────────
def test_build_frontend_from_config():
    fe = build_frontend(OmegaConf.load(REAL_CFG))
    assert isinstance(fe, LogMelFrontend)
    assert fe.n_mels == 80 and fe.hop_length == 160 and fe.win_length == 400


def test_build_specaugment_adaptive_from_real_config():
    sa = build_specaugment(OmegaConf.load(REAL_CFG))
    assert isinstance(sa, SpecAugment)
    assert sa.freq_masks == 2 and sa.freq_width == 27
    assert sa.time_masks == 10 and sa.time_width_ratio == 0.05   # adaptive (ratio) path


def test_build_specaugment_fixed_width_path():
    cfg = {"specaug_conf": {"num_freq_mask": 2, "freq_mask_width_range": [0, 30],
                            "num_time_mask": 2, "time_mask_width_range": [0, 40]}}
    sa = build_specaugment(cfg)
    assert sa.time_width == 40 and sa.time_width_ratio is None and sa.freq_width == 30


def test_build_specaugment_none_when_absent():
    assert build_specaugment({}) is None


def test_build_cmvn_none_and_load(tmp_path):
    assert build_cmvn({}, tmp_path) is None                       # no cmvn key -> None
    stats = {"mean": torch.zeros(80), "std": torch.ones(80), "count": 10}
    torch.save(stats, tmp_path / "cmvn.pt")
    cmvn = build_cmvn({"frontend_conf": {"cmvn": "cmvn.pt"}}, tmp_path)
    assert cmvn is not None
    out = cmvn(torch.ones(1, 3, 80))                              # (x-0)/1 == x
    assert torch.allclose(out, torch.ones(1, 3, 80))


# ── manifests / dataloaders ──────────────────────────────────────────────────
def test_resolve_manifests(tmp_path):
    cfg = {"data": {"manifests_dir": "m", "train_manifest": "train-960",
                    "dev_splits": ["dev-clean", "dev-other"]}}
    train, dev = resolve_manifests(cfg, tmp_path)
    assert train == tmp_path / "m" / "train-960.jsonl"
    assert set(dev) == {"dev-clean", "dev-other"}
    assert dev["dev-clean"] == tmp_path / "m" / "dev-clean.jsonl"


class _FakeTok:
    pad_id = 0


def _write_manifest(path, n, base_frames=8000):
    with open(path, "w") as w:
        for i in range(n):
            w.write(json.dumps({"id": f"u{i}", "audio": f"/nope/{i}.flac",
                                "text": "HELLO", "frames": base_frames + i * 1000}) + "\n")


def test_build_dataloaders_construction(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_manifest(mdir / "train-960.jsonl", 12)
    _write_manifest(mdir / "devx.jsonl", 6)
    cfg = {"batch_bins": 4000, "num_workers": 0,
           "data": {"manifests_dir": str(mdir), "dev_splits": ["devx"]}}
    train_loader, sampler, dev = build_dataloaders(cfg, tmp_path, _FakeTok(), frontend=None,
                                                   world_size=1, rank=0, seed=1)
    assert len(sampler) > 0 and len(sampler) == len(train_loader.batch_sampler)
    assert set(dev) == {"devx"} and len(dev["devx"].batch_sampler) > 0
    assert train_loader.dataset.augment is True and dev["devx"].dataset.augment is False


def test_build_dataloaders_speed_perturb(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_manifest(mdir / "train-960.jsonl", 5)
    cfg = {"batch_bins": 4000, "num_workers": 0, "train": {"speed_perturb": [0.9, 1.0, 1.1]},
           "data": {"manifests_dir": str(mdir), "dev_splits": []}}
    train_loader, _, _ = build_dataloaders(cfg, tmp_path, _FakeTok(), None,
                                           world_size=1, rank=0, seed=0)
    assert len(train_loader.dataset) == 15                       # 5 utts x 3 speed factors
    assert train_loader.dataset.factors == [0.9, 1.0, 1.1]


def test_build_dataloaders_ddp_shards(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_manifest(mdir / "train-960.jsonl", 40)
    cfg = {"batch_bins": 9000, "num_workers": 0,
           "data": {"manifests_dir": str(mdir), "dev_splits": []}}
    _, s1, _ = build_dataloaders(cfg, tmp_path, _FakeTok(), None, world_size=1, rank=0, seed=0)
    _, s2, _ = build_dataloaders(cfg, tmp_path, _FakeTok(), None, world_size=2, rank=0, seed=0)
    assert len(s2) <= len(s1)                                    # rank-0 sees <= half the batches


def test_flatten_config_keep_all_checkpoints():
    assert flatten_config({})["keep_all_checkpoints"] is False
    assert flatten_config({"keep_all_checkpoints": True})["keep_all_checkpoints"] is True


def test_build_dataloaders_dev_not_sharded(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    _write_manifest(mdir / "train-960.jsonl", 40)
    _write_manifest(mdir / "devx.jsonl", 11)
    cfg = {"batch_bins": 9000, "num_workers": 0,
           "data": {"manifests_dir": str(mdir), "dev_splits": ["devx"]}}
    _, _, dev1 = build_dataloaders(cfg, tmp_path, _FakeTok(), None, world_size=1, rank=0, seed=0)
    _, _, dev2 = build_dataloaders(cfg, tmp_path, _FakeTok(), None, world_size=2, rank=1, seed=0)
    # dev is never sharded: every rank sees the FULL split (the trim would drop the longest)
    assert len(dev2["devx"].batch_sampler) == len(dev1["devx"].batch_sampler)
