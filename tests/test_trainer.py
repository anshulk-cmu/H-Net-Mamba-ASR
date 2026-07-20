"""Unit tests for the config-driven Trainer (src/dcasr/training/trainer.py).

CPU-only: a tiny fake model with the (loss, stats) contract exercises the loop mechanics,
config-driven optim/scheduler, multi-dev-split validation, resume, keep-N-best, and the
early-stop AND logic — no Mamba/GPU needed.
"""
import pytest
import torch
import torch.nn as nn

from dcasr.metrics_logger import MetricsLogger
from dcasr.training.trainer import Trainer

FEAT = 4


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(FEAT, 8)

    def forward(self, feats, feat_lens, targets, target_lens):
        loss = self.lin(feats).pow(2).mean()
        z = torch.zeros(())
        stats = {"loss/total": loss.detach(), "loss/ctc": loss.detach(),
                 "loss/aed": z, "loss/ratio": z, "kept_fraction": torch.tensor(1.0)}
        return loss, stats


class _Sampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, e):
        self.epochs.append(e)


def _batch(B=2, T=5, U=3, V=6):
    return {"feats": torch.randn(B, T, FEAT), "feat_lens": torch.full((B,), T),
            "tokens": torch.randint(0, V, (B, U)), "token_lens": torch.full((B,), U),
            "ids": [f"u{i}" for i in range(B)]}


def _loader(n=4):
    return [_batch() for _ in range(n)]


def _cfg(**over):
    cfg = {"optim": "adamw", "optim_conf": {"lr": 0.05}, "scheduler": "warmuplr",
           "scheduler_conf": {"warmup_steps": 3}, "max_epoch": 2, "grad_clip": 1.0,
           "grad_clip_type": 2.0, "accum_grad": 1, "precision": "fp32", "log_interval": 1,
           "valid_interval_epoch": 1, "keep_nbest_models": 5,
           "best_model_criterion": [["valid", "loss", "min"]], "early_stopping": {"enable": False}}
    cfg.update(over)
    return cfg


def _trainer(tmp, cfg, dev=None, metrics=None):
    return Trainer(_Model(), _loader(), cfg, dev_loaders=dev, train_sampler=_Sampler(),
                   metrics=metrics, device="cpu", ckpt_dir=tmp / "ckpts")


def test_train_config_driven(tmp_path):
    tr = _trainer(tmp_path, _cfg(), metrics=MetricsLogger("run", root=tmp_path))
    assert isinstance(tr.optimizer, torch.optim.AdamW)
    assert tr.scheduler.base_lrs[0] == 0.05           # configured peak; warmuplr scales live lr
    w0 = tr.raw_model.lin.weight.detach().clone()
    tr.train()
    assert tr.global_step == 8                       # 4 batches * 2 epochs, accum 1
    assert not torch.equal(w0, tr.raw_model.lin.weight)
    assert tr.train_sampler.epochs == [0, 1]
    assert (tmp_path / "run" / "metrics.jsonl").exists()


def test_grad_accum(tmp_path):
    tr = _trainer(tmp_path, _cfg(accum_grad=2, max_epoch=1))
    tr.train()
    assert tr.global_step == 2                        # 4 batches / accum 2


def test_lr_schedule_advances(tmp_path):
    tr = _trainer(tmp_path, _cfg(max_epoch=1))
    tr.train()
    assert tr.scheduler is not None
    assert tr.scheduler.get_last_lr()[0] != 0.05      # warmuplr moved the LR off base


def test_validate_multi_dev_split(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=1),
                  dev={"dev-clean": _loader(2), "dev-other": _loader(2)}, metrics=ml)
    per_split = tr.validate()
    assert set(per_split) == {"dev-clean", "dev-other"}
    assert all("loss" in v for v in per_split.values())
    assert ("valid", "loss") in tr.metric_history     # aggregate recorded
    ml.close()
    keys = {r_key for r_key in _read_keys(tr, tmp_path)}
    assert "dev_dev-clean/loss" in keys and "dev_dev-other/loss" in keys


def _read_keys(tr, tmp_path):
    import json
    with open(tmp_path / "run" / "metrics.jsonl") as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


def test_checkpoint_resume(tmp_path):
    tr = _trainer(tmp_path, _cfg(max_epoch=1))
    tr.train()
    step, params = tr.global_step, [p.detach().clone() for p in tr.raw_model.parameters()]
    tr2 = _trainer(tmp_path, _cfg(max_epoch=1))
    tr2.load_checkpoint(tr2._resolve_resume("auto"))
    assert tr2.global_step == step and tr2.epoch == 1          # epoch complete -> next epoch
    assert tr2.metric_history == tr.metric_history
    for a, b in zip(params, tr2.raw_model.parameters()):
        assert torch.equal(a, b)


def test_best_symlink_and_prune(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=3, keep_nbest_models=2),
                  dev={"dev-clean": _loader(2)}, metrics=ml)
    tr.train()
    assert (tr.ckpt_dir / "valid.loss.best.pt").exists()       # best symlink written
    assert len(list(tr.ckpt_dir.glob("epoch*.pt"))) <= 2 + 1   # top-2 + latest epoch


def test_early_stop_and_logic(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    tr.early_stopping = {"enable": True, "require_all": True, "criteria": [
        {"phase": "valid", "metric": "wer", "mode": "min", "patience": 2},
        {"phase": "train", "metric": "loss", "mode": "min", "patience": 2}]}
    tr.metric_history = {("valid", "wer"): {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0},
                         ("train", "loss"): {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0}}
    tr.epoch = 3
    assert tr._should_early_stop() is True                     # both stalled > patience 2
    tr.metric_history[("valid", "wer")][3] = 1.0               # valid wer improves at ep 3
    assert tr._should_early_stop() is False                    # AND: one not stalled -> continue
    tr.early_stopping["require_all"] = False
    assert tr._should_early_stop() is True                     # OR: train loss still stalled


def test_best_epoch_min_max(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    tr.metric_history = {("valid", "wer"): {0: 9.0, 1: 4.0, 2: 7.0}}
    assert tr._best_epoch("valid", "wer", "min") == 1
    assert tr._best_epoch("valid", "wer", "max") == 0
    assert tr._best_epoch("valid", "nope", "min") is None


# ── Phase-0 fix regressions (2026-07-18 sweep) ───────────────────────────────
def test_device_type_normalized_cpu(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    assert tr.device_type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda:N device strings need CUDA")
def test_indexed_cuda_device_runs_and_logs_gpu_mem(tmp_path):
    """torchrun passes device='cuda:N'; scaler/autocast/gpu-mem must key on the TYPE."""
    ml = MetricsLogger("run", root=tmp_path)
    tr = Trainer(_Model(), _loader(), _cfg(max_epoch=1, precision="bf16"), metrics=ml,
                 device="cuda:0", ckpt_dir=tmp_path / "c")
    assert tr.device_type == "cuda"
    tr.train()                                       # bf16 autocast under an indexed device
    ml.close()
    assert "sys/gpu_mem_gb" in _read_keys(tr, tmp_path)


def test_keep_all_checkpoints_survive_prune(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=3, keep_nbest_models=1, keep_all_checkpoints=True),
                  dev={"dev-clean": _loader(2)}, metrics=ml)
    tr.train()
    assert len(list(tr.ckpt_dir.glob("epoch*.pt"))) == 3       # nothing pruned (H4 retention)


def test_resume_missing_explicit_path_raises(tmp_path):
    tr = _trainer(tmp_path, _cfg())
    with pytest.raises(FileNotFoundError):
        tr._resolve_resume(str(tmp_path / "nope.pt"))
    assert tr._resolve_resume("auto") is None                  # auto+empty = legit fresh start


def test_max_steps_exit_saves_checkpoint(tmp_path):
    tr = _trainer(tmp_path, _cfg(max_epoch=5, valid_interval_epoch=10, max_steps=2))
    tr.train()
    ck = torch.load(tr.ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["global_step"] == 2                              # non-boundary exit still saved


def test_monitor_values_reach_metrics(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=1), dev={"dev-clean": _loader(2)}, metrics=ml)
    tr.train()
    ml.close()
    keys = _read_keys(tr, tmp_path)
    assert "train/loss" in keys and "valid/loss" in keys       # selection-driving monitors


def test_accum_window_mean_logged(tmp_path):
    import json
    torch.manual_seed(0)
    loader = [_batch() for _ in range(2)]
    ml = MetricsLogger("run", root=tmp_path)
    tr = Trainer(_Model(), loader, _cfg(max_epoch=1, accum_grad=2, log_interval=1),
                 metrics=ml, device="cpu", ckpt_dir=tmp_path / "ck")
    with torch.no_grad():                            # both micros see the pre-step weights
        expect = sum(float(tr.raw_model(b["feats"], b["feat_lens"], b["tokens"],
                                        b["token_lens"])[0]) for b in loader) / 2
    tr.train()
    ml.close()
    with open(tmp_path / "run" / "metrics.jsonl") as f:
        logged = [json.loads(l)["value"] for l in f if '"loss/total"' in l]
    assert logged and abs(logged[0] - expect) < 1e-6           # window MEAN, not last micro


def test_ave_metadata_lists_only_existing(tmp_path):
    tr = _trainer(tmp_path, _cfg(keep_nbest_models=2))
    tr.metric_history = {("valid", "loss"): {0: 1.0, 1: 0.5}}
    tr.epoch = 1
    tr.save_checkpoint()                                       # only epoch0001.pt exists
    tr._average_nbest()
    ave = torch.load(tr.ckpt_dir / "valid.loss.ave.pt", map_location="cpu", weights_only=False)
    assert ave["averaged_epochs"] == [1]                       # epoch 0 was never on disk


def test_best_symlink_targets_existing_file(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    tr = _trainer(tmp_path, _cfg(max_epoch=1), dev={"dev-clean": _loader(2)}, metrics=ml)
    tr.train()
    link = tr.ckpt_dir / "valid.loss.best.pt"
    assert link.is_symlink() and link.resolve().exists()       # created after the save


def test_resume_after_completed_max_steps_does_not_overshoot(tmp_path):
    """Relaunching a finished max_steps run must not train an extra step (F19 finding)."""
    tr = _trainer(tmp_path, _cfg(max_epoch=5, valid_interval_epoch=10, max_steps=2))
    tr.train()
    assert tr.global_step == 2
    tr2 = _trainer(tmp_path, _cfg(max_epoch=5, valid_interval_epoch=10, max_steps=2))
    tr2.train(resume="auto")
    assert tr2.global_step == 2                          # untouched: budget already reached
    ck = torch.load(tr2.ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert ck["global_step"] == 2                        # latest.pt not overwritten past budget


# ── OOM safeguards (DDP-safe group skip; valid-loop skip) + valid/cer ─────────
class _OOMModel(_Model):
    """Raises CUDA-OOM on selected forward calls; records every call's shape."""

    def __init__(self, boom_calls=()):
        super().__init__()
        self.boom_calls = set(boom_calls)
        self.calls = []

    def forward(self, feats, feat_lens, targets, target_lens):
        n = len(self.calls)
        self.calls.append(tuple(feats.shape))
        if n in self.boom_calls:
            raise torch.cuda.OutOfMemoryError("synthetic OOM")
        return super().forward(feats, feat_lens, targets, target_lens)


def _oom_trainer(tmp, cfg, model, dev=None, tokenizer=None):
    return Trainer(model, _loader(), cfg, dev_loaders=dev, train_sampler=_Sampler(),
                   tokenizer=tokenizer, device="cpu", ckpt_dir=tmp / "ckpts")


def test_oom_forward_skip_single_gpu(tmp_path):
    tr = _oom_trainer(tmp_path, _cfg(max_epoch=1), _OOMModel(boom_calls={1}))
    tr.train()
    assert tr.oom_skips == 1
    assert tr.global_step == 3                        # 4 batches - 1 skipped


def test_oom_backward_skip_single_gpu(tmp_path):
    tr = _oom_trainer(tmp_path, _cfg(max_epoch=1), _OOMModel())

    class _Bomb:
        def __init__(self, t, boom):
            self.t, self.boom = t, boom

        def backward(self):
            if self.boom:
                raise torch.cuda.OutOfMemoryError("synthetic backward OOM")
            self.t.backward()

    n = {"i": 0}
    real_scale = tr.scaler.scale
    tr.scaler.scale = lambda t: _Bomb(t, n.__setitem__("i", n["i"] + 1) or n["i"] == 2)
    tr.train()
    assert tr.oom_skips == 1 and tr.global_step == 3
    tr.scaler.scale = real_scale


def test_oom_group_skip_ddp_all_ranks_drop_window(tmp_path):
    """world_size>1 semantics without dist: when ANY rank flags OOM, this rank
    completes its real backward (collective parity) then drops the window."""
    model = _OOMModel()
    tr = _oom_trainer(tmp_path, _cfg(max_epoch=1), model)
    tr.world_size = 2                                  # after init: model stays raw
    tr._reduce = lambda vals: [float(v) for v in vals]
    forced = {"left": 1}
    tr._any_rank_oom = lambda local: (local or
                                      bool(forced and forced.pop("left", None)))
    tr.train()
    assert tr.global_step == 3                         # peer's OOM dropped one window
    assert tr.oom_skips == 0                           # this rank itself never OOMed


def test_oom_recovery_step_joins_collectives(tmp_path):
    """world_size>1 + local OOM: the rank re-runs a minimal (B=1, T<=32) slice so
    DDP backward collectives stay matched, then the group skip discards it."""
    model = _OOMModel(boom_calls={2})
    tr = _oom_trainer(tmp_path, _cfg(max_epoch=1), model)
    tr.world_size = 2
    tr._reduce = lambda vals: [float(v) for v in vals]
    tr._any_rank_oom = lambda local: local
    tr.train()
    assert tr.oom_skips == 1 and tr.global_step == 3
    recovery = model.calls[3]                          # call after the boom
    assert recovery[0] == 1 and recovery[1] <= 32      # minimal slice, not the batch


def test_oom_backward_ddp_raises(tmp_path):
    tr = _oom_trainer(tmp_path, _cfg(max_epoch=1), _OOMModel())
    tr.world_size = 2
    tr._any_rank_oom = lambda local: local

    class _Bomb:
        def __init__(self):
            pass

        def backward(self):
            raise torch.cuda.OutOfMemoryError("synthetic backward OOM")

    tr.scaler.scale = lambda t: _Bomb()
    with pytest.raises(RuntimeError, match="DDP backward"):
        tr.train()


class _Tok:
    pad_id = 0

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class _GreedyModel(_OOMModel):
    def greedy_decode(self, feats, feat_lens):
        return [[1, 2, 3]] * feats.shape[0]


def test_validate_oom_skip_and_cer_aggregate(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    model = _GreedyModel(boom_calls={1})               # second dev batch OOMs
    tr = Trainer(model, _loader(), _cfg(max_epoch=1),
                 dev_loaders={"dev-clean": _loader(2), "dev-other": _loader(2)},
                 tokenizer=_Tok(), metrics=ml, device="cpu", ckpt_dir=tmp_path / "c")
    per_split = tr.validate()
    assert per_split["dev-clean"]["oom_skips"] == 1    # skipped, split still scored
    assert "oom_skips" not in per_split["dev-other"]
    assert ("valid", "cer") in tr.metric_history       # cer aggregate now recorded
    assert ("valid", "wer") in tr.metric_history
    ml.close()
    keys = _read_keys(tr, tmp_path)
    assert {"valid/cer", "dev_dev-clean/cer", "dev_dev-clean/oom_skips"} <= keys


def test_validate_all_oom_split_raises(tmp_path):
    """A split whose every batch OOM-skips must fail loudly — 0.0 metrics would
    silently become the permanent best epoch."""
    model = _GreedyModel(boom_calls={0, 1})
    tr = Trainer(model, _loader(), _cfg(max_epoch=1),
                 dev_loaders={"dev-clean": _loader(2)}, tokenizer=_Tok(),
                 device="cpu", ckpt_dir=tmp_path / "c")
    with pytest.raises(RuntimeError, match="every batch was OOM-skipped"):
        tr.validate()


class _RouterModel(_Model):
    """Fake model exposing a router-named parameter (matches encoder naming)."""

    def __init__(self):
        super().__init__()
        self.router = nn.Module()
        self.router.W_q = nn.Linear(4, 4, bias=False)
        self.router.W_k = nn.Linear(4, 4, bias=False)


def test_weight_decay_param_groups(tmp_path):
    """wd>0 splits into decay (>=2-D weights) + no-decay (biases, norms, and any
    param flagged `_no_weight_decay`, e.g. Mamba A_log/D/dt_bias). Verified fix
    for the uniform-decay bug (runlog 2026-07-20)."""
    m = _Model()
    # tag one param no-decay to emulate an SSM A_log/D/dt_bias
    flag = nn.Parameter(torch.zeros(4))
    flag._no_weight_decay = True
    m.register_parameter("fake_A_log", flag)
    cfg = _cfg(optim_conf={"lr": 0.05, "weight_decay": 0.01})
    tr = Trainer(m, _loader(), cfg, train_sampler=_Sampler(), device="cpu",
                 ckpt_dir=tmp_path / "wd")
    assert len(tr.optimizer.param_groups) == 2
    decay, no_decay = tr.optimizer.param_groups
    assert decay["weight_decay"] == pytest.approx(0.01)
    assert no_decay["weight_decay"] == 0.0
    # lin.weight (2-D) decays; lin.bias (1-D) + fake_A_log (flagged) do not
    decay_ids = {id(p) for p in decay["params"]}
    nod_ids = {id(p) for p in no_decay["params"]}
    assert id(m.lin.weight) in decay_ids
    assert id(m.lin.bias) in nod_ids and id(flag) in nod_ids


def test_router_param_group_wiring(tmp_path):
    """optim_conf.router_lr_mult/router_eps split the router into its own damped
    Adam group (the N=2 divergence fix); absent keys keep one group."""
    cfg = _cfg(optim_conf={"lr": 0.05, "router_lr_mult": 0.5, "router_eps": 1e-5})
    tr = Trainer(_RouterModel(), _loader(), cfg, train_sampler=_Sampler(),
                 device="cpu", ckpt_dir=tmp_path / "a")
    assert len(tr.optimizer.param_groups) == 2
    assert tr.scheduler.base_lrs == pytest.approx([0.05, 0.025])   # router at 0.5x
    assert tr.optimizer.param_groups[1]["eps"] == pytest.approx(1e-5)
    assert tr.optimizer.param_groups[0]["eps"] == pytest.approx(1e-8)
    assert len(tr.optimizer.param_groups[1]["params"]) == 2   # W_q + W_k only
    tr.train()                                     # trains + schedules both groups
    g0, g1 = tr.optimizer.param_groups
    assert g1["lr"] == pytest.approx(g0["lr"] * 0.5)           # ratio survives schedule
    tr2 = Trainer(_RouterModel(), _loader(), _cfg(), train_sampler=_Sampler(),
                  device="cpu", ckpt_dir=tmp_path / "b")
    assert len(tr2.optimizer.param_groups) == 1    # no keys -> single group
    tr3 = Trainer(_Model(), _loader(), cfg, train_sampler=_Sampler(),
                  device="cpu", ckpt_dir=tmp_path / "c")
    assert len(tr3.optimizer.param_groups) == 1    # keys but no router params
