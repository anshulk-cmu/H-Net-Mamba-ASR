"""Tests for the structured metrics logger (TensorBoard + JSONL). CPU-only, no GPU gate."""
import json
import math

import torch

from dcasr.metrics_logger import MetricsLogger, default_metrics_dir


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_scalar_writes_jsonl_and_tb(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalar("train/loss", torch.tensor(1.5), 0, split="train", epoch=0)
    ml.close()
    recs = _read_jsonl(ml.jsonl_path)
    assert len(recs) == 1
    r = recs[0]
    assert r["key"] == "train/loss" and r["value"] == 1.5
    assert r["step"] == 0 and r["split"] == "train" and r["epoch"] == 0
    assert "wall_time" in r
    events = list((ml.run_dir / "tb").glob("events.out.tfevents.*"))
    assert events, "no TensorBoard event file written"


def test_values_coerced_to_float(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalar("a", 3, 0)                 # python int
    ml.log_scalar("b", torch.tensor(2.5), 1)  # 0-dim tensor
    ml.close()
    vals = {r["key"]: r["value"] for r in _read_jsonl(ml.jsonl_path)}
    assert vals == {"a": 3.0, "b": 2.5}
    assert all(isinstance(v, float) for v in vals.values())


def test_non_main_rank_is_noop(tmp_path):
    ml = MetricsLogger("run", root=tmp_path, rank=1)
    assert ml.is_main is False and ml.run_dir is None
    ml.log_scalar("x", 1.0, 0)               # must not raise or create anything
    ml.log_scalars({"y": 2.0}, 0)
    ml.log_histogram("h", torch.rand(4), 0)
    ml.update_summary(k=1)
    ml.close()
    assert not any(tmp_path.iterdir())


def test_resume_appends_not_truncates(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalar("a", 1.0, 0)
    ml.log_scalar("a", 2.0, 1)
    ml.close()
    ml2 = MetricsLogger("run", root=tmp_path, resume=True)
    ml2.log_scalar("a", 3.0, 2)
    ml2.close()
    assert len(_read_jsonl(ml2.jsonl_path)) == 3
    # a fresh (non-resume) logger truncates back to one record
    ml3 = MetricsLogger("run", root=tmp_path)
    ml3.log_scalar("a", 9.0, 0)
    ml3.close()
    assert len(_read_jsonl(ml3.jsonl_path)) == 1


def test_log_scalars_multiple(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalars({"loss_ctc": 1.0, "loss_ratio": 0.5, "lr": 1e-3}, 5, split="train")
    ml.close()
    recs = _read_jsonl(ml.jsonl_path)
    assert len(recs) == 3
    assert all(r["step"] == 5 and r["split"] == "train" for r in recs)
    assert {r["key"] for r in recs} == {"loss_ctc", "loss_ratio", "lr"}


def test_update_summary_atomic_and_valid(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.update_summary(param_count=123, seed=1)
    ml.update_summary(best_wer=5.6)          # merges, not overwrites
    ml.close()
    summary = json.loads(ml.summary_path.read_text())
    assert summary == {"param_count": 123, "seed": 1, "best_wer": 5.6}


def test_histogram_writes_summary_stats(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_histogram("interp/p", torch.linspace(0.0, 1.0, 11), 0)
    ml.close()
    keys = {r["key"]: r["value"] for r in _read_jsonl(ml.jsonl_path)}
    assert keys["interp/p/min"] == 0.0 and keys["interp/p/max"] == 1.0
    assert abs(keys["interp/p/mean"] - 0.5) < 1e-6
    assert "interp/p/std" in keys
    assert list((ml.run_dir / "tb").glob("events.out.tfevents.*"))


def test_context_manager_closes(tmp_path):
    with MetricsLogger("run", root=tmp_path) as ml:
        ml.log_scalar("a", 1.0, 0)
        path = ml.jsonl_path
    assert len(_read_jsonl(path)) == 1       # readable => file was flushed/closed


def test_default_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DCASR_METRICS_DIR", str(tmp_path))
    assert default_metrics_dir() == tmp_path
    ml = MetricsLogger("run", root=None)
    ml.log_scalar("a", 1.0, 0)
    ml.close()
    assert (tmp_path / "run" / "metrics.jsonl").exists()


def test_non_finite_value_roundtrips(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalar("train/loss", float("nan"), 0)
    ml.log_scalar("train/grad_norm", float("inf"), 1)
    ml.close()
    recs = _read_jsonl(ml.jsonl_path)
    assert math.isnan(recs[0]["value"]) and math.isinf(recs[1]["value"])


def test_histogram_empty_tensor_is_noop(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_histogram("h", torch.empty(0), 0)         # TB make_histogram crashes on empty input
    ml.close()
    assert _read_jsonl(ml.jsonl_path) == []


def test_fresh_run_clears_stale_tb_events(tmp_path):
    ml = MetricsLogger("run", root=tmp_path)
    ml.log_scalar("a", 1.0, 0)
    ml.close()
    old = list((ml.run_dir / "tb").glob("events.out.tfevents.*"))
    assert old
    ml2 = MetricsLogger("run", root=tmp_path)        # fresh run: JSONL truncated AND tb cleared
    ml2.close()
    assert all(not f.exists() for f in old)          # no overlapping curves from the stale run
