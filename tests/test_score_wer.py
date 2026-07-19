"""Unit tests for scoring (src/dcasr/eval/score.py + scripts/score_wer.py). CPU-only:
hand-computed WER/S/D/I expectations, bootstrap statistical properties, gate logic,
and an end-to-end run of the real script over a synthetic decode tree."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from dcasr.eval.metrics import word_error_rate
from dcasr.eval.score import (bootstrap_split, discover_cells, format_report,
                              gate_check, load_decode_records, score_decode_dir,
                              score_records)


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _recs(pairs, decode_s=0.1, audio_s=2.0):
    return [{"id": f"u{i:03d}", "ref": r, "hyp": h, "decode_s": decode_s,
             "audio_s": audio_s} for i, (r, h) in enumerate(pairs)]


# ── loading ──────────────────────────────────────────────────────────────────
def test_load_decode_records_valid_and_errors(tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, _recs([("A B", "A B"), ("C D", "C")]))
    recs = load_decode_records(p)
    assert [r["id"] for r in recs] == ["u000", "u001"]

    _write_jsonl(p, [{"id": "a", "ref": "X"}])                  # missing hyp
    with pytest.raises(ValueError, match="hyp"):
        load_decode_records(p)
    _write_jsonl(p, [{"id": "a", "ref": "X", "hyp": "X"}] * 2)  # duplicate id
    with pytest.raises(ValueError, match="duplicate"):
        load_decode_records(p)
    p.write_text("\n")                                          # empty
    with pytest.raises(ValueError, match="no decode records"):
        load_decode_records(p)


# ── scoring math ─────────────────────────────────────────────────────────────
def test_score_records_hand_computed():
    # refs: 3+3+2 = 8 words; 1 sub + 1 del + 1 ins, exactly one exact match
    recs = _recs([("THE CAT SAT", "THE DOG SAT"),        # 1 sub
                  ("A BIG HOUSE", "A BIG"),              # 1 del
                  ("GOOD DAY", "GOOD FINE DAY")])        # 1 ins
    s = score_records(recs)
    w = s["wer"]
    assert (w.n_ref, w.sub, w.dele, w.ins) == (8, 1, 1, 1)
    assert w.n_utt == 3 and w.n_correct == 0
    assert abs(100 * w.er - 100 * 3 / 8) < 1e-9
    assert [u["n_ref"] for u in s["utts"]] == [3, 3, 2]
    assert [u["sub"] for u in s["utts"]] == [1, 0, 0]
    assert [u["del"] for u in s["utts"]] == [0, 1, 0]
    assert [u["ins"] for u in s["utts"]] == [0, 0, 1]
    assert s["decode_s"] == pytest.approx(0.3) and s["audio_s"] == pytest.approx(6.0)
    assert s["rtf"] == pytest.approx(0.3 / 6.0, abs=1e-5)


def test_score_records_matches_word_error_rate():
    refs = ["THE CAT SAT ON THE MAT", "HELLO WORLD", "A B C D", "SAME TEXT"]
    hyps = ["THE CAT SAT MAT", "HELLO BIG WORLD", "A X C", "SAME TEXT"]
    mine = score_records(_recs(list(zip(refs, hyps))))["wer"]
    ref_stats = word_error_rate(refs, hyps)
    for f in ("n_ref", "sub", "dele", "ins", "cor", "n_utt", "n_correct"):
        assert getattr(mine, f) == getattr(ref_stats, f), f


def test_score_records_cer_and_normalize():
    s = score_records(_recs([("AB CD", "AB CE")]))
    c = s["cer"]
    assert c.n_ref == 4 and c.sub == 1 and c.dele == 0 and c.ins == 0
    # normalization: case-insensitive by default, exact when off
    same = score_records(_recs([("HELLO", "hello")]), normalize=True)["wer"]
    diff = score_records(_recs([("HELLO", "hello")]), normalize=False)["wer"]
    assert same.errors == 0 and diff.errors == 1


def test_empty_hyp_all_deletions():
    w = score_records(_recs([("ONE TWO THREE", "")]))["wer"]
    assert (w.sub, w.dele, w.ins) == (0, 3, 0) and w.er == 1.0


# ── paired bootstrap ─────────────────────────────────────────────────────────
def _utts(errors_per_utt, n_ref=10):
    return [{"id": f"u{i:03d}", "n_ref": n_ref, "sub": e, "del": 0, "ins": 0}
            for i, e in enumerate(errors_per_utt)]


def test_bootstrap_identical_systems():
    utts = _utts([1, 0, 2, 1, 0, 3] * 10)
    out = bootstrap_split({"a": utts, "b": [dict(u) for u in utts]}, n_resamples=200, seed=0)
    pair = out["pairs"][0]
    assert pair["delta"] == 0.0 and pair["delta_ci95"] == [0.0, 0.0]
    assert pair["p_value"] == 1.0                        # every resample delta is 0


def test_bootstrap_detects_clear_difference():
    good = _utts([0, 1] * 50)                            # WER 5%
    bad = _utts([3, 4] * 50)                             # WER 35%
    out = bootstrap_split({"good": good, "bad": bad}, n_resamples=1000, seed=1)
    pair = out["pairs"][0]
    assert pair["a"] == "good" and pair["delta"] == pytest.approx(-30.0)
    assert pair["p_value"] < 0.01                        # sign never flips
    assert pair["delta_ci95"][1] < 0                     # CI excludes 0
    assert out["cells"]["good"]["wer"] == pytest.approx(5.0)
    assert out["cells"]["bad"]["wer"] == pytest.approx(35.0)


def test_bootstrap_deterministic_and_seed_sensitive():
    a, b = _utts([0, 1, 2] * 20), _utts([1, 1, 1] * 20)
    r1 = bootstrap_split({"a": a, "b": b}, n_resamples=300, seed=7)
    r2 = bootstrap_split({"a": a, "b": b}, n_resamples=300, seed=7)
    r3 = bootstrap_split({"a": a, "b": b}, n_resamples=300, seed=8)
    assert r1 == r2
    assert (r1["cells"]["a"]["wer_ci95"] != r3["cells"]["a"]["wer_ci95"]
            or r1["pairs"][0]["p_value"] != r3["pairs"][0]["p_value"])


def test_bootstrap_chunking_invariant():
    a, b = _utts([0, 2, 1] * 30), _utts([1, 0, 1] * 30)
    r1 = bootstrap_split({"a": a, "b": b}, n_resamples=250, seed=3, chunk=250)
    r2 = bootstrap_split({"a": a, "b": b}, n_resamples=250, seed=3, chunk=17)
    assert r1 == r2                                      # chunk size must not change draws


def test_bootstrap_id_mismatch_and_shuffled_order():
    a = _utts([1, 2, 3])
    b = _utts([1, 2, 3])
    shuffled = [b[2], b[0], b[1]]                        # same ids, different order: fine
    out = bootstrap_split({"a": a, "b": shuffled}, n_resamples=50, seed=0)
    assert out["pairs"][0]["delta"] == 0.0
    b_bad = _utts([1, 2])                                # different utterance set
    with pytest.raises(ValueError, match="different utterance"):
        bootstrap_split({"a": a, "b": b_bad}, n_resamples=50, seed=0)


def test_bootstrap_full_wer_matches_point_estimate():
    a, b = _utts([2, 0, 1, 1]), _utts([0, 0, 1, 0])
    out = bootstrap_split({"a": a, "b": b}, n_resamples=50, seed=0)
    assert out["cells"]["a"]["wer"] == pytest.approx(100 * 4 / 40, abs=0.005)
    assert out["pairs"][0]["delta"] == pytest.approx(100 * 3 / 40, abs=0.005)


# ── gate ─────────────────────────────────────────────────────────────────────
def test_gate_check_pass_fail_and_missing():
    cells = {"test-clean": {"ctc_greedy": {"wer": 11.5}, "joint_beam": {"wer": 9.8}}}
    goal = {"sane_test_clean_wer_below": 12.0}
    g = gate_check(cells, goal)
    assert g["evaluated"] and g["passed"] and g["cell"] == "joint_beam"  # best cell
    g = gate_check(cells, goal, gate_cell="ctc_greedy")
    assert g["passed"] and g["wer"] == 11.5
    g = gate_check({"test-clean": {"ctc_greedy": {"wer": 15.0}}}, goal)
    assert g["evaluated"] and not g["passed"]
    g = gate_check({"dev-clean": {"c": {"wer": 5.0}}}, goal)             # split missing
    assert not g["evaluated"] and "not decoded" in g["reason"]
    g = gate_check(cells, goal, gate_cell="aed_beam")                    # cell missing
    assert not g["evaluated"]
    g = gate_check(cells, None)                                          # no goal block
    assert not g["evaluated"]


# ── discovery + end-to-end dir scoring ───────────────────────────────────────
def _fake_decode_dir(tmp_path):
    root = tmp_path / "decode" / "ckpt"
    pairs_clean = [("THE CAT SAT", "THE CAT SAT"), ("A BIG DOG", "A BIG DOG"),
                   ("GOOD DAY SIR", "GOOD DAY SIR"), ("ONE TWO", "ONE TWO")]
    pairs_bad = [(r, "X " + h.split(" ", 1)[1] if " " in h else "X")
                 for r, h in pairs_clean]                # first word substituted everywhere
    _write_jsonl(root / "ctc_greedy" / "test-clean.jsonl", _recs(pairs_bad))
    _write_jsonl(root / "joint_beam" / "test-clean.jsonl", _recs(pairs_clean))
    _write_jsonl(root / "ctc_greedy" / "dev-clean.jsonl", _recs(pairs_clean))
    return root


def test_discover_cells(tmp_path):
    root = _fake_decode_dir(tmp_path)
    tree = discover_cells(root)
    assert set(tree) == {"test-clean", "dev-clean"}
    assert set(tree["test-clean"]) == {"ctc_greedy", "joint_beam"}
    assert set(tree["dev-clean"]) == {"ctc_greedy"}
    with pytest.raises(ValueError, match="no decode outputs"):
        discover_cells(tmp_path)


def test_score_decode_dir_end_to_end(tmp_path):
    root = _fake_decode_dir(tmp_path)
    out = score_decode_dir(root, n_bootstrap=300, seed=0,
                           goal_cfg={"sane_test_clean_wer_below": 12.0})
    tc = out["splits"]["test-clean"]["cells"]
    assert tc["joint_beam"]["wer"] == 0.0 and tc["joint_beam"]["sent_acc"] == 100.0
    assert tc["ctc_greedy"]["wer"] == pytest.approx(100 * 4 / 11, abs=0.005)
    assert tc["ctc_greedy"]["wer_sub"] == tc["ctc_greedy"]["wer"]        # all subs
    pair = out["splits"]["test-clean"]["significance"]["pairs"][0]
    assert {pair["a"], pair["b"]} == {"ctc_greedy", "joint_beam"}
    assert out["gate"]["evaluated"] and out["gate"]["passed"]
    assert out["gate"]["cell"] == "joint_beam" and out["gate"]["wer"] == 0.0
    # per-utt counts persisted, skipping the score/ dir on re-discovery
    per_utt = root / "score" / "ctc_greedy" / "test-clean.jsonl"
    utts = [json.loads(l) for l in open(per_utt)]
    assert len(utts) == 4 and all(u["sub"] == 1 for u in utts)
    assert set(discover_cells(root)) == {"test-clean", "dev-clean"}      # score/ ignored
    # single-cell split gets no pairs but keeps its CI
    dev = out["splits"]["dev-clean"]
    assert dev["significance"]["pairs"] == []
    assert dev["cells"]["ctc_greedy"]["wer_ci95"] == [0.0, 0.0]


def test_score_decode_dir_no_bootstrap(tmp_path):
    root = _fake_decode_dir(tmp_path)
    out = score_decode_dir(root, n_bootstrap=0, seed=0, goal_cfg=None)
    assert "significance" not in out["splits"]["test-clean"]
    assert "wer_ci95" not in out["splits"]["test-clean"]["cells"]["ctc_greedy"]


def test_format_report_contents(tmp_path):
    root = _fake_decode_dir(tmp_path)
    out = score_decode_dir(root, n_bootstrap=200, seed=0,
                           goal_cfg={"sane_test_clean_wer_below": 12.0})
    rep = format_report(out["splits"], out["gate"])
    assert "test-clean" in rep and "ctc_greedy" in rep and "joint_beam" in rep
    assert "GATE PASS" in rep and "vs < 12.00" in rep
    rep2 = format_report(out["splits"], {"evaluated": False, "reason": "x"})
    assert "GATE not evaluated" in rep2


# ── the real script (run() + dir resolution) ─────────────────────────────────
def _load_script():
    spec = importlib.util.spec_from_file_location(
        "score_wer", Path(__file__).resolve().parents[1] / "scripts" / "score_wer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_run_end_to_end(tmp_path):
    from omegaconf import OmegaConf
    sw = _load_script()
    root = _fake_decode_dir(tmp_path)
    cfg = OmegaConf.create({
        "experiment": {"name": "fake_run", "seed": 5},
        "goal": {"sane_test_clean_wer_below": 12.0},
        "score": {"n_bootstrap": 200},
    })
    out = sw.run(cfg, decode_dir=root, repo_root=tmp_path)
    scores = json.loads((root / "score" / "scores.json").read_text())
    assert scores["gate"]["passed"] is True
    assert scores["splits"]["test-clean"]["significance"]["seed"] == 5   # experiment.seed
    assert "provenance" in scores
    prov_files = scores["provenance"]["data"]["extra"]
    assert "ctc_greedy/test-clean" in prov_files and "joint_beam/test-clean" in prov_files
    report = (root / "score" / "report.txt").read_text()
    assert "GATE PASS" in report
    assert out["gate"]["cell"] == "joint_beam"
    # checkpoint-stem resolution matches decode.py's out_root convention
    d = sw.resolve_decode_dir(cfg, checkpoint="checkpoints/fake_run/valid.wer.ave.pt",
                              repo_root=tmp_path)
    assert d == tmp_path / "experiments" / "fake_run" / "decode" / "valid.wer.ave"
    with pytest.raises(ValueError, match="--checkpoint or --decode-dir"):
        sw.resolve_decode_dir(cfg)


def test_script_rerun_overwrites_cleanly(tmp_path):
    from omegaconf import OmegaConf
    sw = _load_script()
    root = _fake_decode_dir(tmp_path)
    cfg = OmegaConf.create({"experiment": {"name": "r", "seed": 0},
                            "score": {"n_bootstrap": 50}})
    r1 = sw.run(cfg, decode_dir=root, repo_root=tmp_path)
    r2 = sw.run(cfg, decode_dir=root, repo_root=tmp_path)   # rerun over existing score/
    assert r1["splits"]["test-clean"]["cells"] == r2["splits"]["test-clean"]["cells"]
    assert r1["gate"] == {"evaluated": False, "reason":
                          "no goal.sane_test_clean_wer_below in config"}


# ── fixes from the adversarial verification (wf_30f44cee) ────────────────────
def test_loader_malformed_json_and_types(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id": "a", "ref": "X", "hyp": "X"}\n{bad json\n')
    with pytest.raises(ValueError, match=r"d\.jsonl:2 malformed JSON"):
        load_decode_records(p)
    p.write_text("42\n")
    with pytest.raises(ValueError, match="not an object"):
        load_decode_records(p)
    p.write_text('{"id": "a", "ref": 7, "hyp": "X"}\n')
    with pytest.raises(ValueError, match="'ref'"):
        load_decode_records(p)


def test_gate_uses_unrounded_wer():
    goal = {"sane_test_clean_wer_below": 12.0}
    cells = {"test-clean": {"c": {"wer": 12.0, "wer_exact": 11.996}}}
    g = gate_check(cells, goal)
    assert g["passed"] and g["wer"] == 11.996            # display-rounded 12.0 would FAIL
    g = gate_check({"test-clean": {"c": {"wer": 12.0}}}, goal)   # fallback: rounded only
    assert g["evaluated"] and not g["passed"]


def test_rtf_suppressed_on_missing_audio():
    recs = _recs([("A B", "A B"), ("C D", "C D")])
    recs[1]["audio_s"] = 0.0                             # partial coverage inflates RTF
    s = score_records(recs)
    assert s["rtf"] is None
    assert score_records(_recs([("A B", "A B")]))["rtf"] is not None


def test_consistency_checked_without_bootstrap(tmp_path):
    root = tmp_path / "dec"
    _write_jsonl(root / "a" / "s.jsonl", _recs([("A B", "A B"), ("C D", "C D")]))
    _write_jsonl(root / "b" / "s.jsonl", _recs([("A B", "A B")]))
    with pytest.raises(ValueError, match="different utterance sets"):
        score_decode_dir(root, n_bootstrap=0)            # loud even with bootstrap off


def test_bootstrap_nref_disagreement_and_zero_ref():
    a, b = _utts([1, 2]), _utts([1, 2], n_ref=20)
    with pytest.raises(ValueError, match="reference length"):
        bootstrap_split({"a": a, "b": b}, n_resamples=10, seed=0)
    z = _utts([1, 2])
    z[0]["n_ref"] = 0
    with pytest.raises(ValueError, match="zero-reference-word"):
        bootstrap_split({"a": z, "b": [dict(u) for u in z]}, n_resamples=10, seed=0)


def test_rerun_prunes_stale_score_cells(tmp_path):
    root = _fake_decode_dir(tmp_path)
    score_decode_dir(root, n_bootstrap=0)
    assert (root / "score" / "joint_beam" / "test-clean.jsonl").exists()
    shutil.rmtree(root / "joint_beam")                   # cell dropped from the decode tree
    score_decode_dir(root, n_bootstrap=0)
    assert not (root / "score" / "joint_beam").exists()  # stale per-utt files pruned


def test_discover_missing_dir_and_junk(tmp_path):
    with pytest.raises(ValueError, match="decode dir not found"):
        discover_cells(tmp_path / "nope")
    root = _fake_decode_dir(tmp_path)
    (root / "ctc_greedy" / "junk.jsonl").mkdir()         # a DIRECTORY named *.jsonl
    assert "junk" not in discover_cells(root)


def test_p_value_formula_pinned_independently():
    """Recompute the exact draw stream + two-sided +1-corrected p from scratch —
    dropping the x2 or the +1 correction would fail this."""
    a = _utts([0, 1, 2, 0, 1] * 8)
    b = _utts([1, 1, 1, 1, 1] * 8)
    n, R, seed = 40, 400, 11
    out = bootstrap_split({"a": a, "b": b}, n_resamples=R, seed=seed, chunk=R)
    Ea = np.array([u["sub"] + u["del"] + u["ins"] for u in a])
    Eb = np.array([u["sub"] + u["del"] + u["ins"] for u in b])
    L = np.array([u["n_ref"] for u in a])
    idx = np.random.default_rng(seed).integers(0, n, size=(R, n))
    ref = L[idx].sum(axis=1)
    deltas = 100.0 * Ea[idx].sum(axis=1) / ref - 100.0 * Eb[idx].sum(axis=1) / ref
    p_le = (np.count_nonzero(deltas <= 0) + 1) / (R + 1)
    p_ge = (np.count_nonzero(deltas >= 0) + 1) / (R + 1)
    assert p_le != p_ge                                  # asymmetric: min() is load-bearing
    expect = min(1.0, 2 * min(p_le, p_ge))
    assert out["pairs"][0]["p_value"] == round(expect, 5)
    assert 0 < expect < 1
