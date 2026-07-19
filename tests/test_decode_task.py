"""Unit tests for the decode task (src/dcasr/tasks/decode_task.py). CPU-only:
real CTC/AED heads + real beams on a fake (linear) encoder — the full decode
matrix runs without CUDA; scripts/decode.py's model build is GPU-smoke territory."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dcasr.data.tokenizer import Tokenizer
from dcasr.decoders.aed import AEDHead
from dcasr.decoders.ctc import CTCHead
from dcasr.tasks.decode_task import (audio_seconds_from_manifest, check_heads,
                                     decode_split, expand_cells, load_lm_scorer,
                                     load_model_weights)
from dcasr.tasks.lm_task import LMModel, build_lm

WORDS = ("THE QUICK BROWN FOX JUMPS OVER A LAZY DOG AND THEN RUNS HOME WITH "
         "FRIENDS WHO SPEAK SOFTLY ABOUT RIVERS MOUNTAINS SILENCE").split()


def _lines(n=300, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 9))) for _ in range(n)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("dtok") / "sp"
    return Tokenizer.train(_lines(), prefix, vocab_size=100, hard_vocab_limit=False)


# ── cells ────────────────────────────────────────────────────────────────────
def test_expand_cells_full_matrix():
    dc = {"read_outs": ["ctc", "aed", "joint"], "search": ["greedy", "beam"],
          "lm": "shallow_fusion"}
    names = [c["name"] for c in expand_cells(dc)]
    assert names == ["ctc_greedy", "ctc_beam", "ctc_beam_lm",
                     "aed_beam", "aed_beam_lm", "joint_beam", "joint_beam_lm"]


def test_expand_cells_no_lm_and_string_search():
    assert [c["name"] for c in expand_cells({"read_outs": ["ctc"], "search": "greedy",
                                             "lm": "none"})] == ["ctc_greedy"]
    names = [c["name"] for c in expand_cells({"read_outs": ["ctc", "joint"],
                                              "search": ["greedy", "beam"], "lm": "none"})]
    assert names == ["ctc_greedy", "ctc_beam", "joint_beam"]     # greedy is CTC-only


def test_expand_cells_unknown_raise():
    with pytest.raises(ValueError):
        expand_cells({"read_outs": ["rnnt"]})
    with pytest.raises(ValueError):
        expand_cells({"read_outs": ["ctc"], "search": ["viterbi"]})


def test_check_heads_missing_head_raises():
    m = SimpleNamespace(ctc_head=object(), aed_head=None)
    check_heads(m, expand_cells({"read_outs": ["ctc"], "search": "greedy", "lm": "none"}))
    with pytest.raises(ValueError):
        check_heads(m, expand_cells({"read_outs": ["aed"], "search": ["beam"], "lm": "none"}))
    with pytest.raises(ValueError):
        check_heads(m, expand_cells({"read_outs": ["joint"], "search": ["beam"], "lm": "none"}))


# ── checkpoint loading ───────────────────────────────────────────────────────
def test_load_model_weights_both_formats(tmp_path):
    src, dst = nn.Linear(4, 4), nn.Linear(4, 4)
    torch.save({"model": src.state_dict(), "epoch": 7, "global_step": 123,
                "optimizer": {}}, tmp_path / "full.pt")
    meta = load_model_weights(dst, tmp_path / "full.pt")
    assert meta == {"epoch": 7, "global_step": 123}
    assert torch.equal(dst.weight, src.weight)
    torch.save({"model": src.state_dict(), "averaged_epochs": [3, 5]}, tmp_path / "ave.pt")
    meta = load_model_weights(nn.Linear(4, 4), tmp_path / "ave.pt")
    assert meta == {"averaged_epochs": [3, 5]}


def test_load_lm_scorer_and_vocab_guard(tmp_path, tok):
    from omegaconf import OmegaConf
    lm_cfg = {"lm_conf": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                          "dropout": 0.0}}
    OmegaConf.save(OmegaConf.create(lm_cfg), tmp_path / "lm.yaml")
    good = LMModel(build_lm(lm_cfg, tok.vocab_size))
    torch.save({"model": good.state_dict(), "averaged_epochs": [0]}, tmp_path / "lm.pt")
    dc = {"lm_config": "lm.yaml", "lm_checkpoint": "lm.pt"}
    scorer = load_lm_scorer(dc, tmp_path, tok, torch.device("cpu"))
    lp = scorer.next_logprobs([[5, 6]], torch.device("cpu"))
    assert lp.shape == (1, tok.vocab_size) and torch.isfinite(lp).all()
    bad = LMModel(build_lm(lm_cfg, tok.vocab_size + 7))          # wrong-vocab checkpoint
    torch.save({"model": bad.state_dict()}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):
        load_lm_scorer({"lm_config": "lm.yaml", "lm_checkpoint": "bad.pt"},
                       tmp_path, tok, torch.device("cpu"))


# ── the full matrix on a fake encoder + real heads ───────────────────────────
class _FakeEncoder(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.proj = nn.Linear(80, d)

    def forward(self, feats, feat_lens):
        return SimpleNamespace(features=self.proj(feats), lengths=feat_lens)


class _FakeModel(nn.Module):
    def __init__(self, V, d=16):
        super().__init__()
        self.encoder = _FakeEncoder(d)
        self.ctc_head = CTCHead(d, V)
        self.aed_head = AEDHead(V, d, n_layers=2, n_heads=4, d_ff=32, dropout=0.0,
                                bos_id=1, eos_id=2, pad_id=3)


def _batch(tok, B=3, T=24):
    torch.manual_seed(0)
    texts = _lines(B, seed=4)
    toks = [torch.tensor(tok.encode(t)) for t in texts]
    U = max(len(t) for t in toks)
    tokens = torch.full((B, U), tok.pad_id, dtype=torch.long)
    for i, t in enumerate(toks):
        tokens[i, : len(t)] = t
    return {"feats": torch.randn(B, T, 80), "feat_lens": torch.full((B,), T),
            "tokens": tokens, "token_lens": torch.tensor([len(t) for t in toks]),
            "ids": [f"u{i}" for i in range(B)]}, texts


@pytest.fixture(scope="module")
def lm_scorer(tok):
    from dcasr.decoders.lm_fusion import CausalLMScorer
    return CausalLMScorer(build_lm({"lm_conf": {"d_model": 32, "n_layers": 1,
                                                "n_heads": 2, "d_ff": 64,
                                                "dropout": 0.0}}, tok.vocab_size).eval())


def test_decode_split_all_cells(tmp_path, tok, lm_scorer):
    torch.manual_seed(0)
    model = _FakeModel(tok.vocab_size).eval()
    batch, texts = _batch(tok)
    dc = {"beam_size": 3, "pre_beam": 8, "ctc_weight": 0.3, "lm_weight": 0.3,
          "length_bonus": 0.0}
    audio = {f"u{i}": 1.5 for i in range(3)}
    cells = expand_cells({"read_outs": ["ctc", "aed", "joint"],
                          "search": ["greedy", "beam"], "lm": "shallow_fusion"})
    for cell in cells:
        out = tmp_path / cell["name"] / "dev.jsonl"
        s = decode_split(model, tok, [batch], cell, dc, torch.device("cpu"),
                         audio_seconds=audio, out_path=out, lm=lm_scorer)
        recs = [json.loads(l) for l in open(out)]
        assert [r["id"] for r in recs] == ["u0", "u1", "u2"]
        assert s["n_utts"] == 3 and s["audio_s"] == 4.5 and s["rtf"] > 0
        for r, ref_text in zip(recs, texts):
            assert r["ref"] == ref_text                          # refs decode exactly
            assert isinstance(r["hyp"], str) and r["decode_s"] > 0
            assert "<" not in r["hyp"]                           # no special-token leakage


def test_ctc_greedy_cell_matches_head_reference(tmp_path, tok):
    torch.manual_seed(1)
    model = _FakeModel(tok.vocab_size).eval()
    batch, _ = _batch(tok)
    cell = expand_cells({"read_outs": ["ctc"], "search": "greedy", "lm": "none"})[0]
    out = tmp_path / "g" / "dev.jsonl"
    decode_split(model, tok, [batch], cell, {}, torch.device("cpu"),
                 audio_seconds={}, out_path=out)
    with torch.no_grad():
        enc = model.encoder(batch["feats"], batch["feat_lens"])
        expect = [tok.decode(h) for h in
                  model.ctc_head.greedy_decode(enc.features, enc.lengths)]
    got = [json.loads(l)["hyp"] for l in open(out)]
    assert got == expect                                         # the cell IS the head's greedy


def test_lm_cell_without_scorer_raises(tmp_path, tok):
    model = _FakeModel(tok.vocab_size).eval()
    batch, _ = _batch(tok)
    cell = {"read_out": "ctc", "search": "beam", "lm": True, "name": "ctc_beam_lm"}
    with pytest.raises(ValueError):
        decode_split(model, tok, [batch], cell, {"lm_weight": 0.3}, torch.device("cpu"),
                     audio_seconds={}, out_path=tmp_path / "x.jsonl", lm=None)


def test_audio_seconds_from_manifest(tmp_path):
    with open(tmp_path / "m.jsonl", "w") as f:
        f.write(json.dumps({"id": "a", "audio": "/x", "text": "T", "frames": 32000}) + "\n")
        f.write(json.dumps({"id": "b", "audio": "/y", "text": "T", "frames": 8000}) + "\n")
    assert audio_seconds_from_manifest(tmp_path / "m.jsonl") == {"a": 2.0, "b": 0.5}


# ── fixes from the decode adversarial verification (wf_cbfba9a8) ─────────────
def test_expand_cells_zero_cells_raises():
    with pytest.raises(ValueError):                      # aed/joint have no greedy cell
        expand_cells({"read_outs": ["aed", "joint"], "search": "greedy", "lm": "none"})


def test_expand_cells_null_false_lm_means_off():
    for off in (None, False, "none", "False", ""):
        names = [c["name"] for c in expand_cells({"read_outs": ["ctc"],
                                                  "search": ["beam"], "lm": off})]
        assert names == ["ctc_beam"], (off, names)


def test_expand_cells_bare_string_and_duplicates():
    assert [c["name"] for c in expand_cells({"read_outs": "aed", "search": "beam",
                                             "lm": "none"})] == ["aed_beam"]
    names = [c["name"] for c in expand_cells({"read_outs": ["ctc", "ctc"],
                                              "search": ["beam", "beam"], "lm": "none"})]
    assert names == ["ctc_beam"]                         # deduped, decoded once


def test_lm_scorer_missing_keys_clear_error(tmp_path, tok):
    with pytest.raises(ValueError, match="lm_config"):
        load_lm_scorer({"lm": "shallow_fusion"}, tmp_path, tok, torch.device("cpu"))


def test_ctc_beam_projection_time_included(tmp_path, tok):
    """The shared log_probs projection must be inside the timed accounting."""
    model = _FakeModel(tok.vocab_size).eval()
    batch, _ = _batch(tok)
    cell = expand_cells({"read_outs": ["ctc"], "search": ["beam"], "lm": "none"})[0]
    out = tmp_path / "b" / "dev.jsonl"
    s = decode_split(model, tok, [batch], cell, {"beam_size": 2, "pre_beam": 5},
                     torch.device("cpu"), audio_seconds={}, out_path=out)
    recs = [json.loads(l) for l in open(out)]
    assert s["decode_s"] > 0 and all(r["decode_s"] > 0 for r in recs)
    assert abs(sum(r["decode_s"] for r in recs) - s["decode_s"]) < 1e-3  # summary is 3-dp rounded
