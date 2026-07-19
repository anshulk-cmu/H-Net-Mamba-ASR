"""Unit tests for the external-LM task (src/dcasr/data/lm_text.py, src/dcasr/tasks/lm_task.py,
scripts/train_lm.py). CPU-only — the TransformerLM is pure torch, no Mamba/GPU needed."""
import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from dcasr.data.lm_text import (LMTextDataset, collate_lm, load_line_index,
                                load_token_lengths, make_lm_dataloader)
from dcasr.data.tokenizer import Tokenizer
from dcasr.tasks.lm_task import LMModel, build_lm, build_lm_dataloaders

REPO = Path(__file__).resolve().parents[1]

WORDS = ("THE QUICK BROWN FOX JUMPS OVER A LAZY DOG AND THEN RUNS HOME WITH HIS "
         "FRIEND WHO SPEAKS SOFTLY ABOUT MUSIC RIVERS MOUNTAINS SILENCE").split()


def _lines(n=40, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12))) for _ in range(n)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("lmtok") / "sp"
    return Tokenizer.train(_lines(400), prefix, vocab_size=100, hard_vocab_limit=False)


def _corpus_file(tmp_path, lines):
    p = tmp_path / "corpus.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# ── line index ───────────────────────────────────────────────────────────────
def test_line_index_matches_naive_parse(tmp_path):
    lines = _lines(20) + [""] + _lines(5, seed=1) + ["   "]      # empty/space lines skipped
    p = _corpus_file(tmp_path, lines)
    offsets, words = load_line_index(p)
    kept = [l for l in lines if l.strip()]
    assert len(offsets) == len(kept)
    raw = p.read_bytes()
    for off, w, line in zip(offsets, words, kept):
        got = raw[off:].split(b"\n")[0].decode()
        assert got == line and w == len(line.split())


def test_line_index_cache_reused_and_invalidated(tmp_path):
    p = _corpus_file(tmp_path, _lines(10))
    o1, _ = load_line_index(p)
    o2, _ = load_line_index(p)                                   # served from cache
    assert (o1 == o2).all() and p.with_suffix(".txt.idx.npz").exists()
    p.write_text("\n".join(_lines(7, seed=3)) + "\n", encoding="utf-8")
    o3, _ = load_line_index(p)                                   # file changed -> rebuilt
    assert len(o3) == 7


# ── dataset ──────────────────────────────────────────────────────────────────
def test_lazy_equals_in_memory(tmp_path, tok):
    lines = _lines(15)
    p = _corpus_file(tmp_path, lines)
    lazy = LMTextDataset(tok, corpus_path=p)
    mem = LMTextDataset(tok, lines=lines)
    assert len(lazy) == len(mem) == 15
    for i in range(15):
        assert torch.equal(lazy[i]["tokens"], mem[i]["tokens"])


def test_truncation_and_unk_only_line(tmp_path, tok):
    long_line = " ".join(random.Random(0).choice(WORDS) for _ in range(400))
    ds = LMTextDataset(tok, lines=[long_line, "12345"], max_tokens=16)
    assert len(ds[0]["tokens"]) == 16                            # capped
    assert ds.lengths[0] == 16                                   # estimate capped too
    assert ds[1]["tokens"].numel() >= 1                          # unknown-only never empty


def test_from_manifest(tmp_path, tok):
    m = tmp_path / "dev.jsonl"
    lines = _lines(6, seed=2)
    with open(m, "w") as f:
        for i, t in enumerate(lines):
            f.write(json.dumps({"id": f"u{i}", "audio": "/x.flac", "text": t, "frames": 1}) + "\n")
    ds = LMTextDataset.from_manifest(m, tok)
    assert len(ds) == 6
    assert torch.equal(ds[0]["tokens"], torch.tensor(tok.encode(lines[0])))


def test_collate_contract(tok):
    ds = LMTextDataset(tok, lines=_lines(4))
    batch = collate_lm([ds[i] for i in range(4)], pad_id=tok.pad_id)
    assert set(batch) == {"feats", "feat_lens", "tokens", "token_lens", "ids"}
    assert batch["feats"] is batch["tokens"] and torch.equal(batch["feat_lens"], batch["token_lens"])
    for i in range(4):
        n = int(batch["token_lens"][i])
        assert torch.equal(batch["tokens"][i, :n], ds[i]["tokens"])
        assert (batch["tokens"][i, n:] == tok.pad_id).all()


def test_dataloader_covers_all_and_batches(tmp_path, tok):
    p = _corpus_file(tmp_path, _lines(30))
    ds = LMTextDataset(tok, corpus_path=p)
    loader, sampler = make_lm_dataloader(ds, batch_tokens=200, shuffle=True,
                                         num_workers=0, seed=0)
    seen = [i for b in sampler._rank_batches for i in b]
    assert sorted(seen) == list(range(30))                       # full coverage at world=1
    b = next(iter(loader))
    assert b["tokens"].shape[0] >= 1 and b["tokens"].dtype == torch.long


# ── model / builders ─────────────────────────────────────────────────────────
def test_lm_model_contract(tok):
    lm = build_lm({"lm_conf": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64}},
                  tok.vocab_size)
    assert lm.d_model == 32 and lm.lsm_weight == 0.0             # fusion default: no smoothing
    model = LMModel(lm)
    ds = LMTextDataset(tok, lines=_lines(4))
    batch = collate_lm([ds[i] for i in range(4)], pad_id=tok.pad_id)
    loss, stats = model(batch["feats"], batch["feat_lens"], batch["tokens"], batch["token_lens"])
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert set(stats) == {"loss/total", "lm/token_acc", "batch_weight"}
    assert 0.0 <= float(stats["lm/token_acc"]) <= 1.0
    assert int(stats["batch_weight"]) == int((batch["token_lens"] + 1).sum())
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None) > 0


# ── end-to-end: real train_lm.run() on a tiny corpus (CPU) ───────────────────
def _import_train_lm():
    spec = importlib.util.spec_from_file_location("train_lm", REPO / "scripts" / "train_lm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_repo(tmp_path, tok):
    (tmp_path / "data" / "lm").mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    corpus = tmp_path / "data" / "lm" / "tiny.txt"
    corpus.write_text("\n".join(_lines(24, seed=5)) + "\n", encoding="utf-8")
    with open(tmp_path / "manifests" / "dev-clean.jsonl", "w") as f:
        for i, t in enumerate(_lines(5, seed=6)):
            f.write(json.dumps({"id": f"d{i}", "audio": "/x.flac", "text": t, "frames": 1}) + "\n")
    cfg = OmegaConf.create({
        "experiment": {"name": "lm_tiny", "seed": 1},
        "bpemodel": str(Path(tok.model_path).relative_to(Path(tok.model_path).anchor)),
        "data": {"lm_corpus": "data/lm/tiny.txt", "manifests_dir": "manifests",
                 "dev_splits": ["dev-clean"]},
        "lm_conf": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                    "max_line_tokens": 32},
        "batch_tokens": 256, "num_workers": 0,
        "optim": "adamw", "optim_conf": {"lr": 1e-3}, "scheduler": None,
        "train": {"max_epoch": 1, "precision": "fp32", "log_interval": 1},
        "eval": {"valid_interval_epoch": 1},
        "best_model_criterion": [["valid", "loss", "min"]], "keep_nbest_models": 1,
    })
    cfg.bpemodel = tok.model_path                                # absolute path works too
    return cfg


def test_train_lm_end_to_end_and_resume(tmp_path, tok, monkeypatch):
    monkeypatch.setenv("DCASR_METRICS_DIR", str(tmp_path / "exp"))
    train_lm = _import_train_lm()
    cfg = _tiny_repo(tmp_path, tok)
    tr = train_lm.run(cfg, repo_root=tmp_path)
    assert tr.global_step > 0
    ck = tmp_path / "checkpoints" / "lm_tiny"
    assert (ck / "latest.pt").exists() and (ck / "valid.loss.best.pt").exists()
    recs = [json.loads(l) for l in open(tmp_path / "exp" / "lm_tiny" / "metrics.jsonl")]
    keys = {r["key"] for r in recs}
    assert {"loss/total", "valid/loss", "dev_dev-clean/loss", "batch_weight"} <= keys
    step1 = tr.global_step
    cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"train": {"max_epoch": 2}}))
    tr2 = train_lm.run(cfg2, resume="auto", repo_root=tmp_path)
    assert tr2.epoch == 1 and tr2.global_step > step1            # genuinely resumed epoch 1
    summ = json.loads((tmp_path / "exp" / "lm_tiny" / "summary.json").read_text())
    assert len(summ["provenance"]) == 2                          # appended across resume


def test_loss_return_acc_backward_compatible_and_overfit_acc(tok):
    lm = build_lm({"lm_conf": {"d_model": 32, "n_layers": 1, "n_heads": 2, "d_ff": 64,
                               "dropout": 0.0}}, tok.vocab_size)
    toks = torch.tensor([tok.encode("THE QUICK BROWN FOX")])
    lens = torch.tensor([toks.shape[1]])
    plain = lm.loss(toks, lens)                                  # verified single-tensor path
    with_acc, acc = lm.loss(toks, lens, return_acc=True)
    assert torch.equal(plain, with_acc) and 0.0 <= float(acc) <= 1.0
    opt = torch.optim.Adam(lm.parameters(), lr=5e-3)
    for _ in range(300):
        l, _ = lm.loss(toks, lens, return_acc=True)
        opt.zero_grad(); l.backward(); opt.step()
    _, acc2 = lm.loss(toks, lens, return_acc=True)
    assert float(acc2) == 1.0                                    # memorized -> perfect accuracy


# ── fixes from the #6 adversarial verification (wf_ac4f2316) ─────────────────
def test_corrupt_index_cache_recovers(tmp_path, tok):
    p = _corpus_file(tmp_path, _lines(8))
    load_line_index(p)
    cache = p.with_suffix(".txt.idx.npz")
    cache.write_bytes(b"garbage not an npz")                     # truncated/corrupt cache
    offsets, words = load_line_index(p)                          # must rebuild, not raise
    assert len(offsets) == 8 and cache.exists()


def test_concurrent_cold_cache_index_builds(tmp_path):
    import multiprocessing as mp
    p = _corpus_file(tmp_path, _lines(2000, seed=9))

    def build(path, q):
        from dcasr.data.lm_text import load_line_index as lli
        try:
            off, _ = lli(path)
            q.put(len(off))
        except Exception as e:
            q.put(repr(e))

    q = mp.get_context("fork").Queue()
    procs = [mp.get_context("fork").Process(target=build, args=(p, q)) for _ in range(4)]
    [pr.start() for pr in procs]
    [pr.join() for pr in procs]
    results = [q.get() for _ in range(4)]
    assert results == [2000] * 4, results                        # nobody crashed on the race


def test_dataset_picklable_after_fetch(tmp_path, tok):
    import pickle
    p = _corpus_file(tmp_path, _lines(5))
    ds = LMTextDataset(tok, corpus_path=p)
    _ = ds[0]                                                    # opens a parent file handle
    ds2 = pickle.loads(pickle.dumps(ds))                         # spawn-context requirement
    assert torch.equal(ds2[1]["tokens"], ds[1]["tokens"])


def test_unicode_whitespace_line_mode_parity(tmp_path, tok):
    lines = ["HELLO WORLD", " ", "ANOTHER LINE"]            # NBSP-only middle line
    p = tmp_path / "u.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lazy = LMTextDataset(tok, corpus_path=p)
    mem = LMTextDataset(tok, lines=lines)
    assert len(lazy) == len(mem)                                 # same emptiness semantics
    for i in range(len(mem)):
        assert torch.equal(lazy[i]["tokens"], mem[i]["tokens"])


def test_valid_loss_is_token_weighted(tok):
    """exp(valid/loss) must be TRUE per-token dev perplexity (verification found 26% bias)."""
    from dcasr.training.trainer import Trainer
    model = LMModel(build_lm({"lm_conf": {"d_model": 32, "n_layers": 1, "n_heads": 2,
                                          "d_ff": 64, "dropout": 0.0}}, tok.vocab_size))
    short = collate_lm([{"tokens": torch.tensor(tok.encode("THE DOG")), "id": "a"}], tok.pad_id)
    long_ = collate_lm([{"tokens": torch.tensor(tok.encode(
        "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG AND RUNS HOME WITH HIS FRIEND")),
        "id": "b"}], tok.pad_id)
    cfg = {"optim": "adamw", "optim_conf": {"lr": 1e-3}, "scheduler": None,
           "max_epoch": 1, "precision": "fp32", "valid_interval_epoch": 1}
    tr = Trainer(model, [short], cfg, dev_loaders={"dev": [short, long_]},
                 device="cpu", ckpt_dir=Path("/tmp/nonexistent-unused"))
    tr.validate()
    with torch.no_grad():
        l1, s1 = model(short["feats"], short["feat_lens"], short["tokens"], short["token_lens"])
        l2, s2 = model(long_["feats"], long_["feat_lens"], long_["tokens"], long_["token_lens"])
    w1, w2 = float(s1["batch_weight"]), float(s2["batch_weight"])
    expect = (float(l1) * w1 + float(l2) * w2) / (w1 + w2)       # token-weighted, NOT row mean
    got = tr.metric_history[("valid", "loss")][0]
    assert abs(got - expect) < 1e-6
    assert abs(got - (float(l1) + float(l2)) / 2) > 1e-4         # and it differs from row-mean


def test_exact_lengths_match_items(tmp_path, tok):
    """lengths are EXACT capped token counts (the OOM-proof packing contract)."""
    p = _corpus_file(tmp_path, _lines(25, seed=11))
    ds = LMTextDataset(tok, corpus_path=p, max_tokens=16)
    for i in range(len(ds)):
        assert ds.lengths[i] == len(ds[i]["tokens"])             # exact, not estimated
    mem = LMTextDataset(tok, lines=_lines(25, seed=11), max_tokens=16)
    assert ds.lengths == mem.lengths


def test_token_length_cache_keyed_by_tokenizer(tmp_path, tok, tmp_path_factory):
    p = _corpus_file(tmp_path, _lines(12))
    offsets, _ = load_line_index(p)
    l1 = load_token_lengths(p, tok, offsets)
    assert (p.parent / f"corpus.txt.len.{Path(tok.model_path).stem}.npz").exists()
    l1b = load_token_lengths(p, tok, offsets)                    # cache hit
    assert (l1 == l1b).all()
    other = Tokenizer.train(_lines(300, seed=1), tmp_path_factory.mktemp("t2") / "sp2",
                            vocab_size=60, hard_vocab_limit=False)
    l2 = load_token_lengths(p, other, offsets)                   # different tokenizer -> own cache
    assert (p.parent / "corpus.txt.len.sp2.npz").exists()
    assert not (l1 == l2).all()                                  # different vocab, different counts


def test_batch_token_budget_is_hard_bound(tmp_path, tok):
    """With exact lengths, every collated batch obeys B*U <= batch_tokens."""
    p = _corpus_file(tmp_path, _lines(60, seed=12))
    ds = LMTextDataset(tok, corpus_path=p)
    loader, _ = make_lm_dataloader(ds, batch_tokens=64, shuffle=True, num_workers=0, seed=3)
    for b in loader:
        assert b["tokens"].shape[0] * b["tokens"].shape[1] <= 64


def test_trainer_oom_guard_skips_batch(tmp_path):
    """A cuda-OOM raise inside forward is skipped: grads cleared, training continues."""
    import torch.nn as nn
    from dcasr.training.trainer import Trainer

    class _OOMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.calls = 0

        def forward(self, feats, feat_lens, targets, target_lens):
            self.calls += 1
            if self.calls == 2:                                  # second batch "OOMs"
                raise torch.cuda.OutOfMemoryError("simulated")
            loss = self.lin(feats).pow(2).mean()
            return loss, {"loss/total": loss.detach()}

    batches = [{"feats": torch.randn(2, 3, 4), "feat_lens": torch.full((2,), 3),
                "tokens": torch.zeros(2, 2, dtype=torch.long),
                "token_lens": torch.full((2,), 2), "ids": ["a", "b"]} for _ in range(4)]
    cfg = {"optim": "adamw", "optim_conf": {"lr": 1e-3}, "scheduler": None, "max_epoch": 1,
           "precision": "fp32", "valid_interval_epoch": 10, "log_interval": 1}
    tr = Trainer(_OOMModel(), batches, cfg, device="cpu", ckpt_dir=tmp_path / "ck")
    tr.train()
    assert tr.oom_skips == 1
    assert tr.global_step == 3                                   # 4 batches - 1 skipped
