"""Unit tests for the LibriSpeech data pipeline (src/dcasr/data/librispeech.py).

CPU-only: synthetic audio + a tiny tokenizer. Covers manifests, the dataset/collate
contract, and the DDP bucketed batch sampler's invariants (coverage, disjoint shards,
equal batches/rank, determinism, frame budget).
"""
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from dcasr.data.features import LogMelFrontend, SpecAugment
from dcasr.data.librispeech import (
    DistributedBucketBatchSampler, LibriSpeechDataset, apply_speed_perturb, build_manifest,
    collate_batch, feat_frames, load_manifest,
)
from dcasr.data.tokenizer import Tokenizer

WORDS = "THE QUICK BROWN FOX JUMPS OVER A LAZY DOG RUNS HOME MUSIC SILENCE RIVER LIGHT".split()


def _corpus(n=400, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 10))) for _ in range(n)]


def _noise(n):
    return (np.random.default_rng(n).standard_normal(n) * 0.1).astype("float32")


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("tok") / "sp"
    return Tokenizer.train(_corpus(), prefix, vocab_size=100, hard_vocab_limit=False)


# ── feat_frames + manifest ──────────────────────────────────────────────────
def test_feat_frames_matches_frontend():
    assert feat_frames(16000) == 98 and feat_frames(8000) == 48
    assert feat_frames(399) == 0                       # shorter than one window


def test_build_and_load_manifest(tmp_path):
    ch = tmp_path / "LS" / "train-clean-100" / "103" / "1240"
    ch.mkdir(parents=True)
    sf.write(str(ch / "103-1240-0000.flac"), _noise(8000), 16000)
    sf.write(str(ch / "103-1240-0001.flac"), _noise(12000), 16000)
    (ch / "103-1240.trans.txt").write_text(
        "103-1240-0000 HELLO WORLD\n103-1240-0001 FOO BAR BAZ\n")
    out = build_manifest(tmp_path / "LS", ["train-clean-100"], tmp_path / "m.jsonl")
    ents = load_manifest(out)
    assert len(ents) == 2
    assert ents[0]["text"] == "HELLO WORLD" and ents[0]["frames"] == 8000
    assert ents[0]["audio"].endswith("103-1240-0000.flac")


# ── dataset + collate ───────────────────────────────────────────────────────
@pytest.fixture
def dataset(tmp_path, tok):
    entries, texts, ns = [], ["THE QUICK BROWN FOX", "A LAZY DOG RUNS HOME", "MUSIC SILENCE"], [8000, 16000, 12000]
    for i, (n, txt) in enumerate(zip(ns, texts)):
        p = tmp_path / f"u{i}.flac"
        sf.write(str(p), _noise(n), 16000)
        entries.append({"id": f"u{i}", "audio": str(p), "text": txt, "frames": n})
    return LibriSpeechDataset(entries, LogMelFrontend(), tok)


def test_getitem_shapes_and_tokens(dataset, tok):
    s = dataset[0]
    assert s["feats"].shape == (feat_frames(8000), 80)
    assert s["feats"].shape[0] == dataset.lengths[0]
    assert torch.equal(s["tokens"], torch.tensor(tok.encode("THE QUICK BROWN FOX")))
    assert s["id"] == "u0"


def test_collate_pads(dataset, tok):
    batch = collate_batch([dataset[0], dataset[1], dataset[2]], pad_id=tok.pad_id)
    Tmax = max(dataset.lengths)
    assert batch["feats"].shape == (3, Tmax, 80)
    assert batch["feat_lens"].tolist() == dataset.lengths
    assert batch["tokens"].shape[0] == 3
    # padded token positions are pad_id
    for row, ln in zip(batch["tokens"], batch["token_lens"]):
        assert torch.all(row[ln:] == tok.pad_id)
    assert batch["ids"] == ["u0", "u1", "u2"]


# ── speed perturbation (train-only 3x data) ─────────────────────────────────
@pytest.fixture
def sp_entries(tmp_path):
    texts = ["THE QUICK BROWN FOX", "A LAZY DOG RUNS HOME", "MUSIC SILENCE RIVER"]
    ns = [24000, 40000, 32000]                             # long enough to survive /1.1
    entries = []
    for i, (n, txt) in enumerate(zip(ns, texts)):
        p = tmp_path / f"s{i}.flac"
        sf.write(str(p), _noise(n), 16000)
        entries.append({"id": f"s{i}", "audio": str(p), "text": txt, "frames": n})
    return entries


def test_apply_speed_perturb_identity_and_length():
    w = torch.randn(1, 16000)
    assert torch.equal(apply_speed_perturb(w, 16000, 1.0), w)     # factor 1.0 = bit-exact identity
    fast = apply_speed_perturb(w, 16000, 1.1)
    slow = apply_speed_perturb(w, 16000, 0.9)
    assert fast.shape[-1] < 16000 < slow.shape[-1]               # faster = shorter, slower = longer
    assert torch.isfinite(fast).all() and torch.isfinite(slow).all()


def test_dataset_speed_perturb_expands_x3(sp_entries, tok):
    sp = LibriSpeechDataset(sp_entries, LogMelFrontend(), tok, augment=True,
                            speed_perturb=[0.9, 1.0, 1.1])
    assert len(sp) == 9 and sp.factors == [0.9, 1.0, 1.1]        # 3 utts x 3 factors
    base = LibriSpeechDataset(sp_entries, LogMelFrontend(), tok, augment=True)
    assert len(base) == 3 and base.factors == [1.0]             # no speed_perturb -> identity only
    dev = LibriSpeechDataset(sp_entries, LogMelFrontend(), tok, augment=False,
                             speed_perturb=[0.9, 1.0, 1.1])
    assert len(dev) == 3 and dev.factors == [1.0]              # augment=False ignores speed_perturb


def test_dataset_speed_perturb_lengths_and_ids(sp_entries, tok):
    sp = LibriSpeechDataset(sp_entries, LogMelFrontend(), tok, augment=True,
                            speed_perturb=[0.9, 1.0, 1.1])
    n0 = sp_entries[0]["frames"]
    assert sp.lengths[1] == feat_frames(n0)                     # item 1 = (utt0, factor 1.0)
    assert sp.lengths[0] > sp.lengths[1] > sp.lengths[2]        # 0.9 longer, 1.0, 1.1 shorter
    slow, ident, fast = sp[0], sp[1], sp[2]
    assert ident["id"] == "s0" and slow["id"] == "s0#sp0.9" and fast["id"] == "s0#sp1.1"
    assert ident["feats"].shape[0] == feat_frames(n0)          # identity item: exact base frames
    for k in (0, 2):                                            # perturbed items ~match the length estimate
        assert abs(sp[k]["feats"].shape[0] - sp.lengths[k]) <= 2
    assert torch.equal(ident["tokens"], slow["tokens"])        # transcript unchanged by speed


def test_dataset_speed_perturb_deterministic(sp_entries, tok):
    sp = LibriSpeechDataset(sp_entries, LogMelFrontend(), tok, augment=True,
                            specaugment=SpecAugment(), speed_perturb=[0.9, 1.0, 1.1], seed=3)
    sp.set_epoch(2)
    assert torch.equal(sp[4]["feats"], sp[4]["feats"])         # (seed,epoch,index)-deterministic


# ── DDP bucketed batch sampler ──────────────────────────────────────────────
LENGTHS = [50 + (i * 37) % 400 for i in range(200)]       # varied, deterministic


def test_sampler_world1_full_coverage():
    s = DistributedBucketBatchSampler(LENGTHS, max_frames=2000, num_replicas=1, rank=0, shuffle=False)
    idx = [i for b in s for i in b]
    assert sorted(idx) == list(range(len(LENGTHS)))       # every index exactly once


def test_sampler_respects_frame_budget():
    s = DistributedBucketBatchSampler(LENGTHS, max_frames=2000, num_replicas=1, rank=0, shuffle=False)
    for b in s:
        cost = len(b) * max(LENGTHS[i] for i in b)
        assert cost <= 2000 or len(b) == 1                # singletons may exceed the budget


def test_sampler_ddp_disjoint_and_equal():
    R, mf, seed = 4, 2000, 7
    samplers = [DistributedBucketBatchSampler(LENGTHS, mf, R, r, shuffle=True, seed=seed) for r in range(R)]
    assert len({len(s) for s in samplers}) == 1           # equal #batches per rank (no DDP deadlock)
    per_rank = [[i for b in s for i in b] for s in samplers]
    flat = [i for pr in per_rank for i in pr]
    assert len(flat) == len(set(flat))                    # no index shared across ranks
    assert set(flat).issubset(range(len(LENGTHS)))        # union within the dataset


def test_sampler_deterministic_and_epoch_changes_order():
    a = DistributedBucketBatchSampler(LENGTHS, 2000, 1, 0, shuffle=True, seed=0)
    b = DistributedBucketBatchSampler(LENGTHS, 2000, 1, 0, shuffle=True, seed=0)
    assert list(a) == list(b)                             # same seed/epoch reproduces
    first = list(a)
    a.set_epoch(1)
    assert list(a) != first                               # new epoch reshuffles order
