"""LibriSpeech-960h dataset, manifests, and DDP-aware bucketed batching.

Wires the frontend + tokenizer into batches. The batch sampler buckets by length
under a per-GPU frame budget and shards across DDP ranks (equal count/rank,
epoch-seeded) — so 1 vs N GPUs is config-only and the global batch = the budget.
"""
from __future__ import annotations

import glob
import json
import os
from functools import partial
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

WIN_LENGTH, HOP_LENGTH = 400, 160
TRAIN_960 = ["train-clean-100", "train-clean-360", "train-other-500"]


def feat_frames(n_samples: int) -> int:
    """Feature-frame count T for an n_samples waveform (features.py contract)."""
    return max(0, 1 + (n_samples - WIN_LENGTH) // HOP_LENGTH)


# ── manifests ────────────────────────────────────────────────────────────────
def build_manifest(librispeech_root, splits, out_path) -> Path:
    """Scan `splits` under `librispeech_root`; write a jsonl manifest of
    {id, audio, text, frames} (frames = raw sample count) and return its path."""
    root, out_path = Path(librispeech_root), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as w:
        for split in splits:
            for tf in sorted(glob.glob(str(root / split / "*" / "*" / "*.trans.txt"))):
                d = os.path.dirname(tf)
                with open(tf) as f:
                    for line in f:
                        uid, _, text = line.strip().partition(" ")
                        if not text:
                            continue
                        audio = f"{d}/{uid}.flac"
                        w.write(json.dumps({"id": uid, "audio": audio, "text": text,
                                            "frames": sf.info(audio).frames}) + "\n")
                        n += 1
                        if n % 50000 == 0:
                            logger.info("manifest: %d utterances scanned", n)
    logger.info("manifest written: %d utterances -> %s", n, out_path)
    return out_path


def load_manifest(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── dataset ──────────────────────────────────────────────────────────────────
class LibriSpeechDataset(Dataset):
    """Yields {feats [T,80], tokens [U], id}. frontend/tokenizer injected; cmvn and
    specaugment optional (specaugment applied only when augment=True)."""

    def __init__(self, manifest, frontend, tokenizer, cmvn=None,
                 specaugment=None, augment=False):
        self.entries = (load_manifest(manifest)
                        if isinstance(manifest, (str, Path)) else list(manifest))
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.cmvn = cmvn
        self.specaugment = specaugment
        self.augment = augment
        self.pad_id = tokenizer.pad_id
        self.lengths = [feat_frames(e["frames"]) for e in self.entries]
        logger.debug("LibriSpeechDataset: %d utts, T in [%d,%d]",
                     len(self.entries), min(self.lengths, default=0),
                     max(self.lengths, default=0))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        wave, _ = sf.read(e["audio"])                        # float64 numpy, [N]
        feats, _ = self.frontend(torch.from_numpy(wave).unsqueeze(0))   # [1, T, 80]
        if self.cmvn is not None:
            feats = self.cmvn(feats)
        if self.augment and self.specaugment is not None:
            self.specaugment.train()
            feats = self.specaugment(feats)
        tokens = torch.tensor(self.tokenizer.encode(e["text"]), dtype=torch.long)
        return {"feats": feats[0], "tokens": tokens, "id": e["id"]}


def collate_batch(samples, pad_id: int = 0) -> dict:
    """Pad a list of dataset items into a batch (feats zero-padded, tokens pad_id-padded)."""
    feats = [s["feats"] for s in samples]
    tokens = [s["tokens"] for s in samples]
    flens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    ulens = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    B, Tmax, D = len(samples), int(flens.max()), feats[0].shape[1]
    Umax = int(ulens.max())
    fb = feats[0].new_zeros(B, Tmax, D)
    tb = tokens[0].new_full((B, Umax), pad_id)
    for i, (f, t) in enumerate(zip(feats, tokens)):
        fb[i, : f.shape[0]] = f
        tb[i, : t.shape[0]] = t
    return {"feats": fb, "feat_lens": flens, "tokens": tb,
            "token_lens": ulens, "ids": [s["id"] for s in samples]}


# ── DDP-aware bucketed batch sampler ─────────────────────────────────────────
class DistributedBucketBatchSampler(Sampler):
    """Length-bucketed dynamic batches under a `max_frames` budget (= max B·T per
    batch, the per-GPU memory knob), sharded across DDP ranks with an equal number
    of batches per rank. Deterministic given (seed, epoch); call set_epoch()."""

    def __init__(self, lengths, max_frames: int, num_replicas: int = 1, rank: int = 0,
                 shuffle: bool = True, seed: int = 0):
        self.lengths = list(lengths)
        self.max_frames = max_frames
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._rank_batches = self._compute()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._rank_batches = self._compute()

    def _all_batches(self) -> list[list[int]]:
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            new_max = max(cur_max, self.lengths[i])
            if cur and (len(cur) + 1) * new_max > self.max_frames:
                batches.append(cur)
                cur, cur_max = [i], self.lengths[i]
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def _compute(self) -> list[list[int]]:
        batches = self._all_batches()
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batches = [batches[k] for k in torch.randperm(len(batches), generator=g).tolist()]
        usable = (len(batches) // self.num_replicas) * self.num_replicas   # equal per rank
        return batches[self.rank:usable:self.num_replicas]

    def __iter__(self):
        return iter(self._rank_batches)

    def __len__(self):
        return len(self._rank_batches)


def make_dataloader(dataset, max_frames: int, augment: bool = False, num_workers: int = 4,
                    seed: int = 0, world_size: int | None = None, rank: int | None = None):
    """Build a DDP-aware DataLoader + its sampler. world_size/rank default to the
    active torch.distributed group, else single-process (1 GPU)."""
    if world_size is None:
        if dist.is_available() and dist.is_initialized():
            world_size, rank = dist.get_world_size(), dist.get_rank()
        else:
            world_size, rank = 1, 0
    sampler = DistributedBucketBatchSampler(
        dataset.lengths, max_frames, world_size, rank, shuffle=augment, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=partial(collate_batch, pad_id=dataset.pad_id),
                        num_workers=num_workers, pin_memory=True)
    return loader, sampler
