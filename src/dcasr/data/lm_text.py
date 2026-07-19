"""Text-only dataset for external-LM training (plan #6, the 810M-word LM corpus).

`LMTextDataset` serves tokenized lines either lazily from a large corpus file
(byte-offset index cached beside it — the 40M-line corpus is never held in RAM)
or from an in-memory list (dev sets built from manifest transcripts). Bucketing
uses EXACT per-line token counts (cached per tokenizer), so the sampler's token
budget is a hard per-batch bound (deterministic GPU memory). The collate emits
the Trainer's standard batch dict with feats == tokens, so the model-agnostic
Trainer trains an LM unchanged (tokenizer=None => loss-only validation).
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dcasr.data.librispeech import DistributedBucketBatchSampler
from dcasr.logging_utils import get_logger

logger = get_logger(__name__)


def _build_line_index(corpus_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One pass over the corpus: byte offset + word count per non-empty line."""
    offsets, words = [], []
    pos = 0
    with open(corpus_path, "rb") as f:
        for raw in f:
            if raw.strip():
                offsets.append(pos)
                words.append(len(raw.split()))
            pos += len(raw)
    return np.asarray(offsets, dtype=np.int64), np.asarray(words, dtype=np.int32)


def load_line_index(corpus_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Cached (offsets, word counts) for a corpus file; rebuilt if the file changed."""
    corpus_path = Path(corpus_path)
    cache = corpus_path.with_suffix(corpus_path.suffix + ".idx.npz")
    stat = corpus_path.stat()
    if cache.exists():
        try:
            z = np.load(cache)
            if int(z["size"]) == stat.st_size and int(z["mtime_ns"]) == stat.st_mtime_ns:
                return z["offsets"], z["words"]
            logger.info("corpus changed; rebuilding line index %s", cache.name)
        except Exception:                            # truncated/corrupt cache: rebuild, don't brick
            logger.warning("unreadable line-index cache %s; rebuilding", cache.name)
    offsets, words = _build_line_index(corpus_path)
    tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")   # per-pid: concurrent builders can't race
    np.savez(tmp, offsets=offsets, words=words,
             size=np.int64(stat.st_size), mtime_ns=np.int64(stat.st_mtime_ns))
    os.replace(tmp, cache)
    logger.info("line index: %d non-empty lines -> %s", len(offsets), cache.name)
    return offsets, words


_POOL_TOK = None


def _pool_init(model_path: str) -> None:
    global _POOL_TOK
    import sentencepiece as spm
    _POOL_TOK = spm.SentencePieceProcessor(model_file=model_path)


def _pool_count(lines: list[str]) -> list[int]:
    return [len(_POOL_TOK.encode(l, out_type=int)) for l in lines]


def load_token_lengths(corpus_path: str | Path, tokenizer, offsets: np.ndarray) -> np.ndarray:
    """EXACT per-line token counts (uncapped), cached per (corpus, tokenizer model).

    Exactness makes the sampler's token budget a hard bound (plus only the +1 bos/eos
    wrap), so per-GPU batch memory is deterministic — no estimate-overshoot OOM tail.
    One pooled encode pass over the corpus on first use; rebuilt if either file changes.
    """
    corpus_path = Path(corpus_path)
    stat = corpus_path.stat()
    tok_path = Path(tokenizer.model_path)
    tok_stat = tok_path.stat()
    cache = corpus_path.with_suffix(corpus_path.suffix + f".len.{tok_path.stem}.npz")
    if cache.exists():
        try:
            z = np.load(cache)
            if (int(z["size"]) == stat.st_size and int(z["mtime_ns"]) == stat.st_mtime_ns
                    and int(z["tok_size"]) == tok_stat.st_size
                    and int(z["tok_mtime_ns"]) == tok_stat.st_mtime_ns
                    and len(z["lengths"]) == len(offsets)):
                return z["lengths"]
            logger.info("stale token-length cache %s; rebuilding", cache.name)
        except Exception:
            logger.warning("unreadable token-length cache %s; rebuilding", cache.name)
    import multiprocessing as mp
    t0 = time.time()
    counts = np.empty(len(offsets), dtype=np.int32)
    chunk = 20000
    with open(corpus_path, "rb") as f, \
         mp.get_context("fork").Pool(min(12, os.cpu_count() or 4), _pool_init,
                                     (str(tok_path),)) as pool:
        def chunks():
            buf = []
            for off in offsets:
                f.seek(int(off))
                buf.append(f.readline().decode("utf-8").strip())
                if len(buf) == chunk:
                    yield buf
                    buf = []
            if buf:
                yield buf
        pos = 0
        for res in pool.imap(_pool_count, chunks()):
            counts[pos : pos + len(res)] = res
            pos += len(res)
    tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")
    np.savez(tmp, lengths=counts, size=np.int64(stat.st_size),
             mtime_ns=np.int64(stat.st_mtime_ns), tok_size=np.int64(tok_stat.st_size),
             tok_mtime_ns=np.int64(tok_stat.st_mtime_ns))
    os.replace(tmp, cache)
    logger.info("token lengths: %d lines in %.0f s -> %s", len(counts), time.time() - t0,
                cache.name)
    return counts


class LMTextDataset(Dataset):
    """Tokenized text lines for next-token LM training.

    Lazy mode (`corpus_path`): lines read on demand via the byte-offset index;
    per-worker file handles (fork-safe). In-memory mode (`lines`): small dev sets.
    `lengths` are EXACT (capped) token counts, so the token-budget sampler packs
    deterministically-sized batches.
    """

    def __init__(self, tokenizer, *, corpus_path: str | Path | None = None,
                 lines: list[str] | None = None, max_tokens: int = 512):
        assert (corpus_path is None) != (lines is None), "exactly one source"
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_id
        self.max_tokens = int(max_tokens)
        self._fh: dict[int, object] = {}
        self.corpus_path = Path(corpus_path) if corpus_path is not None else None
        if self.corpus_path is not None:
            self.offsets, _ = load_line_index(self.corpus_path)
            counts = load_token_lengths(self.corpus_path, tokenizer, self.offsets)
            self.lengths = [max(1, min(self.max_tokens, int(c))) for c in counts]
            self.lines = None
        else:                                        # byte-level emptiness == lazy-mode semantics
            self.lines = [l.strip() for l in lines if l.encode("utf-8").strip()]
            self.offsets = None
            self.lengths = [max(1, min(self.max_tokens, len(tokenizer.encode(l))))
                            for l in self.lines]
        logger.info("LMTextDataset: %d lines (%s, max_tokens=%d)", len(self.lengths),
                    self.corpus_path.name if self.corpus_path else "in-memory", self.max_tokens)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, tokenizer, **kw) -> "LMTextDataset":
        """Dev-set variant: the `text` fields of a manifest (dev-clean/dev-other)."""
        with open(manifest_path, encoding="utf-8") as f:
            lines = [json.loads(l)["text"] for l in f if l.strip()]
        return cls(tokenizer, lines=lines, **kw)

    def __getstate__(self) -> dict:                  # open handles aren't picklable (spawn ctx)
        state = dict(self.__dict__)
        state["_fh"] = {}
        return state

    def _line(self, idx: int) -> str:
        if self.lines is not None:
            return self.lines[idx]
        fh = self._fh.get(os.getpid())
        if fh is None:                                   # one handle per worker process
            fh = open(self.corpus_path, "rb")
            self._fh[os.getpid()] = fh
        fh.seek(int(self.offsets[idx]))
        return fh.readline().decode("utf-8").strip()

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> dict:
        ids = self.tokenizer.encode(self._line(idx))[: self.max_tokens]
        if not ids:                                      # unknown-only line: keep 1 token
            ids = [self.tokenizer.unk_id]
        return {"tokens": torch.tensor(ids, dtype=torch.long), "id": f"line{idx}"}


def collate_lm(items: list[dict], pad_id: int) -> dict:
    """Trainer-shaped batch: feats == tokens (the model-agnostic contract's 4 keys)."""
    lens = torch.tensor([len(it["tokens"]) for it in items], dtype=torch.long)
    toks = torch.full((len(items), int(lens.max())), pad_id, dtype=torch.long)
    for i, it in enumerate(items):
        toks[i, : len(it["tokens"])] = it["tokens"]
    return {"feats": toks, "feat_lens": lens, "tokens": toks, "token_lens": lens,
            "ids": [it["id"] for it in items]}


def make_lm_dataloader(dataset: LMTextDataset, batch_tokens: int, *, shuffle: bool,
                       num_workers: int = 2, seed: int = 0, world_size: int = 1, rank: int = 0):
    """Token-budgeted, DDP-shardable loader (train shards; pass world_size=1 for dev)."""
    sampler = DistributedBucketBatchSampler(dataset.lengths, batch_tokens, world_size, rank,
                                            shuffle=shuffle, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        collate_fn=partial(collate_lm, pad_id=dataset.pad_id),
                        num_workers=num_workers, pin_memory=True)
    return loader, sampler
