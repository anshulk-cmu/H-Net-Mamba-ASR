"""BPE tokenizer for DC-ASR (SentencePiece): text <-> subword-id sequences.

Fixed special-id layout: unk=0, bos=1, eos=2, pad=3; subword pieces at 4..V-1.
The CTC blank is appended at id V (`blank_id`) — a CTC head needs V+1 outputs;
encode()/decode() stay in the [0, V) space (AED wraps targets with bos/eos, CTC not).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import sentencepiece as spm

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_VOCAB_SIZE = 500
_UNK_ID, _BOS_ID, _EOS_ID, _PAD_ID = 0, 1, 2, 3


class Tokenizer:
    """SentencePiece BPE tokenizer with a fixed special-token layout."""

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        got = (self.unk_id, self.bos_id, self.eos_id, self.pad_id)
        if got != (_UNK_ID, _BOS_ID, _EOS_ID, _PAD_ID):
            # AED/beam/LM defaults hardcode this layout — a foreign model must not load silently
            raise ValueError(f"{self.model_path}: special ids (unk,bos,eos,pad)={got} violate "
                             f"the fixed contract (0,1,2,3); retrain via Tokenizer.train()")
        self.blank_id = self.vocab_size            # CTC blank, appended beyond the SP vocab
        logger.debug("Tokenizer(%s) vocab=%d blank=%d bos=%d eos=%d pad=%d unk=%d",
                     self.model_path, self.vocab_size, self.blank_id,
                     self.bos_id, self.eos_id, self.pad_id, self.unk_id)

    @classmethod
    def train(cls, corpus: str | Path | Iterable[str], model_prefix: str | Path,
              vocab_size: int = DEFAULT_VOCAB_SIZE, character_coverage: float = 1.0,
              **train_kwargs) -> "Tokenizer":
        """Train a BPE model and return a Tokenizer.

        `corpus` is a text-file path (one transcript per line) or an iterable of
        transcript strings. Writes <model_prefix>.model / .vocab. Extra keyword args
        pass through to SentencePieceTrainer (e.g. hard_vocab_limit=False).
        """
        model_prefix = str(model_prefix)
        Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
        cfg = dict(
            model_prefix=model_prefix, vocab_size=vocab_size, model_type="bpe",
            character_coverage=character_coverage, normalization_rule_name="identity",
            unk_id=_UNK_ID, bos_id=_BOS_ID, eos_id=_EOS_ID, pad_id=_PAD_ID,
            num_threads=8, minloglevel=1)
        cfg.update(train_kwargs)

        tmp = None
        if isinstance(corpus, (str, Path)):
            src = str(corpus)
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                             encoding="utf-8") as f:
                for line in corpus:
                    f.write(line.rstrip("\n") + "\n")
                tmp = src = f.name
        try:
            spm.SentencePieceTrainer.train(input=src, **cfg)
        finally:
            if tmp is not None:
                os.remove(tmp)
        logger.info("trained BPE tokenizer: vocab=%d -> %s.model", vocab_size, model_prefix)
        return cls(f"{model_prefix}.model")

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """text -> subword ids in [0, vocab_size). Optionally wrap with bos/eos (AED targets)."""
        ids = self.sp.encode(text, out_type=int)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """subword ids -> text. Drops blank / bos / eos / pad and out-of-range ids first."""
        drop = {self.bos_id, self.eos_id, self.pad_id}
        pieces = [int(i) for i in ids if 0 <= int(i) < self.vocab_size and int(i) not in drop]
        return self.sp.decode(pieces)

    def id_to_piece(self, idx: int) -> str:
        return "<blank>" if idx == self.blank_id else self.sp.id_to_piece(int(idx))

    def __len__(self) -> int:
        return self.vocab_size
