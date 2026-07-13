"""Entry point: train the DC-ASR BPE tokenizer(s) on LibriSpeech train-960.

Trains one SentencePiece BPE model per vocab size (default: 500 baseline + 750
ablation), saves <out>/spm_bpe_<V>.model / .vocab, and verifies an exact
round-trip on a real-transcript sample before declaring success.

Usage: python scripts/build_tokenizer.py [--vocab-sizes 500 750] [--out data/tokenizer]
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.data.tokenizer import Tokenizer
from dcasr.logging_utils import get_logger, setup_logging

REPO = Path(__file__).resolve().parents[1]
TRAIN_SPLITS = ["train-clean-100", "train-clean-360", "train-other-500"]


def gather_transcripts(root: Path) -> list[str]:
    texts = []
    for split in TRAIN_SPLITS:
        for tf in glob.glob(str(root / split / "*" / "*" / "*.trans.txt")):
            with open(tf) as f:
                for line in f:
                    _, _, txt = line.strip().partition(" ")
                    if txt:
                        texts.append(txt)
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-root", default=str(REPO / "data" / "LibriSpeech"))
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[500, 750])
    ap.add_argument("--out", default=str(REPO / "data" / "tokenizer"))
    args = ap.parse_args()

    setup_logging("build_tokenizer")
    log = get_logger(__name__)

    texts = gather_transcripts(Path(args.librispeech_root))
    if not texts:
        log.error("no transcripts found under %s", args.librispeech_root)
        raise SystemExit(1)
    log.info("gathered %d train-960 transcripts", len(texts))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = texts[::200]                          # ~1400 utts spread across the corpus
    for V in args.vocab_sizes:
        prefix = out / f"spm_bpe_{V}"
        tok = Tokenizer.train(texts, prefix, vocab_size=V)
        bad = sum(tok.decode(tok.encode(s)) != s for s in sample)
        log.info("vocab=%d -> %s.model | round-trip %d/%d exact | blank_id=%d",
                 tok.vocab_size, prefix.name, len(sample) - bad, len(sample), tok.blank_id)
        if bad:
            log.error("round-trip FAILED for vocab=%d (%d mismatches)", V, bad)
            raise SystemExit(1)
    log.info("done: saved %s", sorted(p.name for p in out.glob("*.model")))


if __name__ == "__main__":
    main()
