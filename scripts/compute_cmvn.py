"""Entry point: compute + save global CMVN stats over train-960.

Streams raw log-Mel features (features.py) across the train-960 manifest with a
multi-worker DataLoader, accumulates fp64 mean/var (CMVNAccumulator), and saves
{mean, std, count} for GlobalCMVN.load. Usage: python scripts/compute_cmvn.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.data.features import CMVNAccumulator, LogMelFrontend
from dcasr.data.librispeech import LibriSpeechDataset, make_dataloader
from dcasr.data.tokenizer import Tokenizer
from dcasr.logging_utils import get_logger, setup_logging

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(REPO / "manifests" / "train-960.jsonl"))
    ap.add_argument("--tokenizer", default=str(REPO / "data" / "tokenizer" / "spm_bpe_500.model"))
    ap.add_argument("--out", default=str(REPO / "features" / "cmvn_train960.pt"))
    ap.add_argument("--max-frames", type=int, default=40000)
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()

    setup_logging("compute_cmvn")
    log = get_logger(__name__)
    tok = Tokenizer(args.tokenizer)                       # only to satisfy the dataset API
    ds = LibriSpeechDataset(args.manifest, LogMelFrontend(), tok)     # raw log-Mel (no cmvn/aug)
    loader, _ = make_dataloader(ds, max_frames=args.max_frames, augment=False,
                                num_workers=args.num_workers, world_size=1)
    log.info("computing CMVN over %d utts (%d batches, %d workers)",
             len(ds), len(loader), args.num_workers)

    acc = CMVNAccumulator()
    t0 = time.time()
    for bi, b in enumerate(loader):
        acc.update(b["feats"], b["feat_lens"])
        if (bi + 1) % 200 == 0:
            log.info("  %d/%d batches | %d frames | %.0fs",
                     bi + 1, len(loader), acc.count, time.time() - t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stats = acc.save(args.out)
    log.info("CMVN saved -> %s | count=%d | mean[:3]=%s std[:3]=%s | %.0fs",
             args.out, stats["count"],
             [round(x, 3) for x in stats["mean"][:3].tolist()],
             [round(x, 3) for x in stats["std"][:3].tolist()], time.time() - t0)


if __name__ == "__main__":
    main()
