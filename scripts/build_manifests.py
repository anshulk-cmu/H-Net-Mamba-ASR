"""Entry point: build + persist LibriSpeech manifests (train-960 + dev/test).

Writes manifests/<name>.jsonl of {id, audio, text, frames}. train-960 = the three
train splits combined; dev/test kept separate. Audio paths are resolved to the data
partition so they survive repo moves. Usage: python scripts/build_manifests.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.data.librispeech import TRAIN_960, build_manifest, load_manifest
from dcasr.logging_utils import get_logger, setup_logging

REPO = Path(__file__).resolve().parents[1]
SPECS = {"train-960": TRAIN_960, "dev-clean": ["dev-clean"], "dev-other": ["dev-other"],
         "test-clean": ["test-clean"], "test-other": ["test-other"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--librispeech-root", default=str((REPO / "data" / "LibriSpeech").resolve()))
    ap.add_argument("--out", default=str(REPO / "manifests"))
    args = ap.parse_args()

    setup_logging("build_manifests")
    log = get_logger(__name__)
    root, out = Path(args.librispeech_root), Path(args.out)
    for name, splits in SPECS.items():
        p = build_manifest(root, splits, out / f"{name}.jsonl")
        log.info("%s: %d utterances -> %s", name, len(load_manifest(p)), p.name)
    log.info("done: %s", sorted(x.name for x in out.glob("*.jsonl")))


if __name__ == "__main__":
    main()
