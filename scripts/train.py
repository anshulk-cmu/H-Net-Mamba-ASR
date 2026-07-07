"""Entry point: train one DC-ASR model from a YAML config.

Usage (planned): python scripts/train.py --config configs/typeA_small_N1_ctc.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.logging_utils import get_logger, setup_logging

if __name__ == "__main__":
    setup_logging("train")
    get_logger(__name__).error(
        "Not yet implemented — scaffold. See docs/DC-ASR_experimental_plan.md §6.")
    raise SystemExit(1)
