"""Entry point: decode a checkpoint (CTC / AED / joint; greedy|beam; ±LM). Scaffold."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.logging_utils import get_logger, setup_logging

if __name__ == "__main__":
    setup_logging("decode")
    get_logger(__name__).error("Not yet implemented — scaffold.")
    raise SystemExit(1)
