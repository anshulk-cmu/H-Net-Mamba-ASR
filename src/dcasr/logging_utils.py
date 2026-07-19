"""Central logging for DC-ASR — console + rotating file logs, stored on /data.

Convention (project rule): every module gets its logger via

    from dcasr.logging_utils import get_logger
    logger = get_logger(__name__)

and every *process entry point* (scripts/, trainer) calls setup_logging() once.
Log files go to $DCASR_LOG_DIR if set, else <repo>/logs/ — which on Babel is a
symlink to /data/user_data/anshulk/hnet-asr/logs, so heavy logs never land on
the /home quota. One rotating file per entry point (50 MB x 5 backups).
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def default_log_dir() -> Path:
    """$DCASR_LOG_DIR if set, else <repo>/logs (symlinked to the data node)."""
    env = os.environ.get("DCASR_LOG_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "logs"


def setup_logging(run_name: str = "dcasr", level: int = logging.INFO,
                  log_dir: str | Path | None = None) -> Path:
    """Configure the root logger once per process. Returns the log-file path.

    Under torchrun each rank gets its own file (RotatingFileHandler rotation is
    not multi-process safe): rank>0 logs to <run_name>.rank<N>.log.
    """
    log_dir = Path(log_dir) if log_dir is not None else default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", 0))
    suffix = f".rank{rank}" if rank > 0 else ""
    logfile = log_dir / f"{run_name}{suffix}.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(console)

    filehandler = RotatingFileHandler(logfile, maxBytes=50 * 1024 * 1024,
                                      backupCount=5, encoding="utf-8")
    filehandler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(filehandler)

    logging.getLogger(__name__).info("logging to %s", logfile)
    return logfile


def get_logger(name: str) -> logging.Logger:
    """Module-level logger; inherits the root handlers set by setup_logging()."""
    return logging.getLogger(name)
