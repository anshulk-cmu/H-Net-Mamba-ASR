"""Unit tests for the central logging setup (src/dcasr/logging_utils.py)."""
import logging

from dcasr.logging_utils import default_log_dir, get_logger, setup_logging


def test_setup_logging_writes_to_file(tmp_path):
    logfile = setup_logging("unittest", log_dir=tmp_path)
    get_logger("dcasr.unittest").info("hello from the unit test")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "hello from the unit test" in logfile.read_text(encoding="utf-8")
    logging.getLogger().handlers.clear()   # release the file handle


def test_default_log_dir_lives_on_data_partition(monkeypatch):
    monkeypatch.delenv("DCASR_LOG_DIR", raising=False)
    d = default_log_dir()
    assert d.name == "logs"
    # on Babel, <repo>/logs is a symlink onto /data (heavy files never on /home)
    if d.is_symlink():
        assert str(d.resolve()).startswith("/data/")


def test_env_var_overrides_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DCASR_LOG_DIR", str(tmp_path / "custom"))
    assert default_log_dir() == tmp_path / "custom"


def test_rank_suffix_under_ddp(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")                  # rotation is not multi-process safe
    assert setup_logging("unittest_rank", log_dir=tmp_path).name == "unittest_rank.rank1.log"
    logging.getLogger().handlers.clear()
    monkeypatch.delenv("RANK")
    assert setup_logging("unittest_rank", log_dir=tmp_path).name == "unittest_rank.log"
    logging.getLogger().handlers.clear()
