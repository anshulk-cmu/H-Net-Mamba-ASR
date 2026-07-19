"""Entry point: analytic efficiency report (params + GFLOPs) for one experiment config.

Thin orchestrator over tested library code (eval/efficiency.py). Derives the paper's
per-cell efficiency numbers from the config alone (no GPU); measured decode speed
(RTF) lives in decode.py's summaries. Writes experiments/<run>/efficiency/
{efficiency.json, report.txt}.

Usage:
    python scripts/efficiency.py --config configs/typeA_small_N1_ctc.yaml
    python scripts/efficiency.py --config <cfg> efficiency.audio_seconds=6.9 \
        efficiency.kept_fractions=[0.52]     # realised keep-fractions from a trained run
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dcasr.data.tokenizer import Tokenizer
from dcasr.eval.efficiency import efficiency_report, format_efficiency
from dcasr.logging_utils import get_logger, setup_logging
from dcasr.provenance import collect_provenance
from dcasr.tasks.build import _plain

REPO = Path(__file__).resolve().parents[1]
logger = get_logger(__name__)


def load_config(config_path, overrides):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def run(cfg, repo_root=REPO):
    """Compute + persist the efficiency report. Returns the report dict."""
    ec = _plain(cfg.get("efficiency")) or {}
    tokenizer = Tokenizer(Path(repo_root) / str(cfg.bpemodel))
    report = efficiency_report(_plain(cfg), tokenizer.vocab_size,
                               audio_seconds=float(ec.get("audio_seconds", 10.0)),
                               kept_fractions=ec.get("kept_fractions"))
    report["provenance"] = collect_provenance(
        cfg, repo_root=repo_root, world_size=1, seed=int(cfg.experiment.seed),
        manifests=[], extra={"run_name": str(cfg.experiment.name)})

    out = Path(repo_root) / "experiments" / str(cfg.experiment.name) / "efficiency"
    out.mkdir(parents=True, exist_ok=True)
    (out / "efficiency.json").write_text(json.dumps(report, indent=1))
    table = format_efficiency(report)
    (out / "report.txt").write_text(table + "\n")
    for line in table.splitlines():
        logger.info("%s", line)
    logger.info("efficiency -> %s", out)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="experiment YAML")
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    setup_logging(f"efficiency_{cfg.experiment.name}")
    logger.info("config=%s overrides=%s", args.config, args.overrides)
    run(cfg)


if __name__ == "__main__":
    main()
