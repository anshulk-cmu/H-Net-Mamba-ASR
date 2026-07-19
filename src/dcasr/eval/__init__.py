"""Evaluation: WER/RTF/compression metrics and scoring glue (plan §6.3)."""
from dcasr.eval.metrics import (
    ErrorStats, char_error_rate, levenshtein_counts, normalize_text,
    real_time_factor, token_error_rate, word_error_rate,
)
from dcasr.eval.efficiency import (
    efficiency_report, encoder_flops, encoder_params, format_efficiency,
)
from dcasr.eval.score import (
    bootstrap_split, discover_cells, format_report, gate_check,
    load_decode_records, score_decode_dir, score_records,
)

__all__ = ["ErrorStats", "word_error_rate", "char_error_rate", "token_error_rate",
           "levenshtein_counts", "normalize_text", "real_time_factor",
           "load_decode_records", "score_records", "bootstrap_split", "discover_cells",
           "gate_check", "score_decode_dir", "format_report",
           "efficiency_report", "encoder_params", "encoder_flops", "format_efficiency"]
