"""Evaluation: WER/RTF/compression metrics and scoring glue (plan §6.3)."""
from dcasr.eval.metrics import (
    ErrorStats, char_error_rate, levenshtein_counts, normalize_text,
    real_time_factor, token_error_rate, word_error_rate,
)

__all__ = ["ErrorStats", "word_error_rate", "char_error_rate", "token_error_rate",
           "levenshtein_counts", "normalize_text", "real_time_factor"]
