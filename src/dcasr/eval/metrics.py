"""ASR error metrics: WER / CER / TER with a Substitution/Deletion/Insertion breakdown.

The canonical, tested scorer reused by decode/score scripts and the paper tables (the trainer
keeps a lightweight inline WER for dev monitoring). Error rate = (S+D+I)/N over the chosen units:
words (WER), characters (CER, spaces stripped), or model tokens (TER). A Levenshtein alignment
with backtrace gives the S/D/I split; its total is cross-checked against `editdistance`. A stated
text-normalization policy (lowercase, strip punctuation, keep apostrophes, collapse whitespace)
keeps numbers comparable with ESPnet/Kaldi scoring. Pure-Python / CPU. RTF helper included.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

_PUNCT = re.compile(r"[^\w\s']")            # keep letters/digits/underscore, spaces, apostrophes


def normalize_text(s: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """Lowercase, strip punctuation (apostrophes kept), collapse whitespace."""
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def levenshtein_counts(ref: Sequence, hyp: Sequence) -> tuple[int, int, int, int]:
    """Aligned (sub, del, ins, cor) between two unit sequences via edit-distance DP + backtrace.

    Invariants: cor+sub+del == len(ref) and cor+sub+ins == len(hyp); sub+del+ins == edit distance.
    """
    R, H = len(ref), len(hyp)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, H + 1):
            if ri == hyp[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j - 1], prev[j], row[j - 1])
    # backtrace: prefer match/sub (diagonal), then del (up), then ins (left)
    i, j, sub, dele, ins, cor = R, H, 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            cor += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1; i -= 1
        else:
            ins += 1; j -= 1
    return sub, dele, ins, cor


@dataclass
class ErrorStats:
    """Aggregated error counts over a set of utterances. Rates are fractions (×100 for %)."""
    n_ref: int = 0                          # total reference units
    sub: int = 0
    dele: int = 0
    ins: int = 0
    cor: int = 0
    n_utt: int = 0
    n_correct: int = 0                      # exact-match utterances

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def er(self) -> float:                  # error rate (S+D+I)/N, fraction (can exceed 1)
        return self.errors / max(1, self.n_ref)

    @property
    def sub_rate(self) -> float:
        return self.sub / max(1, self.n_ref)

    @property
    def del_rate(self) -> float:
        return self.dele / max(1, self.n_ref)

    @property
    def ins_rate(self) -> float:
        return self.ins / max(1, self.n_ref)

    @property
    def sentence_acc(self) -> float:
        return self.n_correct / max(1, self.n_utt)

    def as_dict(self, prefix: str = "wer") -> dict[str, float]:
        """Percentages for logging: {prefix, prefix_sub, prefix_del, prefix_ins, sent_acc}."""
        return {prefix: 100 * self.er, f"{prefix}_sub": 100 * self.sub_rate,
                f"{prefix}_del": 100 * self.del_rate, f"{prefix}_ins": 100 * self.ins_rate,
                "sent_acc": 100 * self.sentence_acc}


def _score(pairs: Iterable[tuple[Sequence, Sequence]]) -> ErrorStats:
    st = ErrorStats()
    for ref, hyp in pairs:
        sub, dele, ins, cor = levenshtein_counts(ref, hyp)
        st.n_ref += len(ref); st.sub += sub; st.dele += dele; st.ins += ins; st.cor += cor
        st.n_utt += 1
        st.n_correct += int(list(ref) == list(hyp))
    return st


def word_error_rate(refs: Sequence[str], hyps: Sequence[str], normalize: bool = True) -> ErrorStats:
    norm = normalize_text if normalize else (lambda s: s)
    return _score((norm(r).split(), norm(h).split()) for r, h in zip(refs, hyps))


def char_error_rate(refs: Sequence[str], hyps: Sequence[str], normalize: bool = True,
                    remove_space: bool = True) -> ErrorStats:
    norm = normalize_text if normalize else (lambda s: s)
    def chars(s):
        s = norm(s)
        return list(s.replace(" ", "") if remove_space else s)
    return _score((chars(r), chars(h)) for r, h in zip(refs, hyps))


def token_error_rate(ref_tokens: Sequence[Sequence[int]],
                     hyp_tokens: Sequence[Sequence[int]]) -> ErrorStats:
    """Error rate over model output tokens (ids); no text normalization."""
    return _score((list(r), list(h)) for r, h in zip(ref_tokens, hyp_tokens))


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    """RTF = compute wall-clock / audio duration (≤1 = faster than real time)."""
    return processing_seconds / max(1e-9, audio_seconds)
