"""Unit tests for the ASR error metrics (src/dcasr/eval/metrics.py). CPU-only, pure text.

The S/D/I DP is cross-checked against the independent `editdistance` library (total) and its
own alignment invariants, plus fuzzed on random sequences.
"""
import random

import editdistance

from dcasr.eval import metrics as M


def test_levenshtein_sub_del_ins():
    assert M.levenshtein_counts(["a", "b", "c"], ["a", "x", "c"]) == (1, 0, 0, 2)   # sub
    assert M.levenshtein_counts(["a", "b", "c"], ["a", "c"]) == (0, 1, 0, 2)        # del
    assert M.levenshtein_counts(["a", "c"], ["a", "b", "c"]) == (0, 0, 1, 2)        # ins
    assert M.levenshtein_counts(["a", "b"], ["a", "b"]) == (0, 0, 0, 2)             # exact
    assert M.levenshtein_counts([], []) == (0, 0, 0, 0)


def test_alignment_invariants_and_total_vs_editdistance():
    rng = random.Random(0)
    alpha = "abcde"
    for _ in range(3000):
        ref = [rng.choice(alpha) for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice(alpha) for _ in range(rng.randint(0, 12))]
        sub, dele, ins, cor = M.levenshtein_counts(ref, hyp)
        assert cor + sub + dele == len(ref)                # ref fully consumed
        assert cor + sub + ins == len(hyp)                 # hyp fully consumed
        assert sub + dele + ins == editdistance.eval(ref, hyp)   # total == independent ref


def test_wer_formula_and_rates():
    st = M.word_error_rate(["the cat sat on the mat"], ["the dog sat the mat now"])
    # ref 6 words; hyp: cat->dog (sub), 'on' deleted, 'now' inserted -> S1 D1 I1, errors 3
    assert (st.sub, st.dele, st.ins, st.n_ref) == (1, 1, 1, 6)
    assert abs(st.er - 3 / 6) < 1e-12
    assert abs(st.sub_rate - 1 / 6) < 1e-12


def test_wer_perfect_and_over_one():
    assert M.word_error_rate(["a b c"], ["a b c"]).er == 0.0
    over = M.word_error_rate(["a"], ["a b c d e"])          # 4 insertions on 1 ref word
    assert over.er == 4.0                                   # WER can exceed 1.0


def test_cer_strips_spaces():
    st = M.char_error_rate(["hello world"], ["hallo world"])   # e->a, 10 ref chars (no space)
    assert st.n_ref == 10 and st.sub == 1 and st.er == 0.1


def test_ter_over_token_ids():
    st = M.token_error_rate([[4, 5, 6]], [[4, 9, 6, 7]])       # 1 sub, 1 ins, ref len 3
    assert (st.sub, st.ins, st.n_ref) == (1, 1, 3)
    assert abs(st.er - 2 / 3) < 1e-12


def test_normalize_text():
    assert M.normalize_text("Hello, WORLD!  it's  fine.") == "hello world it's fine"
    assert M.normalize_text("A-B  C") == "a b c"            # punctuation -> space, collapse


def test_sentence_accuracy():
    st = M.word_error_rate(["a b", "c d", "e f"], ["a b", "c x", "e f"])
    assert st.n_utt == 3 and st.n_correct == 2
    assert abs(st.sentence_acc - 2 / 3) < 1e-12


def test_as_dict_percentages():
    d = M.word_error_rate(["the cat sat on the mat"], ["the dog sat the mat now"]).as_dict("wer")
    assert set(d) == {"wer", "wer_sub", "wer_del", "wer_ins", "sent_acc"}
    assert abs(d["wer"] - 50.0) < 1e-9                      # 3/6 -> 50%
    assert d["sent_acc"] == 0.0


def test_char_error_rate_matches_editdistance_total():
    rng = random.Random(1)
    refs = [" ".join(rng.choice(["cat", "dog", "the", "sat"]) for _ in range(rng.randint(1, 8)))
            for _ in range(50)]
    hyps = [" ".join(rng.choice(["cat", "dog", "the", "mat"]) for _ in range(rng.randint(1, 8)))
            for _ in range(50)]
    st = M.char_error_rate(refs, hyps)
    total = sum(editdistance.eval(M.normalize_text(r).replace(" ", ""),
                                  M.normalize_text(h).replace(" ", "")) for r, h in zip(refs, hyps))
    assert st.errors == total


def test_length_mismatch_raises():
    import pytest
    from dcasr.eval.metrics import char_error_rate, token_error_rate, word_error_rate
    with pytest.raises(ValueError):
        word_error_rate(["A B"], ["A B", "C"])       # zip would silently drop the surplus
    with pytest.raises(ValueError):
        char_error_rate(["AB", "CD"], ["AB"])
    with pytest.raises(ValueError):
        token_error_rate([[1, 2]], [])
