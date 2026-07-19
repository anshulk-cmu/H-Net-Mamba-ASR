"""Unit tests for the BPE tokenizer (src/dcasr/data/tokenizer.py).

CPU-only (no GPU gate): the tokenizer is pure text/SentencePiece. Trains a small
model on a synthetic uppercase corpus once per module.
"""
import random

import pytest

from dcasr.data.tokenizer import Tokenizer

WORDS = ("THE QUICK BROWN FOX JUMPS OVER A LAZY DOG AND THEN RUNS HOME WITH HIS "
         "FRIEND WHO SPEAKS SOFTLY ABOUT MUSIC RIVERS MOUNTAINS SILENCE WINTER "
         "MORNING LIGHT SHADOW GARDEN LETTER PROMISE JOURNEY").split()


def _corpus(n=500, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 14))) for _ in range(n)]


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("tok") / "sp"
    return Tokenizer.train(_corpus(), prefix, vocab_size=120, hard_vocab_limit=False)


def test_special_id_layout(tok):
    assert (tok.unk_id, tok.bos_id, tok.eos_id, tok.pad_id) == (0, 1, 2, 3)
    assert tok.blank_id == tok.vocab_size == len(tok)
    assert 31 <= tok.vocab_size <= 120                 # >= 27 chars + 4 specials


def test_roundtrip_is_exact(tok):
    for s in _corpus(n=30, seed=7):
        assert tok.decode(tok.encode(s)) == s


def test_real_pieces_are_above_specials(tok):
    ids = tok.encode("THE QUICK BROWN FOX")
    assert ids and all(4 <= i < tok.vocab_size for i in ids)   # no special ids for plain text


def test_bos_eos_wrapping(tok):
    base = tok.encode("SILENCE RIVERS")
    wrapped = tok.encode("SILENCE RIVERS", bos=True, eos=True)
    assert wrapped[0] == tok.bos_id and wrapped[-1] == tok.eos_id
    assert wrapped[1:-1] == base


def test_decode_drops_specials(tok):
    s = "WINTER MORNING LIGHT"
    noisy = [tok.bos_id] + tok.encode(s) + [tok.eos_id, tok.pad_id, tok.blank_id]
    assert tok.decode(noisy) == s


def test_unknown_char_maps_to_unk(tok):
    ids = tok.encode("12345 @#$")            # digits/symbols never seen in the A-Z corpus
    assert tok.unk_id in ids


def test_encode_is_deterministic(tok):
    s = "GARDEN LETTER PROMISE JOURNEY"
    assert tok.encode(s) == tok.encode(s)


def test_reload_from_disk_matches(tok):
    reloaded = Tokenizer(tok.model_path)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.encode("THE LAZY DOG") == tok.encode("THE LAZY DOG")


def test_id_to_piece_blank(tok):
    assert tok.id_to_piece(tok.blank_id) == "<blank>"
    assert tok.id_to_piece(tok.encode("MUSIC")[0]).startswith("▁")  # SP word-start marker


def test_empty_string(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_foreign_special_layout_raises(tmp_path):
    """A SentencePiece model NOT trained via Tokenizer.train (SP default: pad disabled = -1)
    must fail loudly — AED/beam/LM defaults hardcode the (0,1,2,3) layout."""
    import sentencepiece as spm
    corpus_file = tmp_path / "c.txt"
    corpus_file.write_text("\n".join(_corpus(100)) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus_file), model_prefix=str(tmp_path / "foreign"),
                                   vocab_size=80, model_type="bpe", hard_vocab_limit=False,
                                   minloglevel=1)
    with pytest.raises(ValueError):
        Tokenizer(tmp_path / "foreign.model")
