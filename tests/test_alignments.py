"""Unit tests for MFA alignment support (src/dcasr/interp/alignments.py). CPU-only,
no MFA: crafted long-format TextGrids, tmp corpora, seeded subsets, integrity checks."""
import json
from pathlib import Path

import pytest

from dcasr.interp.alignments import (SAMPLE_RATE, alignment_record, check_alignment,
                                     load_alignments, load_manifest, parse_textgrid,
                                     prepare_corpus, select_subset, speaker_of,
                                     write_alignments)

TG = '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.5
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 0.4
            text = ""
        intervals [2]:
            xmin = 0.4
            xmax = 1.1
            text = "the"
        intervals [3]:
            xmin = 1.1
            xmax = 2.0
            text = "cat"
        intervals [4]:
            xmin = 2.0
            xmax = 2.5
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 2.5
        intervals: size = 5
        intervals [1]:
            xmin = 0.0
            xmax = 0.4
            text = ""
        intervals [2]:
            xmin = 0.4
            xmax = 0.8
            text = "DH"
        intervals [3]:
            xmin = 0.8
            xmax = 1.1
            text = "AH0"
        intervals [4]:
            xmin = 1.1
            xmax = 2.0
            text = "K"
        intervals [5]:
            xmin = 2.0
            xmax = 2.5
            text = ""
'''


def test_parse_textgrid_tiers_and_labels():
    tiers = parse_textgrid(TG)
    assert set(tiers) == {"words", "phones"}
    assert tiers["words"] == [("", 0.0, 0.4), ("the", 0.4, 1.1), ("cat", 1.1, 2.0),
                             ("", 2.0, 2.5)]
    assert tiers["phones"][1] == ("DH", 0.4, 0.8)
    assert len(tiers["phones"]) == 5


def test_parse_textgrid_quote_escaping_and_errors():
    tg = TG.replace('text = "cat"', 'text = "ca""t"')
    assert ('ca"t', 1.1, 2.0) in parse_textgrid(tg)["words"]
    with pytest.raises(ValueError, match="ooTextFile"):
        parse_textgrid("not a textgrid at all")
    with pytest.raises(ValueError, match="no tiers"):
        parse_textgrid('File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\n')


def test_alignment_record_drops_silence_and_requires_content():
    rec = alignment_record("u1", parse_textgrid(TG))
    assert rec["words"] == [["the", 0.4, 1.1], ["cat", 1.1, 2.0]]
    assert [p[0] for p in rec["phones"]] == ["DH", "AH0", "K"]     # sil + "" dropped
    empty = {"words": [("", 0, 1)], "phones": [("", 0, 1)]}
    with pytest.raises(ValueError, match="lacks words/phones"):
        alignment_record("u2", empty)


def test_check_alignment_clean_and_problems():
    rec = alignment_record("u1", parse_textgrid(TG))
    assert check_alignment(rec, "THE CAT", 2.5) == []
    assert check_alignment(rec, "THE DOG", 2.5)                    # sequence mismatch
    assert any("beyond audio" in p for p in check_alignment(rec, "THE CAT", 1.0))
    bad = {"id": "u", "words": [["a", 0.5, 0.4]], "phones": [["A", 0.0, 0.5]]}
    assert any("non-positive" in p for p in check_alignment(bad, "A", 1.0))
    over = {"id": "u", "words": [["a", 0.0, 1.0], ["b", 0.5, 1.5]],
            "phones": [["A", 0.0, 1.5]]}
    assert any("overlap" in p for p in check_alignment(over, "A B", 2.0))


def test_speaker_of():
    assert speaker_of("1272-128104-0000") == "1272"


def _manifest(tmp_path, n=6, dur_s=2.0):
    entries = []
    for i in range(n):
        wav = tmp_path / "audio" / f"spk{i % 2}-c-{i:04d}.flac"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"fake")
        entries.append({"id": f"{100 + i % 2}-000-{i:04d}", "audio": str(wav),
                        "text": f"WORD NUMBER {i}", "frames": int(dur_s * SAMPLE_RATE)})
    p = tmp_path / "m.jsonl"
    with open(p, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return p, entries


def test_prepare_corpus_layout(tmp_path):
    p, entries = _manifest(tmp_path)
    corpus = tmp_path / "corpus"
    assert prepare_corpus(entries, corpus) == 6
    for e in entries:
        spk = speaker_of(e["id"])
        link = corpus / spk / f"{e['id']}.flac"
        assert link.is_symlink() and link.resolve() == Path(e["audio"]).resolve()
        assert (corpus / spk / f"{e['id']}.lab").read_text().strip() == e["text"]
    assert prepare_corpus(entries, corpus) == 6                    # idempotent rerun
    entries[0]["audio"] = str(tmp_path / "nope.flac")
    with pytest.raises(FileNotFoundError, match="audio missing"):
        prepare_corpus(entries, corpus)


def test_select_subset_budget_and_determinism(tmp_path):
    p, entries = _manifest(tmp_path, n=20, dur_s=360.0)            # 0.1 h each
    s1 = select_subset(entries, hours=0.55, seed=3)
    s2 = select_subset(entries, hours=0.55, seed=3)
    s3 = select_subset(entries, hours=0.55, seed=4)
    assert [e["id"] for e in s1] == [e["id"] for e in s2]          # deterministic
    assert [e["id"] for e in s1] != [e["id"] for e in s3]          # seed-sensitive
    total_h = sum(e["frames"] / SAMPLE_RATE for e in s1) / 3600
    assert 0.45 <= total_h <= 0.55                                 # fills, never exceeds
    assert [e["id"] for e in s1] == sorted(e["id"] for e in s1)
    with pytest.raises(ValueError, match="hours"):
        select_subset(entries, hours=0)


def test_write_load_alignments_roundtrip(tmp_path):
    recs = [{"id": "a", "words": [["hi", 0.1, 0.5]], "phones": [["HH", 0.1, 0.3]]}]
    out = tmp_path / "al" / "dev.jsonl"
    assert write_alignments(recs, out) == 1
    assert load_alignments(out) == recs


def test_load_manifest_empty_raises(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("\n")
    with pytest.raises(ValueError, match="empty manifest"):
        load_manifest(p)


# ── fixes from the adversarial verification (wf_fc8865d8) ────────────────────
def test_real_words_silence_sil_are_kept():
    """MFA 3.4.1 marks silence ONLY with ''; the English words SILENCE/SIL must
    survive into the record (the major finding: they were silently deleted)."""
    tg = TG.replace('text = "cat"', 'text = "silence"')
    rec = alignment_record("u1", parse_textgrid(tg))
    assert [w[0] for w in rec["words"]] == ["the", "silence"]
    tg2 = TG.replace('text = "cat"', 'text = "sil"')
    rec2 = alignment_record("u1", parse_textgrid(tg2))
    assert [w[0] for w in rec2["words"]] == ["the", "sil"]
    assert check_alignment(rec, "THE SILENCE", 2.5) == []
    # OOV convention: 'spn' is an ordinary phone, never dropped
    tg3 = TG.replace('text = "DH"', 'text = "spn"')
    assert "spn" in [p[0] for p in alignment_record("u1", parse_textgrid(tg3))["phones"]]


def test_check_alignment_edge_apostrophe_normalized():
    rec = {"id": "u", "words": [["bush", 0.1, 0.5], ["and", 0.5, 0.9]],
           "phones": [["B", 0.1, 0.9]]}
    assert check_alignment(rec, "BUSH' AND", 1.0) == []           # MFA strips the edge '
    bad = check_alignment(rec, "BUSH' OR", 1.0)
    assert bad and "first diff at 1" in bad[0]                    # position reported


def test_multiline_label_raises():
    tg = TG.replace('text = "cat"', 'text = "ca\nt"')
    with pytest.raises(ValueError, match="unterminated label"):
        parse_textgrid(tg)


def test_rounding_6dp_and_negative_start():
    rec = alignment_record("u", {"words": [("w", 0.999996, 0.999998)],
                                 "phones": [("W", 0.999996, 0.999998)]})
    assert rec["words"][0][1] < rec["words"][0][2]                # 6dp keeps it positive
    neg = {"id": "u", "words": [["a", -0.5, 0.4]], "phones": [["A", 0.0, 0.4]]}
    assert any("negative start" in p for p in check_alignment(neg, "A", 1.0))


def test_prepare_corpus_resolves_relative_and_duplicate_id(tmp_path):
    wav = tmp_path / "a" / "x.flac"
    wav.parent.mkdir()
    wav.write_bytes(b"f")
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        e = {"id": "7-0-0000", "audio": "a/x.flac", "text": "HI", "frames": 16000}
        prepare_corpus([e], tmp_path / "c")
        link = tmp_path / "c" / "7" / "7-0-0000.flac"
        assert Path(os.readlink(link)).is_absolute()              # no dangling target
    finally:
        os.chdir(old)
    dup = {"id": "7-0-0001", "audio": str(wav), "text": "A", "frames": 1}
    with pytest.raises(ValueError, match="duplicate utterance id"):
        prepare_corpus([dup, dict(dup)], tmp_path / "c2")


def test_select_subset_single_over_budget_and_take_all(tmp_path):
    _, entries = _manifest(tmp_path, n=1, dur_s=7200.0)           # one 2h utt
    assert len(select_subset(entries, hours=0.5, seed=1)) == 1    # still picks one
    _, small = _manifest(tmp_path / "s", n=3, dur_s=60.0)
    assert len(select_subset(small, hours=100.0, seed=1)) == 3    # budget > corpus: all


# ── script-level contracts (scripts/run_mfa.py) ──────────────────────────────
def _load_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_mfa_script", Path(__file__).resolve().parents[1] / "scripts" / "run_mfa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_mfa_align_env_contract(tmp_path):
    """PATH prepend + MFA_ROOT_DIR are load-bearing (OpenFst discovery, /data state)."""
    rm = _load_script()
    bin_dir = tmp_path / "envbin"
    bin_dir.mkdir()
    fake = bin_dir / "mfa"
    fake.write_text('#!/bin/sh\necho "PATH=$PATH" > "$MFA_DUMP"\n'
                    'echo "ROOT=$MFA_ROOT_DIR" >> "$MFA_DUMP"\nexit 0\n')
    fake.chmod(0o755)
    dump = tmp_path / "dump.txt"
    import os
    os.environ["MFA_DUMP"] = str(dump)
    try:
        rm.run_mfa_align(tmp_path / "c", tmp_path / "o", mfa_bin=fake,
                         mfa_root="/data/somewhere/mfa")
    finally:
        del os.environ["MFA_DUMP"]
    lines = dump.read_text().splitlines()
    assert lines[0].startswith(f"PATH={bin_dir}:")
    assert lines[1] == "ROOT=/data/somewhere/mfa"
    with pytest.raises(RuntimeError, match="mfa binary not found"):
        rm.run_mfa_align(tmp_path / "c", tmp_path / "o",
                         mfa_bin=tmp_path / "nope" / "mfa")


def test_parse_split_corrupt_grid_rejection_and_missing(tmp_path):
    rm = _load_script()
    tg_dir = tmp_path / "tg"
    entries = []
    for i, text in enumerate(["THE CAT", "THE CAT", "THE DOG", "THE CAT"]):
        uid = f"{50 + i}-0-0000"
        entries.append({"id": uid, "audio": "/x", "text": text,
                        "frames": int(2.5 * SAMPLE_RATE)})
    (tg_dir / "50").mkdir(parents=True)
    (tg_dir / "50" / "50-0-0000.TextGrid").write_text(TG)          # clean
    (tg_dir / "51").mkdir()
    (tg_dir / "51" / "51-0-0000.TextGrid").write_text(TG[:100])    # truncated
    (tg_dir / "52").mkdir()
    (tg_dir / "52" / "52-0-0000.TextGrid").write_text(TG)          # text mismatch (DOG)
    records, rejected, missing, problems = rm.parse_split(entries, tg_dir)  # 53 absent
    assert [r["id"] for r in records] == ["50-0-0000"]
    assert [r["id"] for r in rejected] == ["52-0-0000"]            # dirty: not ground truth
    assert set(missing) == {"51-0-0000", "53-0-0000"}              # corrupt + absent
    assert any("unparseable" in p for p in problems)
    assert any("word sequence" in p for p in problems)
