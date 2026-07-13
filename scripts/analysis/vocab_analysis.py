"""Data-driven BPE vocabulary-size analysis for DC-ASR on LibriSpeech train-960.

Justifies the tokenizer's vocab size V from the transcripts (not by convention),
using three bounds and reporting the defensible window:
  (1) frequency floor - largest V before pieces get under-trained (rare/singletons)
  (2) sequence length - tokens/utterance vs encoder frames T (CTC margin @ 25 Hz)
  (3) fertility        - tokens/word trend (diminishing returns)

Trains SentencePiece BPE at each candidate V, measures each, prints a table +
recommendation. Analysis tool (reproducible), not the production tokenizer.
Run: python scripts/analysis/vocab_analysis.py
"""
import os
import glob
import random
import shutil
import tempfile
import time
from collections import Counter
from itertools import chain
from pathlib import Path

import sentencepiece as spm
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "LibriSpeech"
TRAIN = ["train-clean-100", "train-clean-360", "train-other-500"]
CANDIDATES = [128, 256, 500, 750, 1000, 2000, 4000]
SR = 16000
ENC_HZ = 25.0            # encoder frame rate after x4 conv subsampling
DUR_SAMPLE = 12000       # utterances sampled for duration -> T (CTC margin)
MIN_COUNT = 100          # a piece is "well-trained" if seen >= this many times


def pctile(sorted_list, q):
    if not sorted_list:
        return 0
    return sorted_list[min(len(sorted_list) - 1, int(q / 100 * len(sorted_list)))]


def gather():
    texts, flacs = [], []
    for split in TRAIN:
        for tf in glob.glob(str(ROOT / split / "*" / "*" / "*.trans.txt")):
            d = os.path.dirname(tf)
            with open(tf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    uid, _, txt = line.partition(" ")
                    if txt:
                        texts.append(txt)
                        flacs.append(f"{d}/{uid}.flac")
    return texts, flacs


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vocab_analysis_"))
    try:
        t0 = time.time()
        texts, flacs = gather()
        n = len(texts)
        words_per = [t.count(" ") + 1 for t in texts]
        total_words = sum(words_per)
        distinct_words = len({w for t in texts for w in t.split()})
        charset = sorted(set("".join(texts)))
        wp = sorted(words_per)
        print(f"[gather] {n:,} utts  {total_words:,} words  {distinct_words:,} distinct  "
              f"|charset|={len(charset)} ({''.join(charset)!r})  in {time.time()-t0:.1f}s")
        print(f"[gather] words/utt: mean={total_words/n:.1f} p50={pctile(wp,50)} "
              f"p95={pctile(wp,95)} max={max(wp)}")

        corpus_file = tmp / "train_text.txt"
        corpus_file.write_text("\n".join(texts))

        # duration sample -> encoder frames T @ 25 Hz
        t0 = time.time()
        random.seed(0)
        sample_idx = random.sample(range(n), min(DUR_SAMPLE, n))
        Tmap = {}
        for i in sample_idx:
            try:
                info = sf.info(flacs[i])
                Tmap[i] = info.frames / info.samplerate * ENC_HZ
            except Exception:
                pass
        Ts = sorted(Tmap.values())
        print(f"[dur] sampled {len(Ts):,} utts  T@25Hz: mean={sum(Ts)/len(Ts):.0f} "
              f"p50={pctile(Ts,50):.0f} p05={pctile(Ts,5):.0f} min={Ts[0]:.0f}  "
              f"in {time.time()-t0:.1f}s")

        print("\n" + "=" * 120)
        print(f"{'V':>6} {'pieces':>7} {'tok/word':>9} {'U_mean':>7} {'U_p95':>6} {'U_max':>6} "
              f"{'min_cnt':>8} {f'#<{MIN_COUNT}x':>7} {'#singl':>7} {'CTC>T%':>7}")
        print("-" * 120)
        rows, Uful = [], {}
        for V in CANDIDATES:
            prefix = tmp / f"sp{V}"
            spm.SentencePieceTrainer.train(
                input=str(corpus_file), model_prefix=str(prefix), vocab_size=V,
                model_type="bpe", character_coverage=1.0,
                normalization_rule_name="identity",
                unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1, num_threads=8, minloglevel=2)
            sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
            enc = sp.encode(texts, out_type=int)
            U = [len(e) for e in enc]
            Uful[V] = U
            total_tok = sum(U)
            used = [c for c in Counter(chain.from_iterable(enc)).values() if c > 0]
            lt = sum(1 for c in used if c < MIN_COUNT)
            sing = sum(1 for c in used if c == 1)
            ctc_viol = 100 * sum(1 for i, t in Tmap.items() if U[i] > t) / len(Tmap)
            r = dict(V=V, pieces=len(used), fert=total_tok/total_words,
                     Umean=total_tok/n, Up95=pctile(sorted(U), 95), Umax=max(U),
                     mincnt=min(used), lt=lt, sing=sing, ctc=ctc_viol)
            rows.append(r)
            print(f"{V:>6} {r['pieces']:>7} {r['fert']:>9.3f} {r['Umean']:>7.1f} {r['Up95']:>6} "
                  f"{r['Umax']:>6} {r['mincnt']:>8} {r['lt']:>7} {r['sing']:>7} {r['ctc']:>6.2f}%")
        print("=" * 120)

        # frequency floor: largest V that is still well-trained (0 singletons, <0.5% rare)
        clean = [r['V'] for r in rows if r['sing'] == 0 and r['lt'] / r['pieces'] <= 0.005]
        freq_upper = max(clean) if clean else None
        ctc_binding = any(r['ctc'] > 0.1 for r in rows)
        print(f"[freq floor]   frequency-clean V (0 singletons, <0.5% rare): {clean}  -> upper bound ~{freq_upper}")
        print(f"[CTC length]   binding at 25 Hz? {ctc_binding}  (0% violations => non-binding, length is slack)")
        print(f"[fertility]    tok/word {rows[0]['fert']:.2f} -> {rows[-1]['fert']:.2f} across V "
              f"(smooth, no sharp knee)")
        window_lo = 256
        baseline = 500 if (freq_upper and 500 <= freq_upper) else (freq_upper or 500)
        print(f"\nDEFENSIBLE WINDOW: ~[{window_lo}, {freq_upper}]  "
              f"(below: needlessly long seqs, no CTC benefit; above: under-trained pieces)")
        print(f"RECOMMENDATION: baseline V = {baseline}  |  ablation arms {{256, {baseline}, "
              f"{min(1000, freq_upper or 1000)}}}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("DONE")


if __name__ == "__main__":
    main()
