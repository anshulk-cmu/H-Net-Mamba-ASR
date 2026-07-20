# CARVE: Carving Speech at Its Natural Joints

**CARVE** (**C**ontent-**A**daptive **R**ate-**V**arying **E**ncoder; code name `dcasr`) is a speech recognizer that asks a simple question with consequences for how speech encoders are designed: *can a model learn its own acoustic hierarchy — where phones and words begin and end — purely from the recognition objective, instead of having a frame-rate schedule hand-designed for it?*

Mainstream ASR encoders process audio at fixed frame rates chosen by their designers (Conformer at one rate throughout; Zipformer with a hand-tuned U-Net-style rate schedule). Speech itself, however, is hierarchical and variable-rate: phones last tens of milliseconds, words hundreds, and the information density of the signal fluctuates constantly. CARVE replaces the hand-designed schedule with **H-Net dynamic chunking** (Hwang, Wang & Gu, arXiv:2507.07955) — a differentiable, content-based boundary predictor — inserted between stacks of **Mamba-2** state-space blocks, so the encoder itself decides *where* to compress. If a learned hierarchy matches a hand-designed one on word error rate, self-learned chunking becomes the better default: comparable accuracy, less architecture tuning, and — uniquely — **interpretable learned units** whose boundaries can be scored against linguistic ground truth. The name is Plato's metaphor: a good analysis carves nature at its joints; this encoder is trained to find the joints on its own.

Full experimental design: [`docs/DC-ASR_experimental_plan.md`](docs/DC-ASR_experimental_plan.md). Idea narrative: [`docs/DC-ASR_research_proposal.md`](docs/DC-ASR_research_proposal.md).

---

## 1. Scientific questions

| # | Hypothesis |
|---|---|
| H1 | A learned hierarchy is **WER-competitive** with hand-designed hierarchical encoders (published Zipformer-M as the reference bar) at matched parameter count. |
| H2 | **Learned** boundaries beat **fixed-stride pooling** at the same compression rate. |
| H3 | Splitting one N× compression into two √N× stages (Type B) yields cleaner phone-then-word tiers than a single stage (Type A). |
| H4 | Boundary structure is **self-learned from the recognition signal**: boundary-F1 against forced alignments rises during training alongside WER improvement (emergence). |
| H5 | There is a compression sweet spot where units are maximally linguistic at minimal WER cost (compression–accuracy–linguistics surface). |

A negative result is informative: if learned boundaries track acoustics (silence, energy) rather than linguistic units, that is a finding about what the ASR objective does and does not induce, reported against a matched random-boundary floor.

## 2. Architecture

```
      80-dim log-mel @ 100 Hz
              │
        [ Conv subsampling ×4 ]     → 25 Hz frame sequence
              │
   ┌──────────▼───────────────────────────────┐
   │  ENCODER stage: 4 × Mamba-2 blocks        │
   │   [ Router ]  p_t = ½(1 − cos(q_t,k_t−1)) │
   │   [ Downsample ] keep frames with p ≥ 0.5 │
   └──────────┬────────────────────────────────┘
              │   (Type B repeats this stage once more)
   ┌──────────▼───────────┐
   │  MAIN: 12 × Mamba-2   │   ← runs at the compressed rate
   └──────────┬───────────┘
   ┌──────────▼────────────────────────────────┐
   │  DECODER stage: chunk-EMA smoothing →      │
   │  upsample (gather) → confidence-STE →      │
   │  4 × Mamba-2 blocks                        │
   └──────────┬────────────────────────────────┘
              │
     [ CTC head ]  +  [ 6-layer Transformer AED head ]
```

- **Router.** A cosine-dissimilarity boundary predictor: a boundary is declared where consecutive frames stop resembling each other. Gradients flow through a confidence-weighted straight-through estimator; a ratio loss (β = 0.03) steers the mean keep-fraction toward the 1/N target. The dechunk path follows the H-Net paper exactly (Eq. 5 → 8 → 9: chunk-rate EMA with downsampled probabilities, then upsample, then STE) — verified against an independent implementation of the equations.
- **Two types at matched compression.** Type A compresses once by N; Type B compresses twice by √N (iso-compression), so any difference between them is attributable to *staging*, not rate. N ∈ {1, 2, 3, 4}; **N = 1 is the no-chunk control** (pure Mamba encoder).
- **One model, three decoders.** Each cell trains a single hybrid model (loss = 0.3·CTC + 0.7·AED with label smoothing); CTC, AED, and joint CTC+AED decoding are read-outs of that one model, each with and without an external Transformer LM (shallow fusion) — a 7-cell decode matrix per run.
- **Sizes.** Small = 78.9 M parameters (61.7 M encoder + 16.9 M AED + 0.2 M CTC); Large per the plan. Vocabulary: 500 BPE units (750 as an ablation), CTC blank appended.
- **Data.** LibriSpeech-960h only, speed-perturbed ×3, SpecAugment. Baselines are **cited from the literature, not re-trained**; the entire compute budget goes to CARVE's own grid.

**The grid: 22 training runs** — {A, B} × N{1–4} × {Small, Large} hybrid cells (16), fixed-stride pooling controls (4), and a 750-vocabulary pair (2). The first cell (A·Small·N=1) doubles as the go/no-go gate: its CTC-greedy test-clean WER must beat 12 %.

## 3. Repository layout

```
configs/       one YAML per experiment cell — every knob lives here, no code edits to swap
docs/          experimental plan (design authority) + research proposal
scripts/       entry points: train, decode, score_wer, efficiency, run_interp, run_mfa,
               build_manifests, build_tokenizer, compute_cmvn; slurm/ launchers
src/dcasr/     the package: data/ models/ decoders/ training/ tasks/ eval/ interp/
tests/         446 unit/regression tests (CPU + CUDA; pytest)
data/ experiments/ checkpoints/ features/ manifests/ alignments/ logs/
               symlinks → the cluster data partition (code on /home, bytes on /data)
```

Everything is **config-driven** in the ESPnet style: components resolve through `name + _conf` registries (`build_model`, `build_frontend`, `build_optimizer`, …), so a new encoder or scheduler is a registry entry plus a YAML key, and the Trainer is model-agnostic.

## 4. Setup

Requirements: Linux, NVIDIA GPU (Ampere or newer; bf16), CUDA-12.9-compatible driver, Python 3.10, and the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) in its own conda env for the interpretability ground truth.

```bash
conda create -n hnet-asr python=3.10 -y && conda activate hnet-asr
pip install torch==2.12.1+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm
pip install -r requirements.txt
# separate env for forced alignment:
conda create -n mfa -c conda-forge montreal-forced-aligner -y
```

The `--no-deps` flag on the Mamba kernels is deliberate: it prevents pip from silently swapping in a torch build for a newer CUDA than the driver supports. One-shot cluster script: `scripts/setup/setup_env_babel.sh`.

## 5. Reproduction, end to end

**Step 1 — data preparation** (once):

```bash
python scripts/build_manifests.py      # scan LibriSpeech → {id, audio, text, frames} JSONL manifests
python scripts/build_tokenizer.py      # SentencePiece BPE-500 (and BPE-750) on the 960h transcripts
python scripts/compute_cmvn.py         # global CMVN statistics over train-960
python scripts/run_mfa.py              # MFA forced alignment: phone/word ground truth for dev sets
                                       #   + a seeded 10 h train subset (probe training data)
python scripts/train_lm.py --config configs/lm_transformer_500.yaml   # external LM for fusion
```

**Step 2 — train + evaluate one cell.** A single SLURM submission runs the *entire* experiment — training, the full decode matrix, scoring, efficiency accounting, and the interpretability suite — and survives preemption, node failures, and time limits by design:

```bash
sbatch -J carve_asn1 scripts/slurm/run_cell_e2e_4gpu.sh \
    configs/typeA_small_N1.yaml typeA_small_N1
```

The pipeline stages, each skipped automatically on requeue if already complete:

1. **Train** — `torchrun` ×4 GPUs, DDP with a per-GPU frame budget (`batch_bins`; global batch = bins × GPUs), bf16, AdamW + warmup-then-inverse-sqrt schedule, up to 200 epochs with plateau early-stopping on dev WER + CER + loss; checkpoint every 5 epochs, all retained (the emergence analysis needs them). Training resumes bit-exactly after any interruption (`--resume auto` restores weights, optimizer, scheduler, scaler, RNG, and metric history).
2. **Decode** — all 7 cells × 4 eval splits, one split per GPU in parallel: `ctc_greedy`, `ctc_beam`, `aed_beam`, `joint_beam`, and the three beam variants again with LM fusion.
3. **Score** — WER **and CER** with substitution/deletion/insertion decomposition, sentence accuracy, real-time factors, paired-bootstrap significance between cells, and the go/no-go gate (pinned to `ctc_greedy` on test-clean).
4. **Efficiency** — parameter and analytic GFLOPs accounting per architecture stage.
5. **Interpretability** — see below.

Manual single-stage equivalents, for interactive use:

```bash
python scripts/train.py      --config configs/typeA_small_N1.yaml --resume auto
python scripts/decode.py     --config configs/typeA_small_N1.yaml --checkpoint checkpoints/typeA_small_N1/valid.wer.ave.pt
python scripts/score_wer.py  --config configs/typeA_small_N1.yaml --checkpoint checkpoints/typeA_small_N1/valid.wer.ave.pt
python scripts/run_interp.py --config configs/typeA_small_N1.yaml \
    --checkpoint checkpoints/typeA_small_N1/valid.wer.ave.pt \
    --modes boundaries,probes,robustness,emergence
```

Multi-GPU scaling is **config-only**: the same YAML runs on 1–8 GPUs; `batch_bins` is the per-GPU knob, and the dev sets are evaluated unsharded so metrics are world-size-invariant.

## 6. The interpretability suite

This is the scientific core — the part that asks *what* the model learned, not just how well it transcribes:

- **Boundary alignment.** Learned chunk boundaries vs. MFA phone/word boundaries: precision/recall/F1 at ±20 ms and Räsänen's R-value (which penalizes boundary-spraying), each reported against a **matched-count random-boundary floor** — the phone-tier floor is high (F1 ≈ 0.33), so every phone plot carries its floor.
- **Linear probes.** Frozen frame and chunk representations → phone identity (39 classes), phone manner class (7), and word identity (top-500, kept-fraction reported). Probes are L2-regularized multinomial logistic regressions; a GPU LBFGS backend fits the exact sklearn objective ~135× faster (parity verified to ≤2×10⁻⁵ in predicted probabilities).
- **Robustness.** Boundaries recollected under additive noise (20/10/5/0 dB SNR), speed change (0.9×/1.1×), and inserted silence, then compared against time-transformed clean boundaries and ground truth. A dedicated statistic counts boundaries appearing *inside* inserted pure silence — the acoustic-artefact detector.
- **Emergence (H4).** Boundary-F1 and probe accuracy recomputed for **every retained epoch checkpoint**, yielding curves of structure-vs-training-time to compare against the WER curve.

Hard safeguards are built into the drivers: probe train/test utterance disjointness is asserted on the ids actually consumed (contamination silently inflates accuracy by +0.77 on a random encoder); true audio durations are required for the random floor; partial boundary collections raise rather than silently biasing corpus metrics.

## 7. Engineering guarantees

- **Everything is logged.** Every metric (loss components, learning rate, gradient norm, throughput, per-interval GPU memory, per-stage keep-fractions, CTC-infeasibility counts, OOM skips, per-split dev WER/CER) streams to TensorBoard and an append-only JSONL; every artifact embeds full provenance (git commit, resolved config, data fingerprints, environment, batch math).
- **Fault tolerance.** DDP-safe OOM handling (a group flag keeps NCCL collectives matched when a rank skips a batch), a CUDA preflight that requeues off nodes with broken GPU state, shared-memory NCCL transport (a P2P transport deadlock was reproduced and pinned), atomic checkpoint writes, and requeue-safe stage markers.
- **Verification discipline.** Every module ships with unit tests plus an adversarial verification pass against independent oracles: the log-mel frontend is bit-identical to a from-scratch STFT reference; the boundary matcher equals brute-force maximum bipartite matching on 19k+ cases; the dechunk path matches the H-Net equations; WER/bootstrap/gate arithmetic is re-derived independently. 446 tests, zero warnings.

## 8. Status

The first two cells are training: **A·Small·N=1** (the gate run / no-chunk control) and **A·Small·N=2** (the first cell with active dynamic chunking) — 4 GPUs each, with their full evaluation pipelines queued behind them. Remaining: the other 20 grid cells, then figures and analysis.

## 9. References

Key sources (full list of 33 in the plan): Hwang, Wang & Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv:2507.07955) · Dao & Gu, *Transformers are SSMs* / Mamba-2 (ICML 2024) · Panayotov et al., *LibriSpeech* (ICASSP 2015) · Watanabe et al., *Hybrid CTC/Attention* (IEEE JSTSP 2017) · Yao et al., *Zipformer* (ICLR 2024) · McAuliffe et al., *Montreal Forced Aligner* (Interspeech 2017) · Räsänen et al., boundary evaluation and the R-value (Interspeech 2009).
