# DC-ASR — Interpretable Hierarchical Speech Recognition via Dynamic Chunking

A **Mamba–H-Net ASR** encoder that *learns* its own acoustic hierarchy end-to-end, instead of using a hand-designed frame-rate schedule. Two architecture types (1-stage and 2-stage), an iso-compression sweep, CTC/AED/joint decoding with and without an LM, on **LibriSpeech-960h**, benchmarked against published Conformer / Zipformer / E-Branchformer numbers.

> Full design: [`docs/DC-ASR_experimental_plan.md`](docs/DC-ASR_experimental_plan.md) (formal plan + experimental protocol + 33 references)
> Idea narrative: [`docs/DC-ASR_research_proposal.md`](docs/DC-ASR_research_proposal.md)

## The one-paragraph idea
Speech is hierarchical and variable-rate, but mainstream ASR encoders run at a fixed frame rate. DC-ASR inserts **H-Net dynamic chunking** (a differentiable, content-based boundary predictor) between stacks of **Mamba-2** blocks, so the encoder learns *where* to compress. We ask whether a **learned** hierarchy matches a **hand-designed** one (Zipformer) on WER — if it does, self-learned chunking is a better default: same accuracy, less hand-tuning, and interpretable learned units.

## Architecture (locked scope)
- **Type A — 1-stage:** `Mamba → H-Net → Mamba`
- **Type B — 2-stage:** `Mamba → H-Net → Mamba → H-Net → Mamba`
- Each in **Large** and **Small** variants (4 base encoders).
- **Compression level N ∈ {1,2,3,4}** = overall downsampling factor (frames kept = 1/N: 100/50/33/25%). Type B splits N across its two blocks at per-block factor √N so the two types are compared at **matched overall compression** (iso-compression). **N=1 = no-chunk control** (pure Mamba).
- **Decoders:** CTC, AED, and joint CTC+AED — three read-outs of **one hybrid-trained model** — under greedy and beam (B=10), each with and without an external LM.
- **Data:** LibriSpeech-960h **only**.
- **Baselines:** cited from the literature (not re-trained); see the plan.

## First experiment (go/no-go)
Type A · Small · **N=1** · CTC head on LibriSpeech-960h. Purpose: validate the full data→feature→encoder→CTC→decode→WER pipeline and establish the **no-chunk reference WER** before any chunking is enabled. Then flip N=1→N=2 and check that learned boundaries beat fixed-2×-pooling on WER *and* align with forced-aligned phones above chance. Config: [`configs/typeA_small_N1_ctc.yaml`](configs/typeA_small_N1_ctc.yaml).

## Layout
```
docs/         design docs (plan, proposal)
configs/      YAML experiment configs
src/dcasr/    package: data / models / decoders / training / eval / interp / logging_utils
scripts/      entry points (train, decode, score_wer, run_mfa) + setup/ (env, git bootstrap)
tests/        unit tests (CUDA — run on a GPU node)
data/         symlink -> /data/user_data/anshulk/hnet-asr/data        (corpora)
experiments/  symlink -> /data/user_data/anshulk/hnet-asr/experiments (run outputs)
logs/         symlink -> /data/user_data/anshulk/hnet-asr/logs        (process logs)
checkpoints/, features/, manifests/, alignments/  — same pattern
```

**Storage rule (Babel):** all heavy data — corpora, features, checkpoints, run
outputs, logs, conda envs, pip/HF caches — lives on the data partition
(`/data/user_data/anshulk/`); only code lives in `/home`. The repo reaches the
data side through the symlinks above, so one VS Code session sees everything.

**Logging rule:** every module gets its logger via
`dcasr.logging_utils.get_logger(__name__)`; every entry point calls
`setup_logging(<name>)` once. Logs go to `logs/` (i.e. the data partition),
rotating 50 MB × 5.

## Setup (Babel, CUDA-only)
One-shot: `bash scripts/setup/setup_env_babel.sh`. What it does:
```bash
# env lives on /data via ~/.condarc envs_dirs, not /home
conda create -n hnet-asr python=3.10 -y && conda activate hnet-asr
pip install torch==2.12.1+cu129 torchaudio==2.11.0+cu129 --index-url https://download.pytorch.org/whl/cu129
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm   # CUDA kernels; --no-deps so pip can't swap torch for a CUDA-13 build
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner   # for interpretability alignments
```
Driver note: Babel L40S nodes run driver 575.x (CUDA ≤ 12.9) — use cu129 wheels,
never the default PyPI torch (it ships CUDA 13.0 builds).

## Status
**Training + decoding stack complete (code + 277 tests, GPU-validated; multi-GPU proven on 2×GPU).**
Implemented end-to-end: acoustic frontend + CMVN + adaptive SpecAugment (`data/features.py`),
H-Net dynamic-chunking core (`models/hnet_chunk.py`) and the **fixed-stride pooling control**
(`models/fixed_pool.py`, H2), BPE tokenizer (`data/tokenizer.py`), LibriSpeech dataset with
speed-perturb ×3 + DDP-aware bucketed loader (`data/librispeech.py`), Mamba-2 backbone
(`models/mamba_block.py`), the **Mamba–H-Net encoder** (`models/encoder.py`, Type A/B,
config-selectable chunker), **all decoders** — CTC greedy/prefix-beam, AED, joint CTC+AED
one-pass beam, external-LM shallow fusion (`decoders/`) — hybrid CTC/AED loss, the
config-driven resumable Trainer with keep-N-best + checkpoint retention (`training/`),
WER/CER/TER scoring (`eval/metrics.py`), run provenance (`provenance.py`), structured
metrics (TB + JSONL, `metrics_logger.py`), and the thin train entry (`scripts/train.py`,
single-GPU default, `torchrun` for 2/4/8). **Artifacts:** LibriSpeech-960h extracted &
verified; BPE tokenizers (500 + 750); manifests + global CMVN (345M frames).
Every module carries unit tests + a real-audio GPU smoke + an independent adversarial
verification pass.

**Standing design decisions:** bidirectional encoder, **not params-matched to Zipformer**
(Zipformer/Conformer are WER references); runs on 1×48 GB GPU by default, scales via config
only (DDP, constant global batch — divide `batch_bins` by the GPU count); epoch-boundary
exact-resume checkpoints; hybrid λ_ctc=0.3/aed=0.7 for grid runs; ratio-loss β=0.03 at N≥2.

**Remaining:** external LM (810M-word corpus) → `decode.py`/`score_wer.py` (decode matrix ±LM)
→ efficiency (RTF/GFLOPs) → MFA + interpretability suite (boundary F1, probes, emergence)
→ grid configs + figures. Then the run phase: go/no-go (Type A · Small · N=1, test-clean
WER < 12) → Small screening → Large promotion → full decode matrix → interp. Hypotheses
(H1–H5) and protocol in the plan.
