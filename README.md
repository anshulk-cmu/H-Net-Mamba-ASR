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
src/dcasr/    package: data / models / decoders / training / interp
scripts/      entry points (train, decode, score, align, probe)
tests/        unit tests
experiments/  run outputs (git-ignored)
```

## Setup (target: Babel, 1 GPU)
```bash
conda create -n dcasr python=3.10 -y && conda activate dcasr
pip install -r requirements.txt          # torch/mamba-ssm need a matching CUDA toolchain
conda install -c conda-forge montreal-forced-aligner   # for interpretability alignments
```

## Status
Scaffold only — planning docs are complete; model/data/training code is stubbed and under active development. See the plan for the full grid and hypotheses (H1–H5).
