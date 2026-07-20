# H-Net Mamba ASR - Interpretable Hierarchical Speech Recognition via Dynamic Chunking

This project asks one question: **can a speech recognizer learn its own acoustic hierarchy, meaning where phones and words begin and end, purely from the task of transcribing speech?**

Most speech recognition encoders process audio at fixed frame rates that a human designer chose. Conformer runs at one rate throughout. Zipformer uses a hand-tuned schedule of rates. Speech itself is not like that: phones last tens of milliseconds, words last hundreds, and the amount of information in the signal changes from moment to moment. This project removes the hand-designed schedule and replaces it with **H-Net dynamic chunking** (Hwang, Wang and Gu, arXiv:2507.07955), a differentiable boundary predictor placed between stacks of **Mamba-2** state-space blocks. The encoder itself decides where one unit ends and the next begins, and it compresses the sequence at exactly those learned boundaries.

If the learned hierarchy matches a hand-designed one on word error rate, then learned chunking is the better default: similar accuracy, less manual tuning, and, uniquely, **interpretable learned units** whose boundaries can be checked against linguistic ground truth from forced alignment.

Full experimental design: [`docs/experimental_plan.md`](docs/experimental_plan.md). Idea narrative: [`docs/research_proposal.md`](docs/research_proposal.md). Day-to-day experimental history, decisions, and verification verdicts live in `runlog.md`, an append-only lab notebook kept local to the working machine.

---

## 1. Architecture

```
      80-dim log-mel @ 100 Hz
              |
        [ Conv subsampling x4 ]     -> 25 Hz frame sequence
              |
   +---------------------------------------------+
   |  ENCODER stage: 4 x Mamba-2 blocks           |
   |   [ Router ]  p_t = 0.5 (1 - cos(q_t,k_t-1)) |
   |   [ Downsample ] keep frames with p >= 0.5   |
   +---------------------------------------------+
              |   (Type B repeats this stage once more)
   +----------------------+
   |  MAIN: 12 x Mamba-2   |   <- runs at the compressed rate
   +----------------------+
   +---------------------------------------------+
   |  DECODER stage: chunk-level EMA smoothing -> |
   |  upsample (gather) -> confidence STE ->      |
   |  4 x Mamba-2 blocks                          |
   +---------------------------------------------+
              |
     [ CTC head ]  +  [ 6-layer Transformer attention head ]
```

- **Router.** A boundary is declared where two consecutive frames stop resembling each other, measured by cosine similarity of learned projections. Gradients flow through a confidence-weighted straight-through estimator. A ratio loss (weight 0.03) steers the average keep-fraction toward the 1/N target. The upsampling path follows the H-Net paper exactly (its equations 5, 8, and 9: an exponential moving average over the compressed sequence using downsampled probabilities, then upsampling, then the straight-through scaling). This path is verified against an independent implementation of the equations.
- **Two types at matched compression.** Type A compresses once by a factor of N. Type B compresses twice by sqrt(N) each time, so the two types always have the same overall compression and any difference between them comes from the staging, not the rate. N is 1, 2, 3, or 4. **N = 1 is the no-chunk control**, a pure Mamba encoder.
- **One model, three decoders.** Each experiment trains a single hybrid model (loss = 0.3 CTC + 0.7 attention with label smoothing). CTC, attention, and joint CTC-plus-attention decoding are three read-outs of that one model, each run with and without an external Transformer language model (shallow fusion). That gives a 7-cell decode matrix per run.
- **Sizes.** Small is 78.9 M parameters (61.7 M encoder, 16.9 M attention head, 0.2 M CTC head). Large follows the plan. The vocabulary is 500 BPE units (750 as an ablation), with the CTC blank appended.
- **Data.** LibriSpeech-960h only, with 3-way speed perturbation and SpecAugment. Baselines are **cited from the literature, not re-trained**, so the entire compute budget goes to this project's own grid.

**The grid: 22 training runs.** Sixteen hybrid cells ({A, B} x N{1-4} x {Small, Large}), four fixed-stride pooling controls, and a 750-vocabulary pair. The first cell (Type A, Small, N=1) doubles as the go or no-go gate: its CTC greedy word error rate on test-clean must beat 12 percent.

## 2. Repository structure

```
.
|-- README.md
|-- pyproject.toml                  package metadata, pytest configuration
|-- requirements.txt                python dependencies
|-- configs/
|   |-- typeA_small_N1.yaml         grid cell 1: Type A, Small, N=1 (the gate run)
|   |-- typeA_small_N2.yaml         grid cell 2: Type A, Small, N=2 (first active chunking)
|   |-- typeA_small_N1_ctc.yaml     early CTC-only template (superseded, kept for smokes)
|   |-- lm_transformer_500.yaml     external language model, 500-unit vocabulary
|   `-- lm_transformer_750.yaml     external language model, 750-unit vocabulary
|-- docs/
|   |-- experimental_plan.md        the design authority: protocol, grid, references
|   `-- research_proposal.md        the idea narrative
|-- scripts/
|   |-- build_manifests.py          scan LibriSpeech into JSONL manifests
|   |-- build_tokenizer.py          train the SentencePiece BPE tokenizers
|   |-- compute_cmvn.py             global feature normalization statistics
|   |-- train.py                    training entry point (single or multi GPU)
|   |-- train_lm.py                 language model training entry point
|   |-- decode.py                   the 7-cell decode matrix over all eval splits
|   |-- score_wer.py                WER and CER tables, significance tests, the gate
|   |-- efficiency.py               parameter and GFLOPs accounting
|   |-- run_mfa.py                  forced alignment: phone and word ground truth
|   |-- run_interp.py               the interpretability suite driver
|   |-- analysis/vocab_analysis.py  offline vocabulary-size study
|   |-- setup/setup_env_babel.sh    one-shot environment setup
|   |-- setup/init_git.sh           one-time repository bootstrap
|   `-- slurm/
|       |-- run_cell_e2e_4gpu.sh    parameterized end-to-end pipeline (any cell)
|       |-- run_asn1_e2e_4gpu.sh    the same pipeline pinned to the gate cell
|       |-- smoke_2gpu.sh           2-GPU distributed-training smoke test
|       `-- train_lm_{500,750}.sh   language model training jobs
|-- src/dcasr/                      the python package
|   |-- data/
|   |   |-- features.py             log-mel frontend, CMVN, SpecAugment
|   |   |-- librispeech.py          dataset, manifests, bucketed batch sampler
|   |   |-- tokenizer.py            SentencePiece wrapper with fixed special ids
|   |   `-- lm_text.py              language model text corpus handling
|   |-- models/
|   |   |-- encoder.py              the full encoder: Mamba stacks + chunking stages
|   |   |-- hnet_chunk.py           router, ratio loss, chunk and dechunk (the core)
|   |   |-- fixed_pool.py           fixed-stride pooling control baseline
|   |   `-- mamba_block.py          Mamba-2 block wrapper
|   |-- decoders/
|   |   |-- ctc.py                  CTC head, greedy and prefix beam search
|   |   |-- aed.py                  attention decoder head
|   |   |-- joint.py                joint CTC-plus-attention beam search
|   |   `-- lm_fusion.py            shallow fusion with the external language model
|   |-- training/
|   |   |-- trainer.py              resumable, distributed-safe training loop
|   |   `-- loss.py                 hybrid loss combination
|   |-- tasks/
|   |   |-- asr_task.py             model assembly from config (registries)
|   |   |-- build.py                frontend, data, and trainer builders
|   |   |-- decode_task.py          decode matrix expansion and execution
|   |   `-- lm_task.py              language model task wiring
|   |-- eval/
|   |   |-- metrics.py              edit-distance metrics
|   |   |-- score.py                scoring, bootstrap significance, the gate
|   |   `-- efficiency.py           analytic parameter and GFLOPs accounting
|   |-- interp/
|   |   |-- alignments.py           forced-alignment corpus prep and parsing
|   |   |-- boundary_align.py       boundary scoring against ground truth
|   |   |-- probes.py               linear probes (sklearn and GPU backends)
|   |   `-- driver.py               orchestration: reports, robustness, emergence
|   |-- logging_utils.py            rank-aware logging
|   |-- metrics_logger.py           TensorBoard plus JSONL metrics sink
|   |-- optim.py                    optimizer and scheduler registries
|   `-- provenance.py               config, git, environment, and data fingerprints
|-- tests/                          446 tests across 29 files (pytest)
`-- data/ experiments/ checkpoints/ features/ manifests/ alignments/ logs/
                                    symlinks to the cluster data partition
```

Everything is **config-driven**: components resolve through name-plus-configuration registries, so a new encoder or scheduler is one registry entry plus a YAML key, and the trainer is model-agnostic. Code lives on the home partition; all heavy data lives on the data partition behind the symlinks.

## 3. Setup

Requirements: Linux, an NVIDIA GPU (Ampere or newer, bf16), a CUDA 12.9 compatible driver, Python 3.10, and the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) in its own conda environment for the ground-truth alignments.

```bash
conda create -n hnet-asr python=3.10 -y && conda activate hnet-asr
pip install torch==2.12.1+cu129 torchaudio==2.11.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129
pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm
pip install -r requirements.txt
# separate environment for forced alignment:
conda create -n mfa -c conda-forge montreal-forced-aligner -y
```

The `--no-deps` flag on the Mamba kernels is deliberate: it stops pip from silently replacing torch with a build for a newer CUDA than the driver supports.

## 4. Reproduction, end to end

**Step 1: data preparation** (run once):

```bash
python scripts/build_manifests.py      # LibriSpeech -> {id, audio, text, frames} manifests
python scripts/build_tokenizer.py      # BPE-500 and BPE-750 tokenizers
python scripts/compute_cmvn.py         # global normalization statistics
python scripts/run_mfa.py              # phone and word ground truth + a 10 h probe subset
python scripts/train_lm.py --config configs/lm_transformer_500.yaml
```

**Step 2: train and evaluate one cell.** A single SLURM submission runs the entire experiment, from training through every evaluation, and survives preemption, bad nodes, and time limits:

```bash
sbatch -J hnet_asn1 scripts/slurm/run_cell_e2e_4gpu.sh \
    configs/typeA_small_N1.yaml typeA_small_N1
```

The five pipeline stages, each skipped automatically on requeue when already complete:

1. **Train**: 4 GPUs with `torchrun`, bf16, AdamW with warmup then inverse-square-root decay, up to 200 epochs with early stopping once dev WER, CER, and loss have all plateaued. A checkpoint is written every 5 epochs and all of them are kept, because the emergence analysis needs them. Training resumes exactly after any interruption: weights, optimizer, scheduler, scaler, random state, and metric history are all restored.
2. **Decode**: all 7 cells on all 4 evaluation splits, one split per GPU in parallel.
3. **Score**: WER **and CER** with substitution, deletion, and insertion breakdowns, sentence accuracy, real-time factors, paired-bootstrap significance, and the go or no-go gate (pinned to CTC greedy on test-clean).
4. **Efficiency**: parameter and analytic GFLOPs accounting per architecture stage.
5. **Interpretability**: described below.

Manual equivalents of each stage exist for interactive use (`scripts/train.py`, `scripts/decode.py`, `scripts/score_wer.py`, `scripts/run_interp.py`), all reading the same YAML. Multi-GPU scaling is config-only: `batch_bins` is the per-GPU batch knob, and the dev sets are evaluated unsharded so metrics do not depend on the GPU count.

## 5. The interpretability suite

This is the scientific core, the part that asks what the model learned rather than only how well it transcribes:

- **Boundary alignment.** Learned chunk boundaries against forced-alignment phone and word boundaries: precision, recall, F1 within 20 ms, and the R-value (which penalizes boundary spraying), each reported against a matched-count random floor. The phone floor is high (F1 near 0.33), so every phone plot carries its floor.
- **Linear probes.** Frozen frame and chunk vectors are probed for phone identity (39 classes), phone manner class (7 classes), and word identity (top 500 words, with the kept fraction reported). Probes are L2-regularized logistic regressions. A GPU backend fits the exact same convex objective about 135 times faster than the CPU reference, with parity verified to 2e-5 in predicted probabilities.
- **Robustness.** Boundaries are collected again under added noise (20, 10, 5, 0 dB), speed change (0.9x, 1.1x), and inserted silence, then compared with time-transformed clean boundaries and ground truth. A dedicated statistic counts boundaries that appear inside inserted pure silence, which is the acoustic-artifact detector.
- **Emergence.** Boundary F1 and probe accuracy are recomputed for every retained checkpoint, giving curves of structure against training time to compare with the word error rate curve.

Hard safeguards are built in: probe train and test utterance sets are checked for overlap on the utterances actually consumed (contamination silently inflates probe accuracy), true audio durations are required for the random floor, and partial boundary collections raise an error instead of silently biasing the corpus metric.

## 6. Engineering guarantees

- **Everything is logged.** Every metric streams to TensorBoard and an append-only JSONL file: loss components, learning rate, gradient norm, throughput, per-interval GPU memory, per-stage keep fractions, CTC feasibility counts, out-of-memory skips, and per-split dev WER and CER. Every artifact embeds full provenance: git commit, resolved config, data fingerprints, environment, and batch arithmetic.
- **Fault tolerance.** Distributed-safe out-of-memory handling (a group flag keeps collective operations matched when one rank must skip a batch), a CUDA preflight that moves the job off nodes with broken GPU state, shared-memory NCCL transport (a peer-to-peer transport deadlock was reproduced and pinned), atomic checkpoint writes, and requeue-safe stage markers.
- **Verification discipline.** Every module ships with unit tests plus an adversarial verification pass against independent oracles: the log-mel frontend is bit-identical to a from-scratch STFT reference, the boundary matcher equals brute-force optimal matching on more than 19,000 cases, the dechunk path matches the H-Net equations, and the WER, bootstrap, and gate arithmetic is re-derived independently. 446 tests, zero warnings.

## 7. Status

The first two cells are training in parallel on 4 GPUs each: Type A Small N=1 (the gate run and no-chunk control) and Type A Small N=2 (the first cell with active dynamic chunking). The remaining 20 grid cells follow, then figures and analysis.

## 8. References

Key sources (the full list of 33 is in the plan): Hwang, Wang and Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (arXiv:2507.07955); Dao and Gu, Mamba-2 (ICML 2024); Panayotov et al., *LibriSpeech* (ICASSP 2015); Watanabe et al., *Hybrid CTC/Attention* (IEEE JSTSP 2017); Yao et al., *Zipformer* (ICLR 2024); McAuliffe et al., *Montreal Forced Aligner* (Interspeech 2017); Rasanen et al., boundary evaluation and the R-value (Interspeech 2009).
