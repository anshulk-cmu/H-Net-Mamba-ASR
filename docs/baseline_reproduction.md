# Baseline Reproduction: Complete Analysis

**H-Mamba ASR Project — Anshul Kumar**
**Carnegie Mellon University, April 2026**
**Advisor: Prof. Shinji Watanabe**

This document records every decision, every number, every piece of architecture, and every
result from the baseline reproduction stage. It is the truth document for this stage.
The H-Mamba (Dynamic Chunking) experiments are documented separately.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [Why These Baselines](#2-why-these-baselines)
3. [The Dataset](#3-the-dataset)
4. [The Three Encoder Architectures](#4-the-three-encoder-architectures)
5. [The Two Decoder Paradigms](#5-the-two-decoder-paradigms)
6. [The Two Model Scales](#6-the-two-model-scales)
7. [The Eight Experiments](#7-the-eight-experiments)
8. [Shared Training Infrastructure](#8-shared-training-infrastructure)
9. [Feature Extraction and Augmentation](#9-feature-extraction-and-augmentation)
10. [Optimizer and Scheduler](#10-optimizer-and-scheduler)
11. [Loss Functions](#11-loss-functions)
12. [Decoding and Evaluation](#12-decoding-and-evaluation)
13. [Small Model Configurations](#13-small-model-configurations)
14. [Large Model Configurations](#14-large-model-configurations)
15. [CTC Model Configurations](#15-ctc-model-configurations)
16. [SLURM Job Configuration](#16-slurm-job-configuration)
17. [Training Runs and Timeline](#17-training-runs-and-timeline)
18. [Results: Word Error Rate](#18-results-word-error-rate)
19. [Results Analysis](#19-results-analysis)
20. [Reproducibility](#20-reproducibility)
21. [What This Stage Does NOT Do](#21-what-this-stage-does-not-do)
22. [What This Stage Establishes](#22-what-this-stage-establishes)

---

## 1. Purpose of This Stage

We are studying whether speech recognition can learn to skip — specifically, whether a
learned Dynamic Chunking (DC) mechanism inserted mid-network into a ConMamba encoder can
compress acoustic representations at inference time without degrading Word Error Rate (WER).
The hypothesis is that not all frames carry equal information: silence, steady-state vowels,
and repeated feature patterns can be compressed, while transients and boundaries must be
preserved.

Before testing that hypothesis, we need baselines. Specifically:

1. **Conformer baselines** — the dominant encoder architecture in modern ASR. Any DC
   mechanism must be compared against the best Conformer we can train.
2. **ConMamba baselines** — the encoder we will actually modify with DC. We need to know
   how ConMamba performs without DC so that any WER change from DC can be attributed to
   the chunking mechanism itself, not to the underlying encoder.
3. **ConMambaMamba baselines** — a variant where the Transformer decoder is replaced with
   a Mamba decoder. This tests whether the SSM paradigm works end-to-end.
4. **CTC baselines** — encoder-only models that test whether the architectures work
   without an autoregressive decoder, providing a simpler (and faster) evaluation signal.

This stage produces WER numbers for all eight configurations. The downstream H-Mamba
experiments consume these as reference points. This stage does not involve any Dynamic
Chunking code.

---

## 2. Why These Baselines

### Why Conformer

The Conformer (Gulati et al., 2020) combines self-attention with depthwise convolutions
in a macaron-style feed-forward sandwich. It has been the dominant ASR encoder since its
introduction. On LibriSpeech, Conformer-based systems hold state-of-the-art or near-SOTA
results across multiple scales.

We include Conformer as an upper-bound reference. If ConMamba + DC achieves WER within
some tolerance of Conformer, that is a meaningful result: it means we can get comparable
accuracy with learned compression.

### Why ConMamba

ConMamba (Jiang, 2024) replaces the self-attention module in each Conformer layer with a
bidirectional Mamba (BiMamba) State Space Model. The layer structure becomes:

```
FFN(half-step) → LayerNorm → BiMamba → Residual → ConvModule → FFN(half-step) → LayerNorm
```

This is the encoder we will modify with DC. The ConMamba encoder processes the full
sequence through all layers without any compression. Our H-Mamba system will split this
encoder at a configurable layer boundary, insert a routing module that identifies
acoustic boundaries, compress the sequence, process the remaining layers on the
compressed sequence, and then decompress before the decoder.

### Why ConMambaMamba

ConMambaMamba pairs the ConMamba encoder with a Mamba-based decoder instead of the
standard Transformer decoder. The Mamba decoder uses:

- **Self-Mamba**: unidirectional Mamba over the target token sequence
- **Cross-Mamba**: concatenates encoder memory with target query, runs through Mamba,
  and takes the last `len(query)` outputs as cross-attention equivalent
- **Positional FFN**: standard feed-forward network

This tests the full-SSM hypothesis: if both encoder and decoder use SSMs instead of
attention, does the system still achieve competitive WER? This matters because Mamba's
linear-time complexity could provide significant inference speed advantages.

### Why CTC

CTC (Connectionist Temporal Classification) models use only the encoder — no decoder,
no beam search with language model at training time. They provide:

- A simpler training signal (faster convergence feedback)
- A test of encoder quality independent of decoder strength
- A comparison point for how much the Transformer decoder + LM contributes

We train CTC variants only at the large scale (256 d_model, 18 layers) because CTC
models need more capacity to compensate for the lack of an autoregressive decoder.

### The resulting 8-experiment grid

| # | Encoder | Decoder | Scale | Name |
|---|---------|---------|-------|------|
| 1 | Conformer | Transformer | Small | conformer_small_S2S |
| 2 | ConMamba | Transformer | Small | conmamba_small_S2S |
| 3 | ConMamba | Mamba | Small | conmambamamba_small_S2S |
| 4 | Conformer | Transformer | Large | conformer_large_S2S |
| 5 | ConMamba | Transformer | Large | conmamba_large_S2S |
| 6 | ConMamba | Mamba | Large | conmambamamba_large_S2S |
| 7 | ConMamba | CTC | Large | conmamba_large_CTC |

---

## 3. The Dataset

### LibriSpeech 960h

We use LibriSpeech (Panayotov et al., 2015), the standard benchmark for English ASR
research. The dataset is derived from read audiobooks from the LibriVox project.

| Split | Hours | Purpose |
|-------|-------|---------|
| train-clean-100 | 100.6 | Clean training data |
| train-clean-360 | 363.6 | Additional clean training data |
| train-other-500 | 496.7 | Noisier training data |
| **train (merged)** | **960.9** | **All training data combined** |
| dev-clean | 5.4 | Validation (clean) |
| dev-other | 5.3 | Validation (noisy) — not used for early stopping |
| test-clean | 5.4 | Final evaluation (clean conditions) |
| test-other | 5.1 | Final evaluation (noisy conditions) |

All audio is 16 kHz, 16-bit, mono FLAC. The three training splits are merged into a
single training CSV by SpeechBrain's `librispeech_prepare.py`. All models train on the
full 960 hours.

### Data location

- Raw LibriSpeech data: `/data/user_data/anshulk/hnet_asr/LibriSpeech`
- SentencePiece tokenizer (S2S): 5000 unigram tokens, trained on LibriSpeech transcripts
- Character tokenizer (CTC): 31 characters (26 letters + space + apostrophe + special tokens)

### Why LibriSpeech

LibriSpeech is the standard benchmark. Every Conformer and Mamba ASR paper reports
LibriSpeech numbers. Using it means our results are directly comparable to the
literature. The 960-hour training set is large enough to train models at our scale
(13M–123M parameters) without underfitting.

---

## 4. The Three Encoder Architectures

### 4.1 Conformer Encoder

The Conformer encoder (`modules/Conformer.py`) uses a macaron-style architecture where
each layer contains:

```
x = x + 0.5 * FFN_1(x)           # First half-step feed-forward
x = x + MHSA(x)                   # Multi-head self-attention (RelPosMHAXL)
x = x + ConvModule(x)             # Depthwise separable convolution
x = LayerNorm(x + 0.5 * FFN_2(x)) # Second half-step feed-forward + norm
```

**RelPosMHAXL** (Relative Positional Multi-Head Attention with XL-style encoding) is
the attention mechanism. It uses relative position encodings rather than absolute
sinusoidal positions, which helps the model generalize to sequences longer than those
seen during training.

**ConvModule** structure:
1. LayerNorm
2. Pointwise Conv1d (expand to 2x channels) + GLU activation
3. Depthwise Conv1d (kernel_size=31, groups=input_size)
4. LayerNorm + Swish activation
5. Pointwise Linear (project back) + Dropout

The depthwise convolution with kernel_size=31 gives each frame a receptive field of
31 frames (~310ms at the subsampled rate), capturing local acoustic patterns that
attention alone handles poorly.

### 4.2 ConMamba Encoder

The ConMamba encoder (`modules/Conmamba.py`) replaces MHSA with BiMamba:

```
x = x + 0.5 * FFN_1(x)           # First half-step feed-forward
x = x + BiMamba(LayerNorm(x))     # Bidirectional Mamba SSM (replaces attention)
x = x + ConvModule(x)             # Same depthwise separable convolution
x = LayerNorm(x + 0.5 * FFN_2(x)) # Second half-step feed-forward + norm
```

**BiMamba** (bidirectional Mamba, `modules/mamba/bimamba.py`, type='v2') processes the
sequence in both forward and backward directions. Mamba is a Structured State Space
Model (SSM) with input-dependent (selective) parameters:

- **d_state=16**: SSM hidden state dimension
- **expand=2**: internal dimension = 2 * d_model
- **d_conv=4**: local convolution kernel in the SSM input projection

Key difference from attention: Mamba processes sequences in O(N) time and memory
(linear in sequence length), compared to O(N^2) for self-attention. For long utterances,
this is a significant computational advantage.

The ConvModule is identical to Conformer's — same pointwise-depthwise-pointwise
structure with kernel_size=31. This means the local feature extraction capability
is shared; only the global context mechanism differs (BiMamba vs. attention).

### 4.3 Architecture comparison

| Component | Conformer | ConMamba |
|-----------|-----------|---------|
| Global context | RelPosMHAXL (O(N^2)) | BiMamba SSM (O(N)) |
| Local context | ConvModule (k=31) | ConvModule (k=31) |
| FFN structure | Macaron (2 half-step) | Macaron (2 half-step) |
| Positional encoding | Relative (in attention) | Implicit (in SSM dynamics) |
| Dropout in SSM | N/A | Not used (Mamba has internal gating) |
| Bidirectional | Via attention over full seq | Via BiMamba v2 |

---

## 5. The Two Decoder Paradigms

### 5.1 Sequence-to-Sequence (S2S) with Transformer Decoder

The S2S models use a standard Transformer decoder with:

- Multi-head self-attention over target tokens (causal mask)
- Multi-head cross-attention over encoder output
- Positional feed-forward network
- NormalizedEmbedding for token embeddings

At training time, both CTC loss and sequence-to-sequence (KL-divergence) loss are
computed. The joint loss is:

```
loss = ctc_weight * CTC_loss + (1 - ctc_weight) * seq2seq_loss
```

At decoding time, joint CTC-attention beam search with an external Transformer language
model is used.

### 5.2 Sequence-to-Sequence with Mamba Decoder

The Mamba decoder (`MambaDecoderLayer` in `modules/Conmamba.py`) replaces the Transformer
decoder's attention mechanisms with Mamba SSMs:

- **Self-Mamba**: unidirectional Mamba over target tokens (replaces causal self-attention)
- **Cross-Mamba**: concatenates `[encoder_memory, target_query]`, runs through Mamba, and
  takes the last `len(query)` outputs. This replaces cross-attention. The intuition is
  that Mamba's recurrent processing over `[memory; query]` lets the query tokens attend
  to the encoder output through the SSM's hidden state.
- **Positional FFN**: identical to Transformer decoder

The Mamba decoder is always unidirectional (causal) regardless of the `bidirectional`
config flag, because autoregressive generation requires causal processing.

### 5.3 CTC (Encoder-Only)

CTC models have no decoder. The encoder output is projected directly to the vocabulary
size (31 characters) through a linear layer. CTC loss handles the alignment between
the encoder output sequence and the target character sequence using the CTC forward-
backward algorithm.

At test time, CTC beam search (beam_size=100) is used. No external language model is
applied during CTC decoding in our setup.

---

## 6. The Two Model Scales

### Small (S)

| Parameter | Value |
|-----------|-------|
| d_model | 144 |
| nhead | 4 |
| num_encoder_layers | 12 |
| num_decoder_layers | 4 |
| d_ffn | 1024 |
| Trainable parameters | 13.3M (Conformer), 14.1M (ConMamba), 14.6M (ConMambaMamba) |

The small scale is designed for faster iteration. With ~14M parameters, these models
train in 3–14 hours on 2x A6000 GPUs. The d_model of 144 with 4 heads gives 36
dimensions per head, which is small but sufficient for learning acoustic representations
on LibriSpeech.

### Large (L)

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| nhead | 8 |
| num_encoder_layers | 12 (S2S) / 18 (CTC) |
| num_decoder_layers | 6 (S2S) / 0 (CTC) |
| d_ffn | 2048 |
| Trainable parameters | 109.1M (Conformer S2S), 115.2M (ConMamba S2S), 122.9M (ConMambaMamba S2S), 31.6M (ConMamba CTC) |

The large scale is the primary comparison point. d_model=512 with 8 heads gives 64
dimensions per head, standard for Transformer-based ASR. The CTC models use 18 encoder
layers (vs. 12 for S2S) to compensate for lacking a decoder — the encoder must do more
work to produce good CTC alignments.

### Why two scales

Running the same architecture at two scales tests whether findings are robust. If
ConMamba beats Conformer at small scale but not at large, the result is scale-dependent
and less interesting. If the ranking is consistent across scales, we have stronger
evidence.

---

## 7. The Eight Experiments

### Complete parameter table

| Config | Encoder | Decoder | d_model | nhead | Enc Layers | Dec Layers | d_ffn | Params | Vocab | Seed |
|--------|---------|---------|---------|-------|------------|------------|-------|--------|-------|------|
| conformer_small_S2S | Conformer | Transformer | 144 | 4 | 12 | 4 | 1024 | 13.3M | 5000 | 7775 |
| conmamba_small_S2S | ConMamba | Transformer | 144 | 4* | 12 | 4 | 1024 | 14.1M | 5000 | 7775 |
| conmambamamba_small_S2S | ConMamba | Mamba | 144 | 4* | 12 | 4 | 1024 | 14.6M | 5000 | 7775 |
| conformer_large_S2S | Conformer | Transformer | 512 | 8 | 12 | 6 | 2048 | 109.1M | 5000 | 3407 |
| conmamba_large_S2S | ConMamba | Transformer | 512 | 8* | 12 | 6 | 2048 | 115.2M | 5000 | 3407 |
| conmambamamba_large_S2S | ConMamba | Mamba | 512 | 8* | 12 | 6 | 2048 | 122.9M | 5000 | 3407 |
| conmamba_large_CTC | ConMamba | CTC | 256 | 4* | 18 | 0 | 1024 | 31.6M | 31 | 3402 |

\* nhead is configured but unused — ConMamba/BiMamba does not use multi-head attention.

### Why the parameter counts differ

ConMamba has slightly more parameters than Conformer because BiMamba's internal
projections (in_proj, out_proj, x_proj, dt_proj, plus the conv1d) have a different
parameter count than RelPosMHAXL's Q/K/V projections. The difference is small (~6%
at both scales), so the comparison is fair.

ConMambaMamba adds parameters over ConMamba because the Mamba decoder's self-Mamba and
cross-Mamba modules add parameters that the Transformer decoder does not have (the
Transformer decoder reuses attention weights more efficiently through Q/K/V projections).

### Why different seeds

Small models use seed 7775 and large models use seed 3407. This is an artifact of the
original codebase from Jiang (2024). We kept the original seeds to enable exact
reproduction of published results. The CTC models use seed 3402.

---

## 8. Shared Training Infrastructure

### Framework

All models are trained using **SpeechBrain** (v1.0+), an open-source PyTorch-based
speech toolkit. SpeechBrain provides:

- The `Brain` class for training loop management
- `DynamicItemDataset` for efficient data loading
- `DynamicBatchSampler` for length-based dynamic batching
- Checkpoint management with top-K saving
- `EpochCounterWithStopper` for early stopping

### Distributed training

All experiments use **PyTorch DDP** (DistributedDataParallel) via `torchrun`:

```bash
torchrun --nproc_per_node=$NGPU --nnodes=1 --rdzv_endpoint=localhost:$PORT \
    train_S2S.py hparams/S2S/config.yaml --distributed_launch
```

Each GPU runs a separate process. Gradients are synchronized via all-reduce after each
backward pass. The effective batch size is:

```
effective_batch = per_gpu_batch * num_gpus * grad_accumulation_factor
```

### Convolutional frontend

All models share the same CNN frontend that subsamples the input features before the
encoder:

- 2 convolutional blocks
- Each block: Conv2d(kernel=3x3, stride=2x2) + BatchNorm + activation
- Total downsampling: 4x in time dimension
- Output dimension: 640 (from 80 mel bins * 2^3 channel expansion, then projected)
- A linear projection maps 640 → d_model

This frontend converts 80-dim mel filterbank features at ~100 frames/sec into d_model-dim
vectors at ~25 frames/sec.

### Precision

All S2S models and the ConMamba CTC model train in **bf16** (bfloat16) mixed precision.

### Early stopping

All models use `EpochCounterWithStopper`:

- S2S models: patience=20 epochs, warmup=50 epochs, monitor=ACC (higher is better)
- CTC models: patience=30 epochs, warmup=50 epochs, monitor=WER (lower is better)

In practice, all S2S models ran the full 150 epochs without early stopping triggering.
The CTC models are configured for 500 epochs but converge within ~200.

### Checkpoint averaging

At evaluation time, the best 10 checkpoints (by validation ACC for S2S, by validation
WER for CTC) are loaded and their parameters are averaged. This is a standard technique
in ASR that smooths out training noise and typically improves WER by 0.1–0.3% absolute.

---

## 9. Feature Extraction and Augmentation

### Mel filterbank features

| Parameter | Small Models | Large S2S Models | Large CTC Models |
|-----------|-------------|-----------------|-----------------|
| n_fft | 400 | 512 | 512 |
| n_mels | 80 | 80 | 80 |
| win_length | default (=n_fft) | 32 | 25 |
| sample_rate | 16000 | 16000 | 16000 |

All models use 80-dimensional log-mel filterbank features computed from 16 kHz audio.
The small models use a smaller FFT window (400 samples = 25ms) while large models use
512 samples (32ms). Features are globally normalized using running statistics.

### Speed perturbation

Training data is augmented with speed perturbation at three rates: 0.95, 1.0, and 1.05.
This effectively triples the training data by creating slower and faster versions of
each utterance. Speed perturbation is applied at the waveform level before feature
extraction.

### SpecAugment

SpecAugment (Park et al., 2019) is applied to the mel features during training:

- **Time masking**: random contiguous time steps are zeroed
- **Frequency masking**: random contiguous frequency bins are zeroed
- **Time warping**: slight temporal distortion

SpecAugment is only applied during training (not validation or test). It is controlled
by the `augmentation` module in each config file.

---

## 10. Optimizer and Scheduler

### Optimizer

| Parameter | Small Models | Large S2S Models | Large CTC Models |
|-----------|-------------|-----------------|-----------------|
| Optimizer | Adam | AdamW | AdamW |
| Betas | (0.9, 0.98) | (0.9, 0.98) | (0.9, 0.98) |
| Weight decay | 0 | 0 (AdamW default) | 0.0005 |
| Learning rate | 0.001 | 0.0008 | 0.001 |
| Max grad norm | 5.0 | 5.0 | 5.0 |

Small models use Adam (no weight decay). Large S2S models upgrade to AdamW. Large CTC
models use AdamW with explicit weight_decay=0.0005, which provides additional
regularization for the encoder-only architecture.

### Noam scheduler

All models use the **Noam learning rate schedule** (Vaswani et al., 2017):

```
lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

This schedule linearly warms up the learning rate for `warmup_steps`, then decays it
proportionally to the inverse square root of the step number.

| Config | Warmup Steps |
|--------|-------------|
| conformer_small_S2S | 3,125 |
| conmamba_small_S2S | 25,000 |
| conmambamamba_small_S2S | 25,000 |
| conformer_large_S2S | 30,000 |
| conmamba_large_S2S | 3,750 |
| conmambamamba_large_S2S | 3,750 |
| conmamba_large_CTC | 7,500 |

The warmup steps differ across configs because they were tuned per-architecture by Jiang
(2024). ConMamba small models use longer warmup (25,000 steps) because BiMamba's
selective SSM parameters need more gradual initialization. Conformer small uses shorter
warmup (3,125) because RelPosMHAXL attention stabilizes faster.

### Mamba-specific: update_norm_until_epoch

ConMamba models include `update_norm_until_epoch: 4`, which updates BatchNorm statistics
in the CNN frontend for the first 4 epochs of training. After epoch 4, BatchNorm
switches to eval mode (frozen statistics). This stabilizes training when the Mamba
layers' initial outputs have different statistics than what the frontend BatchNorm
was initialized for.

---

## 11. Loss Functions

### S2S models: Joint CTC + Attention

The S2S training loss is:

```
L = 0.3 * L_CTC + 0.7 * L_seq2seq
```

- **L_CTC**: CTC loss computed between encoder output (projected to vocab) and target
  token sequence. CTC handles the alignment problem (encoder output is longer than
  target tokens) using the CTC forward-backward algorithm.

- **L_seq2seq**: KL-divergence loss between the decoder's predicted token distribution
  and the target distribution (with optional label smoothing).

- **CTC weight = 0.3**: standard for joint CTC-attention ASR. The CTC branch acts as
  a regularizer that encourages monotonic alignment in the encoder output.

Label smoothing:
- Small models: 0.0 (no smoothing)
- Large models: 0.1 (distributes 10% probability mass uniformly)

### CTC models: CTC only

```
L = L_CTC
```

No sequence loss, no label smoothing. The CTC models rely entirely on the CTC objective.

---

## 12. Decoding and Evaluation

### S2S decoding

S2S models use **joint CTC-attention beam search** with an external language model:

| Parameter | Validation | Test |
|-----------|-----------|------|
| Beam size | 10 | 66 |
| CTC weight | 0.40 | 0.40 |
| LM weight | 0.60 | 0.60 |

The test beam size of 66 is large but standard in SpeechBrain LibriSpeech recipes. The
language model is a pre-trained Transformer LM (d_model=768, 12 heads, 12 layers) that
rescores hypotheses during beam search.

**No-LM evaluation**: All S2S models are also evaluated without the language model
(CTC weight=0.40, LM weight=0.0) to measure the acoustic model's standalone quality.
These results are stored in `no_lm/` subdirectories.

### CTC decoding

| Parameter | Validation | Test |
|-----------|-----------|------|
| Method | Greedy | Beam search |
| Beam size | — | 100 |
| beam_prune_logp | — | -12.0 |
| token_prune_min_logp | — | -1.2 |

CTC validation uses greedy decoding (argmax at each timestep) for speed. Test
evaluation uses beam search with beam_size=100 and pruning thresholds.

### Metrics

- **WER** (Word Error Rate): (substitutions + insertions + deletions) / total reference
  words. Primary metric for all models.
- **SER** (Sentence Error Rate): fraction of utterances with at least one error. Reported
  for CTC models.
- **ACC** (token accuracy): used internally by S2S models for checkpoint selection and
  early stopping.
- **CER** (Character Error Rate): reported for CTC models alongside WER.

Test sets:
- **test-clean**: 2,620 utterances, 52,576 words. Clean recording conditions.
- **test-other**: 2,939 utterances, 52,343 words. Noisier conditions (accented speakers,
  background noise, recording artifacts).

---

## 13. Small Model Configurations

### conformer_small_S2S

**Config**: `hparams/S2S/conformer_small.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | Conformer, 12 layers, d_model=144, nhead=4, d_ffn=1024 |
| Decoder | Transformer, 4 layers |
| Attention | RelPosMHAXL |
| Activation | GELU |
| Dropout | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 8 |
| Effective batch | 16 * 2 GPUs * 8 = 256 |
| Dynamic batching | max_batch_length_train=1050 |
| Epochs | 150 |
| LR / Warmup | 0.001 / 3,125 steps |
| Precision | bf16 |
| Parameters | 13.3M |

### conmamba_small_S2S

**Config**: `hparams/S2S/conmamba_small.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | ConMamba, 12 layers, d_model=144, d_ffn=1024, BiMamba(d_state=16, expand=2, d_conv=4) |
| Decoder | Transformer, 4 layers |
| Activation | GELU |
| Dropout | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 1 |
| Effective batch | 16 * 2 GPUs * 1 = 32 |
| Dynamic batching | max_batch_length_train=1050 |
| Epochs | 150 |
| LR / Warmup | 0.001 / 25,000 steps |
| Precision | bf16 |
| Parameters | 14.1M |

Note: conmamba_small uses grad_accumulation=1 (effective batch=32), much smaller than
conformer_small's effective batch of 256. This follows the original Jiang (2024) recipe.

### conmambamamba_small_S2S

**Config**: `hparams/S2S/conmambamamba_small.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | ConMamba, 12 layers, d_model=144, d_ffn=1024, BiMamba(d_state=16, expand=2, d_conv=4) |
| Decoder | Mamba, 4 layers (self-Mamba + cross-Mamba + FFN) |
| Activation | GELU |
| Dropout | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 1 |
| Effective batch | 32 |
| Dynamic batching | max_batch_length_train=1050 |
| Epochs | 150 |
| LR / Warmup | 0.001 / 25,000 steps |
| Precision | bf16 |
| Parameters | 14.6M |

---

## 14. Large Model Configurations

### conformer_large_S2S

**Config**: `hparams/S2S/conformer_large.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | Conformer, 12 layers, d_model=512, nhead=8, d_ffn=2048 |
| Decoder | Transformer, 6 layers |
| Attention | RelPosMHAXL |
| Activation | GELU |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 8 |
| Effective batch | 16 * 2 GPUs * 8 = 256 |
| Dynamic batching | max_batch_length_train=600 |
| Epochs | 150 |
| LR / Warmup | 0.0008 / 30,000 steps |
| Optimizer | AdamW |
| Precision | bf16 |
| Parameters | 109.1M |

### conmamba_large_S2S

**Config**: `hparams/S2S/conmamba_large.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | ConMamba, 12 layers, d_model=512, d_ffn=2048, BiMamba(d_state=16, expand=2, d_conv=4) |
| Decoder | Transformer, 6 layers |
| Activation | GELU |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 8 |
| Effective batch | 256 |
| Dynamic batching | max_batch_length_train=600 |
| Epochs | 150 |
| LR / Warmup | 0.0008 / 3,750 steps |
| Optimizer | AdamW |
| Precision | bf16 |
| Parameters | 115.2M |

### conmambamamba_large_S2S

**Config**: `hparams/S2S/conmambamamba_large.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | ConMamba, 12 layers, d_model=512, d_ffn=2048, BiMamba(d_state=16, expand=2, d_conv=4) |
| Decoder | Mamba, 6 layers |
| Activation | GELU |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| Batch size (per GPU) | 16 |
| Grad accumulation | 8 |
| Effective batch | 256 |
| Dynamic batching | max_batch_length_train=600 |
| Epochs | 150 |
| LR / Warmup | 0.0008 / 3,750 steps |
| Optimizer | AdamW |
| Precision | bf16 |
| Parameters | 122.9M |

---

## 15. CTC Model Configurations

### conmamba_large_CTC

**Config**: `hparams/CTC/conmamba_large.yaml`

| Parameter | Value |
|-----------|-------|
| Encoder | ConMamba, 18 layers, d_model=256, d_ffn=1024, BiMamba(d_state=16, expand=2, d_conv=4) |
| Decoder | CTC (linear projection only) |
| Activation | GELU |
| Dropout | 0.1 |
| Batch size (per GPU) | 32 |
| Grad accumulation | 4 |
| Effective batch | 32 * 4 GPUs * 4 = 512 |
| Dynamic batching | max_batch_length_train=1100 |
| Epochs | 500 |
| LR / Warmup | 0.001 / 7,500 steps |
| Optimizer | AdamW (weight_decay=0.0005) |
| Precision | bf16 |
| Parameters | 31.6M |
| Vocab | 31 characters |

### Why CTC models differ from S2S

- **18 layers** (vs 12): CTC models lack a decoder, so the encoder needs more depth.
- **d_model=256** (vs 512 for large S2S): smaller per-layer width compensated by more
  layers. This keeps the total parameter count manageable (~30M vs ~110M).
- **4 GPUs** (vs 2): CTC models use larger batch sizes to stabilize CTC training.
- **500 epochs** (vs 150): CTC convergence is slower because the loss landscape is
  harder (many valid CTC alignments for each target sequence).
- **Character tokenization**: CTC models use 31-character vocab instead of 5000 BPE
  tokens. Character-level CTC is simpler and avoids BPE segmentation artifacts.

---

## 16. SLURM Job Configuration

All jobs ran on the **Babel HPC cluster** at CMU.

### SLURM parameters

| Config | GPUs | CPUs | Memory | Partition | Precision | SLURM Script |
|--------|------|------|--------|-----------|-----------|-------------|
| conformer_small_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conformer_small_S2S.sh` |
| conmamba_small_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conmamba_small_S2S.sh` |
| conmambamamba_small_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conmambamamba_small_S2S.sh` |
| conformer_large_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conformer_large_S2S.sh` |
| conmamba_large_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conmamba_large_S2S.sh` |
| conmambamamba_large_S2S | 2x A6000 | 16 | 128 GB | preempt | bf16 | `slurm/conmambamamba_large_S2S.sh` |
| conmamba_large_CTC | 4x A6000 | 32 | 256 GB | preempt | bf16 | `slurm/conmamba_large_CTC.sh` |

### SLURM overrides

The SLURM scripts override certain YAML parameters at the command line to adapt to the
cluster environment. Common overrides include:

- `--data_folder`: points to `/data/user_data/anshulk/hnet_asr/LibriSpeech`
- `--output_folder`: points to `/data/user_data/anshulk/hnet_asr/results/<config_name>`
- `--batch_size`: adjusted per job (24 for small, 20 for large S2S, 32 for CTC)
- `--max_batch_length_train` / `--max_batch_length_val`: tuned per GPU memory
- `--precision`: bf16 for all models
- `--grad_accumulation_factor`: adjusted per effective batch target

### Preempt partition

The `preempt` partition on Babel allows up to 14 days of walltime but jobs can be
preempted by higher-priority jobs. All our models checkpoint after every epoch, so
preemption only loses at most one epoch of work. SpeechBrain's checkpoint system
automatically resumes from the latest checkpoint on restart.

### GPU: NVIDIA RTX A6000

- VRAM: 49,140 MiB (48 GB)
- Architecture: Ampere (GA102)
- bf16 support: native
- CUDA version: 11.8 (loaded via `module load cuda/11.8`)

---

## 17. Training Runs and Timeline

### Run timeline

All baseline training occurred in March 2026. Multiple runs were needed for some
experiments due to preemptions or errors.

| Experiment | Job ID | Start | End | Duration | Exit | Notes |
|-----------|--------|-------|-----|----------|------|-------|
| conformer_small_S2S | 6484205 | Mar 3, 17:41 | Mar 4, 07:39 | 13h 58m | 0 | Clean run, 150 epochs |
| conmamba_small_S2S | 6484208 | Mar 5, 05:20 | Mar 5, 08:40 | 3h 19m | 0 | Resumed from prior checkpoint |
| conmambamamba_small_S2S | 6484210 | Mar 5, 03:54 | Mar 5, 06:43 | 2h 50m | 0 | Resumed from prior checkpoint |
| conformer_large_S2S | 6484204 | Mar 6, 20:28 | Mar 7, 00:05 | 3h 37m | 0 | Resumed, completed remaining epochs |
| conmamba_large_S2S | 6473000 | Mar 2, 02:12 | Mar 3, 11:26 | 33h 14m | 1 | Training + eval (exit 1: eval error) |
| conmamba_large_S2S | 6484207 | Mar 5, 19:04 | Mar 5, 22:41 | 3h 37m | 0 | Re-eval run |
| conmambamamba_large_S2S | 6484209 | Mar 6, 07:42 | Mar 6, 20:54 | 13h 12m | 0 | Clean or resumed run |
| conmamba_large_CTC | 6473003 | Mar 3, 05:26 | — | — | — | Training run (~200+ epochs) |
| conmamba_large_CTC | 6484206 | Mar 14, 17:46 | Mar 14, 17:52 | 6m | 0 | Test-only eval run |

### Key observations

1. **Preemption and resumption**: Several jobs show short durations (3–6 hours) because
   they resumed from checkpoints saved by prior preempted runs. The total training time
   for each model (all epochs) is significantly longer than any single job's walltime.

2. **conmamba_large_S2S**: Required two jobs. The first (6473000, 33h) completed training
   but failed during evaluation with exit code 1. The second (6484207, 3.6h) successfully
   completed evaluation.

3. **conmamba_large_CTC**: Training completed in a multi-day run (6473003). The short
   6-minute run (6484206) was a test-only evaluation using the trained checkpoints.

---

## 18. Results: Word Error Rate

### Primary results: with language model (S2S) / standard decoding (CTC)

| Model | test-clean WER | test-other WER | Ins | Del | Sub (clean) | Total Words |
|-------|---------------|---------------|-----|-----|-------------|-------------|
| conformer_small_S2S | **2.52** | **5.97** | 163 | 96 | 1,065 | 52,576 |
| conmamba_small_S2S | **2.22** | **5.56** | 143 | 79 | 944 | 52,576 |
| conmambamamba_small_S2S | **2.52** | **5.98** | 149 | 115 | 1,059 | 52,576 |
| conformer_large_S2S | **2.03** | **4.70** | 138 | 72 | 857 | 52,576 |
| conmamba_large_S2S | **2.27** | **5.12** | 152 | 78 | 961 | 52,576 |
| conmambamamba_large_S2S | **2.41** | **5.72** | 147 | 92 | 1,030 | 52,576 |
| conmamba_large_CTC | **3.93** | **10.40** | 192 | 156 | 1,716 | 52,576 |

### No-LM results (S2S models only)

| Model | test-clean WER | test-other WER |
|-------|---------------|---------------|
| conformer_small_S2S | 4.13 | 10.13 |
| conmamba_small_S2S | 3.34 | 8.47 |
| conmambamamba_small_S2S | 3.64 | 8.70 |
| conformer_large_S2S | 2.57 | 5.94 |
| conmamba_large_S2S | 2.82 | 6.60 |
| conmambamamba_large_S2S | 2.93 | 6.99 |

### LM contribution (WER reduction from LM)

| Model | test-clean delta | test-other delta |
|-------|-----------------|-----------------|
| conformer_small_S2S | -1.61 (39%) | -4.16 (41%) |
| conmamba_small_S2S | -1.12 (34%) | -2.91 (34%) |
| conmambamamba_small_S2S | -1.12 (31%) | -2.72 (31%) |
| conformer_large_S2S | -0.54 (21%) | -1.24 (21%) |
| conmamba_large_S2S | -0.55 (20%) | -1.48 (22%) |
| conmambamamba_large_S2S | -0.52 (18%) | -1.27 (18%) |

The LM contributes more to small models (31–41% relative improvement) than large models
(18–22%), as expected — larger acoustic models capture more linguistic context internally.

---

## 19. Results Analysis

### Finding 1: ConMamba matches or beats Conformer at small scale

At small scale, conmamba_small achieves the best WER: 2.22/5.56 with LM, vs.
conformer_small's 2.52/5.97. This is a meaningful 12%/7% relative improvement.

The ConMamba small model also has the best no-LM performance (3.34/8.47 vs. 4.13/10.13),
suggesting the ConMamba encoder itself is producing better acoustic representations,
not just benefiting more from the LM.

### Finding 2: Conformer leads at large scale

At large scale, the ranking reverses: conformer_large_S2S achieves 2.03/4.70, the best
result overall. ConMamba large trails at 2.27/5.12 (12%/9% relative worse).

This suggests that self-attention's quadratic capacity is more valuable at larger model
dimensions, where it can leverage more attention heads (8) with larger head dimensions
(64). BiMamba's linear-time processing may sacrifice some representational power at
this scale.

### Finding 3: Mamba decoder slightly hurts WER

ConMambaMamba consistently performs slightly worse than ConMamba (same encoder, Transformer
decoder):

| Scale | ConMamba WER | ConMambaMamba WER | Delta |
|-------|-------------|-------------------|-------|
| Small (clean) | 2.22 | 2.52 | +0.30 |
| Small (other) | 5.56 | 5.98 | +0.42 |
| Large (clean) | 2.27 | 2.41 | +0.14 |
| Large (other) | 5.12 | 5.72 | +0.60 |

The Mamba decoder's cross-attention substitute (concatenate + Mamba + slice) appears
less effective than true cross-attention for sequence-to-sequence ASR. The gap is
consistent but small (0.1–0.6% absolute).

### Finding 4: CTC significantly trails S2S

conmamba_large_CTC (3.93/10.40) significantly trails conmamba_large_S2S (2.27/5.12).
The autoregressive decoder + LM provides a massive WER improvement, especially on
test-other where the gap is 5.28% absolute.

This is expected — the S2S decoder + LM provides powerful sequence-level modeling that
CTC cannot match. The CTC results primarily serve as an encoder quality diagnostic.

### Finding 5: Error type distribution

Across all models, substitutions dominate (70–80% of errors), followed by insertions
(10–15%) and deletions (5–10%). ConMamba models tend to have slightly fewer deletions
than Conformer, possibly because BiMamba's bidirectional processing helps preserve
frame-level information.

---

## 20. Reproducibility

### Software environment

| Package | Version |
|---------|---------|
| Python | 3.9 (miniconda3, `hnetasr` conda env) |
| PyTorch | 2.x (with CUDA 11.8) |
| SpeechBrain | 1.0+ |
| mamba-ssm | Latest (with CUDA kernels) |
| causal-conv1d | Required by mamba-ssm |
| sentencepiece | For BPE tokenization |
| tensorboard | >= 2.14.0 |
| wandb | 0.25.0 (optional, for experiment tracking) |

### To reproduce

1. Set up the `hnetasr` conda environment (see `requirements.txt`)
2. Download LibriSpeech 960h to `/data/user_data/anshulk/hnet_asr/LibriSpeech`
3. Download the pre-trained SentencePiece tokenizer and Transformer LM (SpeechBrain
   provides these via HuggingFace)
4. Submit SLURM scripts from `slurm/` directory

Each SLURM script is self-contained and includes all command-line overrides.

### Result locations

All results are stored under `/data/user_data/anshulk/hnet_asr/results/`:

```
results/
├── conformer_small_S2S/
│   ├── wer_test-clean.txt
│   ├── wer_test-other.txt
│   ├── no_lm/
│   │   ├── wer_test-clean.txt
│   │   └── wer_test-other.txt
│   ├── log.txt
│   ├── hyperparams.yaml
│   └── save/          # checkpoints
├── conmamba_small_S2S/
│   └── ...
├── conmambamamba_small_S2S/
│   └── ...
├── conformer_large_S2S/
│   └── ...
├── conmamba_large_S2S/
│   └── ...
├── conmambamamba_large_S2S/
│   └── ...
└── conmamba_large_CTC/
    ├── wer_test-clean.txt
    ├── wer_test-other.txt
    └── ...
```

---

## 21. What This Stage Does NOT Do

- **No Dynamic Chunking**: This stage trains vanilla encoders without any compression
  mechanism. The H-Mamba experiments are documented separately.
- **No hyperparameter search**: All hyperparameters follow the original Jiang (2024)
  recipes. We did not tune learning rates, batch sizes, or architecture parameters.
- **No data augmentation experiments**: We use the standard SpecAugment + speed
  perturbation pipeline without modification.
- **No streaming/chunked inference**: All models process the full utterance at once.
  Streaming evaluation is out of scope.
- **No cross-dataset evaluation**: We evaluate only on LibriSpeech test-clean and
  test-other. No evaluation on other datasets (CommonVoice, GigaSpeech, etc.).

---

## 22. What This Stage Establishes

1. **ConMamba is a viable encoder for ASR**: ConMamba achieves competitive WER at both
   scales, matching or beating Conformer at small scale and trailing slightly at large
   scale. This validates it as the encoder for H-Mamba experiments.

2. **The Transformer decoder matters**: S2S models significantly outperform CTC models,
   and the Transformer decoder outperforms the Mamba decoder. For the H-Mamba experiments,
   we will use the Transformer decoder.

3. **The LM contribution is quantified**: We know exactly how much the external LM
   contributes at each scale, which helps interpret H-Mamba results. If DC degrades
   no-LM WER but the LM compensates, we need to know the LM's contribution to
   understand whether the degradation is real.

4. **Reference numbers for the paper**: Every number in this document can be cited
   directly. The WER results, parameter counts, and training details are the ground
   truth for the baselines table in the paper.

### Summary results table (for paper)

| Model | Encoder | Decoder | Params | test-clean | test-other |
|-------|---------|---------|--------|------------|------------|
| Conformer-S | Conformer | Transformer+LM | 13.3M | 2.52 | 5.97 |
| ConMamba-S | ConMamba | Transformer+LM | 14.1M | 2.22 | 5.56 |
| ConMambaMamba-S | ConMamba | Mamba | 14.6M | 2.52 | 5.98 |
| Conformer-L | Conformer | Transformer+LM | 109.1M | 2.03 | 4.70 |
| ConMamba-L | ConMamba | Transformer+LM | 115.2M | 2.27 | 5.12 |
| ConMambaMamba-L | ConMamba | Mamba | 122.9M | 2.41 | 5.72 |
| ConMamba-L-CTC | ConMamba | CTC | 31.6M | 3.93 | 10.40 |

---

*Document last updated: April 1, 2026*
*All WER numbers are from checkpoint-averaged evaluation on LibriSpeech test sets.*
