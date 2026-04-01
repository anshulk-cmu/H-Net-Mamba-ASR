# H-Mamba Dynamic Chunking: Complete Analysis

**H-Mamba ASR Project — Anshul Kumar**
**Carnegie Mellon University, April 2026**
**Advisor: Prof. Shinji Watanabe**

This document records every decision, every piece of math, every architectural choice,
every hyperparameter, and every line of logic behind the H-Mamba Dynamic Chunking
system. It is the truth document for the H-Mamba stage of this project. The baseline
experiments are documented separately in `baseline_reproduction.md`.

---

## Table of Contents

1. [Purpose of This Stage](#1-purpose-of-this-stage)
2. [The Core Idea](#2-the-core-idea)
3. [Why Dynamic Chunking](#3-why-dynamic-chunking)
4. [Why Mamba Specifically](#4-why-mamba-specifically)
5. [The H-Net Paper](#5-the-h-net-paper)
6. [Architecture Overview](#6-architecture-overview)
7. [Stage 0: Frame-Level Processing](#7-stage-0-frame-level-processing)
   - [BiMamba v2: The Bidirectional SSM](#bimamba-v2-the-bidirectional-ssm)
8. [The Routing Module](#8-the-routing-module)
9. [The Boundary Decision Math](#9-the-boundary-decision-math)
10. [Gumbel-Softmax and Differentiability](#10-gumbel-softmax-and-differentiability)
11. [The Straight-Through Estimator](#11-the-straight-through-estimator)
12. [The Chunk Layer](#12-the-chunk-layer)
13. [Stage 1: Chunk-Level Processing](#13-stage-1-chunk-level-processing)
   - [How the decoder receives the encoder output](#how-the-decoder-receives-the-encoder-output)
14. [The DeChunk Layer](#14-the-dechunk-layer)
15. [The EMA Expansion Math](#15-the-ema-expansion-math)
16. [The Residual Connection](#16-the-residual-connection)
17. [The Load Balancing Loss](#17-the-load-balancing-loss)
18. [The Five Loss Terms Explained](#18-the-five-loss-terms-explained)
19. [The HMambaEncoder Class](#19-the-hmambaencoder-class)
20. [The HMambaEncoderWrapper](#20-the-hmambaencoderwrapper)
21. [The Training Script](#21-the-training-script)
22. [Warm-Up Schedules](#22-warm-up-schedules)
23. [Gumbel Temperature Annealing](#23-gumbel-temperature-annealing)
24. [The OOM Handler](#24-the-oom-handler)
25. [The HMamba Logger](#25-the-hmamba-logger)
26. [The Eight Experiments](#26-the-eight-experiments)
27. [Small Model Configurations](#27-small-model-configurations)
28. [Large Model Configurations](#28-large-model-configurations)
29. [SLURM Job Configuration](#29-slurm-job-configuration)
30. [The 100-Hour Pilot Study](#30-the-100-hour-pilot-study)
31. [Bug Fixes and Code Audit](#31-bug-fixes-and-code-audit)
32. [Gradient Flow Verification](#32-gradient-flow-verification)
33. [Known Deferred Issues](#33-known-deferred-issues)
   - [Positional encoding and compression interaction](#positional-encoding-and-compression-interaction)
   - [Streaming / online ASR incompatibility](#streaming--online-asr-incompatibility)
34. [The Alternative Loss Formula](#34-the-alternative-loss-formula)
35. [Package Upgrade and API Patches](#35-package-upgrade-and-api-patches)
36. [Tensor Shape Reference](#36-tensor-shape-reference)
37. [File Reference](#37-file-reference)
38. [What This Stage Will Establish](#38-what-this-stage-will-establish)

---

## 1. Purpose of This Stage

The baseline stage (documented in `baseline_reproduction.md`) established that ConMamba
is a viable ASR encoder. ConMamba Small achieves 2.22/5.56 WER on LibriSpeech test-clean/
test-other with LM. ConMamba Large achieves 2.27/5.12. These are strong results.

But ConMamba processes every frame equally. A 30-second utterance produces ~750 frames
after the CNN frontend (4x downsampling from 16kHz), and all 750 frames pass through
all 12 encoder layers. This is wasteful. Not all frames carry equal information:

- **Silence frames** carry no phonemic content
- **Steady-state vowels** repeat the same spectral pattern for many frames
- **Transitions** (consonant onsets, formant changes) carry critical boundary information

The H-Mamba stage tests whether we can teach the encoder to identify which frames are
important and compress the sequence mid-network, processing only the important frames
through the second half of the encoder. The key constraint: WER must not degrade
significantly. If we can compress 50% of frames (N=2) and maintain WER within 0.3%
absolute of the uncompressed baseline, we have a meaningful result.

This stage produces:

1. **8 trained H-Mamba models** (4 small, 4 large) at compression ratios from 0% to 75%
2. **WER numbers** with and without language model for each model
3. **Compression statistics** (actual ratio achieved, boundary distribution, chunk sizes)
4. **The WER-compression Pareto frontier** for both model scales

---

## 2. The Core Idea

The core idea comes from a simple observation about speech signals. Consider the
spectrogram of a typical English sentence:

- When a speaker says "the", the /dh/ sound occupies 2-3 frames and the /ax/ vowel
  occupies 5-8 frames. The vowel frames are highly redundant — each one looks almost
  identical to the previous one.
- When a speaker says "cat", the /k/ onset is a single sharp frame, the /ae/ vowel
  spans 6-10 frames, and the /t/ release is another sharp frame. The vowel can be
  compressed without losing much information. The onsets cannot.
- Silence between words can span 20-50 frames. All carry zero linguistic content.

If we could detect these patterns mid-network and keep only the "interesting" frames,
the second half of the encoder would process a much shorter sequence. For Mamba (which
runs in O(N) time), this directly reduces computation. For attention-based encoders
(O(N^2)), the savings would be even larger — but we use Mamba, which already has
linear complexity, so the savings are proportional to the compression ratio.

The Dynamic Chunking mechanism works as follows:

```
Full sequence: [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
                 ^               ^           ^         ^
              boundary        boundary     boundary  boundary

Compressed:   [f1, f4, f7, f10]  (4 frames instead of 10 — 2.5x compression)
```

The model learns where to place the boundaries. After processing the compressed sequence,
we expand it back to the original length using EMA interpolation, so the decoder sees
the full-length sequence and does not need any modification.

---

## 3. Why Dynamic Chunking

Several approaches exist for reducing sequence length in ASR:

### Fixed downsampling (Squeezeformer, Fast Conformer)

Apply a fixed stride (e.g., 2x or 8x) at a specific layer. Every Nth frame is kept
regardless of content. This is simple but suboptimal: it compresses a voiced consonant
onset exactly as much as it compresses silence. The compression ratio is fixed at
inference time.

### Token merging (A-ToMe)

Compute cosine similarity between all pairs of tokens and merge the most similar ones.
This is O(N^2) in the similarity computation and does not learn which frames to merge —
it uses a fixed similarity threshold. It also requires no training; it is applied
post-hoc to a trained model.

### Continuous Integrate-and-Fire (CIF)

Accumulate a learned "weight" at each frame. When the cumulative weight exceeds a
threshold, fire a boundary and emit a chunk. This produces variable-length output but
the weight accumulation is strictly left-to-right, which prevents the boundary decision
from using future context. CIF was designed for streaming ASR where this is a feature,
not a bug. For offline ASR (our setting), we want the boundary decision to use both
past and future context.

### Dynamic Chunking (H-Net / our approach)

Learn boundary positions using cosine similarity between adjacent frames in a projected
space. This is:

- **Content-aware**: boundaries are placed where the signal actually changes
- **Bidirectional**: the routing module sees the full sequence (frame t and frame t+1)
  before deciding whether t+1 is a boundary
- **Differentiable**: Gumbel-Softmax + STE allow end-to-end gradient flow
- **Variable-rate**: different utterances get different compression ratios based on
  their content (more silence → more compression)

The key advantage over CIF: our boundary detection uses both the current frame and the
next frame (via cosine similarity), giving it look-ahead capability. The key advantage
over A-ToMe: our system trains jointly with the ASR objective, so the boundary positions
are optimized for recognition accuracy, not just similarity.

---

## 4. Why Mamba Specifically

### The architectural compatibility argument

Mamba (and SSMs in general) process sequences frame-by-frame through a recurrent state.
They do not have an attention matrix that assumes a fixed sequence length. This means:

1. **Stage 0** (frame-level BiMamba) processes L frames and produces L outputs
2. **Stage 1** (chunk-level BiMamba) processes M frames and produces M outputs
3. Neither stage cares about the other's sequence length

If we used attention-based layers in Stage 1, we would need to handle the variable
sequence length more carefully — attention masks, position encodings for the compressed
sequence, etc. Mamba just processes whatever sequence length it receives.

### The linear complexity argument

Mamba runs in O(N) time and O(N) memory. If we compress from L to M = L/N frames:

- Stage 1 takes O(L/N) time instead of O(L)
- Total encoder time is O(L) + O(L/N) instead of O(L) + O(L) = O(2L)
- For N=2: O(L) + O(L/2) = O(1.5L) — a 25% reduction in encoder time
- For N=4: O(L) + O(L/4) = O(1.25L) — a 37.5% reduction

For attention-based encoders with O(N^2) per layer:

- Stage 1 would take O((L/N)^2) instead of O(L^2)
- For N=2: O(L^2) + O((L/2)^2) = O(1.25L^2) — a 37.5% reduction
- For N=4: O(L^2) + O((L/4)^2) = O(1.0625L^2) — a 46.9% reduction

The savings are larger for attention, but Mamba's base cost is already lower, so the
absolute savings matter more than the relative savings.

### The practical argument

ConMamba is the encoder we want to study. It already uses BiMamba. Adding DC to it
is a natural extension — we split the existing ConMamba layers, insert the DC mechanism,
and the downstream layers continue to work unchanged.

---

## 5. The H-Net Paper

The Dynamic Chunking mechanism is adapted from H-Net (Hwang, Wang, Gu, arXiv 2025).
H-Net was originally designed for text, not speech. It compresses byte sequences in a
language model by learning which bytes are word boundaries. The H-Net paper demonstrates
DC on English text, Chinese text, code, and DNA sequences.

### What we take from H-Net

1. **The boundary detection idea**: cosine similarity between adjacent hidden states
2. **The Gumbel-Softmax training**: differentiable discrete boundary decisions
3. **The EMA expansion**: exponential moving average to restore non-boundary positions
4. **The concept of a load balancing loss**: a loss term that controls compression ratio

### What we change from H-Net

1. **Bidirectional**: H-Net is a causal language model. We use bidirectional Mamba
   because ASR is offline (we have the full utterance). This means our boundary
   decisions benefit from future context.

2. **The encoder architecture**: H-Net uses its own Transformer-based architecture.
   We use ConMamba, which has the Conformer-style macaron FFN + convolution structure.

3. **The loss function**: H-Net uses a single-term load balancing formula. We use a
   5-term loss (BCE + mean + variance + entropy + ratio) that provides more gradient
   signals. This was a design choice to ensure stable compression convergence. The
   official H-Net formula is documented as a fallback (see Section 34).

4. **The split point**: H-Net can apply DC at multiple points. We apply it at a single
   fixed point (layer 6 of 12), splitting the encoder into two equal halves.

5. **The residual connection**: We project the pre-DC hidden states through a linear
   layer and add them to the post-DC output via STE-weighted addition. H-Net uses a
   similar approach but with different weighting.

---

## 6. Architecture Overview

The H-Mamba encoder replaces the vanilla ConMamba encoder. Instead of 12 layers
processing the full sequence, the encoder is split into two stages with a Dynamic
Chunking mechanism between them:

```
Audio Input (16 kHz waveform)
    |
CNN Frontend (2x Conv2d, stride 2 each → 4x downsampling)
    |
    v
(B, L, 640) — L ≈ T/4 where T = number of raw samples / 16000 * 100
    |
Linear Projection (640 → d_model)
    |
    v
(B, L, D) — D is 144 (small) or 512 (large)
    |
+------ Stage 0: ConMamba Layers 0-5 (frame-level) ------+
|   Each layer: FFN → BiMamba → ConvModule → FFN → Norm   |
+----------------------------------------------------------+
    |
    v
(B, L, D) — same shape, but now with 6 layers of context
    |
    +--- Save residual: residual_proj(stage0_out) → (B, L, D) ---+
    |                                                              |
    v                                                              |
+------ RoutingModule ------+                                      |
|  cos_sim → logits → prob  |                                      |
|  Gumbel-Softmax → mask    |                                      |
+----------------------------+                                     |
    |                                                              |
    v                                                              |
boundary_mask: (B, L) — True at boundary positions                 |
    |                                                              |
    v                                                              |
+------ ChunkLayer ------+                                         |
|  Select boundary frames |                                        |
|  L → M frames           |                                        |
+--------------------------+                                       |
    |                                                              |
    v                                                              |
(B, M, D) — M = number of boundaries, M << L                      |
    |                                                              |
+------ Stage 1: ConMamba Layers 6-11 (chunk-level) ------+        |
|   Same layer structure, but processing M frames          |       |
+----------------------------------------------------------+       |
    |                                                              |
    v                                                              |
(B, M, D) — M frames with 12 total layers of processing           |
    |                                                              |
+------ DeChunkLayer (EMA expansion) ------+                       |
|  Expand M → L using boundary_prob         |                      |
|  EMA interpolation fills gaps             |                      |
+-------------------------------------------+                      |
    |                                                              |
    v                                                              |
(B, L, D) — expanded back to original length                       |
    |                                                              |
    +--- Residual: expanded * STE(selected_probs) + residual ------+
    |
    v
(B, L, D) — final encoder output
    |
LayerNorm
    |
    v
To Decoder (Transformer S2S with CTC auxiliary)
```

### Why split at layer 6

The split at layer 6 (out of 12) divides the encoder into two equal halves. This is
a design choice, not a theoretical requirement. The reasoning:

- **Too early** (layer 2-3): the hidden states have not developed enough context for
  the routing module to make good boundary decisions. Early layers capture local
  acoustic features; boundaries require higher-level patterns.
- **Too late** (layer 9-10): most of the computation has already happened, so
  compression saves very little.
- **Layer 6**: a balance. The first 6 layers build sufficient context for boundary
  detection (including the ConvModule's 31-frame receptive field accumulated over 6
  layers), and the last 6 layers are fully compressed.

The split index is configurable via `hmamba_split_idx` in the YAML config. All our
experiments use 6.

---

## 7. Stage 0: Frame-Level Processing

Stage 0 consists of ConMamba layers 0 through 5. These layers process the full sequence
at frame level — every frame passes through every layer. The output has the same shape
as the input: (B, L, D).

Each ConMamba layer applies the following operations in order:

```python
x = x + 0.5 * FFN_1(x)              # Half-step feed-forward
x = x + BiMamba(LayerNorm(x))        # Bidirectional Mamba SSM
x = x + ConvModule(x)                # Depthwise separable convolution
x = LayerNorm(x + 0.5 * FFN_2(x))   # Half-step feed-forward + final norm
```

**FFN** (PositionalwiseFeedForward): Linear(d_model → d_ffn) → activation → dropout →
Linear(d_ffn → d_model). With d_ffn=1024 (small) or 2048 (large).

**BiMamba**: Bidirectional Mamba v2 (see detailed subsection below).

**ConvModule**: LayerNorm → Pointwise Conv (expand 2x, GLU) → Depthwise Conv (kernel=31)
→ LayerNorm → Swish → Linear → Dropout. The kernel size of 31 means each frame sees
31 neighboring frames at each layer. Over 6 layers, the effective receptive field grows
substantially.

After Stage 0, each frame's hidden state encodes both local acoustic features (from the
convolution) and global sequence context (from BiMamba). This is the representation that
the routing module will analyze to decide where to place boundaries.

### BiMamba v2: The Bidirectional SSM

BiMamba v2 (`bimamba.py`) is the core sequence model in every ConMamba encoder layer and
in the Mamba decoder layers. It processes the sequence in both forward and backward
directions using **completely separate parameter sets** for each direction, then averages
the results.

#### Key dimensions

| Dimension | Formula | Small (d_model=144) | Large (d_model=512) |
|-----------|---------|---------------------|---------------------|
| d_inner | expand × d_model | 288 | 1024 |
| dt_rank | ceil(d_model / 16) | 9 | 32 |
| d_state | fixed | 16 | 16 |
| d_conv | fixed | 4 | 4 |
| expand | fixed | 2 | 2 |

`d_inner` is the internal working dimension — all SSM operations happen at this expanded
width. `dt_rank` controls the bottleneck for the discretization step (delta). The formula
`ceil(d_model / 16)` keeps dt_rank proportional to model size.

#### Forward and backward parameter sets

BiMamba v2 maintains **separate parameters for the backward direction**. This is not
a simple flip-and-reuse — the backward path has its own learned conv1d, projections,
and skip connection:

| Parameter | Shape | Forward | Backward |
|-----------|-------|---------|----------|
| in_proj | (d_model, d_inner×2) | Shared | Shared |
| conv1d | (d_inner, 1, d_conv) | `conv1d` | `conv1d_b` |
| x_proj | (d_inner, dt_rank + d_state×2) | `x_proj` | `x_proj_b` |
| dt_proj | (dt_rank, d_inner) | `dt_proj` | `dt_proj_b` |
| A_log | (d_inner, d_state) | `A_log` | `A_b_log` |
| D (skip) | (d_inner,) | `D` | `D_b` |
| out_proj | (d_inner, d_model) | Shared | Shared |

The `in_proj` and `out_proj` are shared between directions. Everything in the SSM core
(conv1d, projections, state transition matrix A, skip connection D) is separate. This
gives the backward direction freedom to learn different temporal patterns than the
forward direction.

#### Forward pass step by step

```python
# 1. Project input to 2× width (shared for both directions)
xz = in_proj(hidden_states)            # (B, L, d_inner*2) → split into x and z

# 2. Forward direction
A = -exp(A_log)                        # (d_inner, d_state) — negative for stability
out = mamba_inner_fn_no_out_proj(
    xz, conv1d.weight, conv1d.bias,
    x_proj.weight, dt_proj.weight,
    A, None, None, D,                  # B, C are input-dependent (computed inside)
    delta_bias=dt_proj.bias,
    delta_softplus=True
)                                       # (B, d_inner, L)

# 3. Backward direction — FLIP input, use separate parameters
A_b = -exp(A_b_log)
out_b = mamba_inner_fn_no_out_proj(
    xz.flip([-1]),                      # Reverse time dimension
    conv1d_b.weight, conv1d_b.bias,
    x_proj_b.weight, dt_proj_b.weight,
    A_b, None, None, D_b,
    delta_bias=dt_proj_b.bias,
    delta_softplus=True
)                                       # (B, d_inner, L) — in reversed order

# 4. Average and project back to d_model
out = 0.5 * out + 0.5 * out_b.flip([-1])   # Flip backward output back
output = out_proj(out)                       # (B, L, d_model)
```

The `0.5` averaging (enabled by `if_devide_out=True`, the default) ensures the output
magnitude stays consistent regardless of direction. Without it, the output would be
2× the expected magnitude.

#### The SSM recurrence (from selective_scan_ref)

Inside `mamba_inner_fn_no_out_proj`, the core computation is the Selective State Space
Model scan. The reference implementation in `selective_scan_interface.py` makes this
explicit:

```python
# Inputs (all at d_inner width):
#   u: (B, d_inner, L)  — the gated input
#   delta: (B, d_inner, L) — discretization step sizes (from dt_proj + softplus)
#   A: (d_inner, d_state) — state transition matrix (negative, from -exp(A_log))
#   B: (B, d_state, L) — input-dependent input gate (from x_proj)
#   C: (B, d_state, L) — input-dependent output gate (from x_proj)
#   D: (d_inner,) — skip connection

# Step 1: Discretize A and B
deltaA = exp(einsum('bdl,dn->bdln', delta, A))   # (B, d_inner, L, d_state)
deltaB_u = einsum('bdl,bnl,bdl->bdln', delta, B, u)  # (B, d_inner, L, d_state)

# Step 2: Sequential scan (the recurrence)
x = zeros(B, d_inner, d_state)              # Initial hidden state
for i in range(L):
    x = deltaA[:,:,i] * x + deltaB_u[:,:,i]  # State update: decay + input
    y[i] = einsum('bdn,bn->bd', x, C[:,:,i]) # Output: read from state

# Step 3: Skip connection and gating
out = y + u * D                              # Add skip (D acts like a residual)
out = out * silu(z)                          # Gate with the other half of in_proj
```

**The key insight**: `delta` (discretization step) is **input-dependent** — it comes
from projecting the input through `x_proj` → `dt_proj` → `softplus`. This makes the
state transition selective: the model can learn to update its state quickly (large delta)
at transitions and slowly (small delta) during steady states. This selectivity is what
distinguishes Mamba from classical linear SSMs like S4.

**B and C are also input-dependent**: they are computed from the input via `x_proj`,
which outputs `(dt_rank + d_state*2)` values per frame. The first `dt_rank` values
become delta (via dt_proj), and the remaining `2*d_state` values are split into B and C.

#### A_log initialization (S4D real)

The state transition matrix A is initialized using the S4D (Structured State Spaces for
Sequences, Diagonal) real initialization:

```python
A = repeat(
    arange(1, d_state + 1, dtype=float32),   # [1, 2, 3, ..., 16]
    "n -> d n",                               # Broadcast to (d_inner, d_state)
    d=d_inner
)
A_log = log(A)                                # Store as log for numerical stability
```

This creates `A_log[d, n] = log(n+1)` for `n = 0, ..., d_state-1`. When used:
```python
A = -exp(A_log)   # A[d, n] = -(n+1)
```

The negative values ensure the state decays over time (`exp(delta * A)` < 1 when A < 0
and delta > 0). Larger state indices decay faster, creating a multi-scale memory: state
dimension 0 (A=-1) has the longest memory, state dimension 15 (A=-16) has the shortest.

Both forward and backward directions initialize their A matrices identically, but they
diverge during training because they receive different gradients.

#### Dropout note

The ConmambaEncoder constructor prints `dropout=0.1 is not used in Mamba` — this is
accurate. BiMamba itself has no dropout. Dropout is only applied in the surrounding
ConMamba layer structure (FFN and ConvModule), not inside the SSM computation. The SSM
recurrence operates deterministically on the gated inputs.

---

## 8. The Routing Module

The RoutingModule (`HMambaEncoder.py`, lines 60-184) is the brain of Dynamic Chunking.
It takes the Stage 0 output and produces a binary boundary decision for each frame.

### Architecture

The routing module has exactly 5 learnable parameters:

| Parameter | Shape | Initial Value | Purpose |
|-----------|-------|---------------|---------|
| q_proj_layer.weight | (D, D) | N(0, 0.02) | Projects frame t for query |
| k_proj_layer.weight | (D, D) | N(0, 0.02) | Projects frame t+1 for key |
| temperature | scalar | 0.5 | Controls decision sharpness |
| boundary_bias | scalar | 1.0 | Shifts decision threshold |
| gumbel_tau | scalar (not nn.Parameter) | 1.0 | Gumbel-Softmax temperature |

The q_proj and k_proj are the largest parameters (D*D each — 20,736 for d_model=144,
262,144 for d_model=512). Temperature and boundary_bias are single scalars.

### Forward pass step by step

**Step 1: Project adjacent frames**

```python
q = F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1)  # (B, L-1, D)
k = F.normalize(self.k_proj_layer(hidden_states[:, 1:]),  dim=-1)  # (B, L-1, D)
```

Frame t is projected through q_proj to create a "query" vector. Frame t+1 is projected
through k_proj to create a "key" vector. Both are L2-normalized so that the dot product
gives cosine similarity in [-1, 1].

The reason for separate projections: if we used the same projection (or no projection),
all frames in a steady-state region would project to nearly identical vectors, giving
cos_sim ≈ 1.0 everywhere. With separate projections, the model can learn to project
into a space where specific types of changes (e.g., formant transitions, voicing changes)
produce low similarity, while other changes (e.g., amplitude fluctuations within a vowel)
produce high similarity.

**Step 2: Compute cosine similarity**

```python
cos_sim = torch.einsum("bld,bld->bl", q, k)  # (B, L-1)
```

This computes the dot product between each pair of adjacent normalized vectors. Since
both are L2-normalized, this is exactly cosine similarity. The result is one value per
adjacent pair, so L frames produce L-1 similarity values.

**Step 3: Compute boundary logits**

```python
temp = torch.clamp(self.temperature.abs(), min=0.1, max=2.0)
logits = (1 - cos_sim + self.boundary_bias) / temp    # (B, L-1)
```

**Step 4: Convert to probabilities**

```python
boundary_prob_single = torch.sigmoid(logits)  # (B, L-1)
```

**Step 5: Add first-frame boundary**

```python
boundary_prob_single = F.pad(boundary_prob_single, (1, 0), "constant", 1.0)  # (B, L)
```

The first frame is always a boundary (probability 1.0). This ensures at least one
chunk exists and the first frame is always selected.

**Step 6: Format as 2-class distribution**

```python
boundary_prob = torch.stack(
    (1 - boundary_prob_single, boundary_prob_single), dim=-1
)  # (B, L, 2)
```

Channel 0 is P(not boundary), channel 1 is P(boundary). This format is required for
Gumbel-Softmax, which expects a categorical distribution.

**Step 7: Make discrete decision**

During training: Gumbel-Softmax with straight-through estimator (see Section 10).
During inference: simple threshold at 0.5.

---

## 9. The Boundary Decision Math

The core equation for boundary probability at frame t+1 is:

```
P(boundary at t+1) = sigmoid( (1 - cos_sim(q_t, k_{t+1}) + bias) / temperature )
```

Let us trace through what this means for different scenarios.

### Scenario 1: Identical adjacent frames (steady-state vowel)

If frame t and frame t+1 are nearly identical (e.g., two frames in the middle of /aa/):

```
cos_sim ≈ 1.0
logit = (1 - 1.0 + 1.0) / 0.5 = 1.0 / 0.5 = 2.0
P(boundary) = sigmoid(2.0) = 0.88
```

Wait — this says there is an 88% chance of a boundary even for identical frames! That
is because boundary_bias is initialized at 1.0, which keeps most frames at the start
of training (the warm-up target is N≈1, meaning keep everything). As training
progresses and the target N increases, the model learns to decrease boundary_bias.

### Scenario 2: After training, bias has decreased

Suppose training has adjusted boundary_bias to -0.5:

```
cos_sim ≈ 1.0 (same adjacent frames)
logit = (1 - 1.0 + (-0.5)) / 0.5 = -0.5 / 0.5 = -1.0
P(boundary) = sigmoid(-1.0) = 0.27
```

Now the probability is 27% — most steady-state frames will not be selected.

### Scenario 3: Transition frame (consonant onset)

If frame t is the end of a vowel and frame t+1 is a consonant onset:

```
cos_sim ≈ 0.3 (very different frames)
logit = (1 - 0.3 + (-0.5)) / 0.5 = 0.2 / 0.5 = 0.4
P(boundary) = sigmoid(0.4) = 0.60
```

The transition frame has a 60% chance of being selected. For sharper transitions
(cos_sim ≈ 0.0):

```
logit = (1 - 0.0 + (-0.5)) / 0.5 = 0.5 / 0.5 = 1.0
P(boundary) = sigmoid(1.0) = 0.73
```

### The role of temperature

Temperature controls how sharp the boundary decisions are:

- **High temperature (2.0)**: all probabilities are pushed toward 0.5 (uncertain)
- **Low temperature (0.1)**: probabilities are pushed toward 0 or 1 (sharp decisions)

The temperature is learned. It starts at 0.5 and the model adjusts it based on the
loss gradients. It is clamped to [0.1, 2.0] to prevent numerical issues.

### The role of boundary_bias

Boundary_bias shifts the decision threshold. It is the most important parameter for
controlling the compression ratio:

- **bias = 1.0** (initial): almost all frames are boundaries (keep everything)
- **bias = 0.0**: frames with cos_sim > 0.5 are not boundaries
- **bias = -1.0**: only frames with cos_sim < 0 are boundaries (very aggressive compression)

The load balancing loss pushes boundary_bias toward a value that achieves the target
compression ratio. For N=2 (50% compression), the model typically learns
boundary_bias ≈ -0.3 to -0.5.

---

## 10. Gumbel-Softmax and Differentiability

### The problem

The boundary decision is discrete: each frame is either a boundary or not. We cannot
use standard backpropagation through a discrete decision because the gradient of a
step function is zero everywhere (except at the threshold, where it is undefined).

### The Gumbel-Softmax solution

Gumbel-Softmax (Jang et al., 2017; Maddison et al., 2017) provides a differentiable
approximation to discrete sampling. It works as follows:

1. Take the log-probabilities: `log_probs = log(boundary_prob + 1e-8)`
2. Add Gumbel noise: `log_probs + G` where G ~ Gumbel(0, 1)
3. Apply softmax with temperature tau: `softmax((log_probs + G) / tau)`

The result is a continuous vector that approximates a one-hot vector. As tau → 0,
the output approaches a true one-hot vector. As tau → infinity, the output approaches
a uniform distribution.

### Hard vs. soft Gumbel-Softmax

We use `hard=True` in the Gumbel-Softmax call:

```python
boundary_hard = F.gumbel_softmax(
    torch.log(boundary_prob + 1e-8),
    tau=self.gumbel_tau,
    hard=True
)
```

With `hard=True`, PyTorch does the following:

**Forward pass**: round the soft output to one-hot (argmax)
**Backward pass**: use the soft output's gradients (straight-through estimator)

This means the forward pass uses discrete decisions (a frame is either selected or not),
but the backward pass can still compute gradients.

### The boundary_hard field

The Gumbel-Softmax hard output is stored in `RoutingModuleOutput.boundary_hard`. This
is critical for the ratio_loss gradient (Bug 1 fix — see Section 31). The boundary_hard
tensor has shape (B, L, 2) and carries STE gradients. Channel 1 (`boundary_hard[..., 1]`)
gives the differentiable boundary indicator.

The boundary_mask (`boundary_hard[..., 1] > 0.5`) is a boolean tensor with no gradients.
It is used for the actual frame selection in ChunkLayer but cannot be used in any loss
computation that needs gradient flow.

### During inference

At inference time, we do not use Gumbel-Softmax. We simply threshold:

```python
boundary_mask = boundary_prob_single > 0.5
```

This is deterministic — no randomness. The same input always produces the same boundaries.

---

## 11. The Straight-Through Estimator

The STE is used in the residual connection (Section 16). It is a custom autograd function
defined in `HMambaEncoder.py`:

```python
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def ste_func(x):
    return STE.apply(x)
```

### What it does

**Forward pass**: `ste_func(x)` returns a tensor of ones. This means multiplying by
`ste_func(x)` has no effect on the forward computation.

**Backward pass**: gradients flow through unchanged. This means the gradient with
respect to x is the same as the gradient with respect to the output.

### Why it is needed

The residual connection is:

```python
output = expanded_states * ste_func(selected_probs) + residual
```

In the forward pass, this is effectively:

```python
output = expanded_states * 1.0 + residual = expanded_states + residual
```

But in the backward pass, gradients flow through `ste_func(selected_probs)` to
`selected_probs`, which are the probabilities of the selected boundary decisions.
This gives the routing module a gradient signal from the ASR loss:

- If the ASR loss is high for a particular utterance, the gradient flows back through
  the STE to the selected_probs, which flows back to the boundary_prob, which flows
  back to the cosine similarity computation, which flows back to the q_proj and k_proj
  weight matrices.

Without the STE, the routing module would only receive gradients from the DC loss.
With the STE, it receives gradients from both the DC loss (which controls compression
ratio) and the ASR loss (which controls recognition accuracy). This dual gradient
signal is what allows the model to learn boundaries that are good for both compression
and recognition.

---

## 12. The Chunk Layer

The ChunkLayer (`HMambaEncoder.py`, lines 191-243) compresses the sequence by selecting
only the boundary frames. It is a purely deterministic operation with no learnable
parameters.

### Step-by-step operation

**Input**: hidden_states (B, L, D), boundary_mask (B, L)

**Step 1: Count boundaries**

```python
num_boundaries = boundary_mask.sum(dim=-1)  # (B,)
max_chunks = int(num_boundaries.max().item())
```

Each sequence in the batch may have a different number of boundaries. We take the
maximum across the batch as the output sequence length. Shorter sequences will be
padded with a validity mask.

**Step 2: Handle edge case**

```python
if max_chunks == 0:
    boundary_mask[:, 0] = True
    num_boundaries = boundary_mask.sum(dim=-1)
    max_chunks = 1
```

If no boundaries were selected (can happen early in training when boundary_bias is
very negative), force the first frame to be a boundary. This prevents empty sequences.

**Step 3: Sort boundaries to front**

```python
token_idx = torch.arange(L, device=device)[None, :].expand(B, -1)  # (B, L)
token_idx = token_idx + (~boundary_mask).long() * L                  # (B, L)
sorted_indices = torch.argsort(token_idx, dim=1)                     # (B, L)
```

This is a clever sorting trick. Non-boundary frames get their index increased by L,
which pushes them to the end after sorting. Boundary frames keep their original index,
so they sort to the front in their original order.

Example with L=6 and boundaries at positions 0, 2, 5:

```
Original indices:  [0, 1, 2, 3, 4, 5]
Boundary mask:     [T, F, T, F, F, T]
After offset:      [0, 7, 2, 9, 10, 5]   (non-boundaries get +6)
After argsort:     [0, 2, 5, 1, 3, 4]    (boundaries first, in order)
```

**Step 4: Gather boundary frames**

```python
gather_indices = sorted_indices[:, :max_chunks, None].expand(-1, -1, D)
chunked_states = torch.gather(hidden_states, dim=1, index=gather_indices)  # (B, M, D)
```

Take only the first `max_chunks` indices (which correspond to the boundary frames)
and gather the corresponding hidden states.

**Step 5: Create validity mask**

```python
chunk_mask = torch.arange(max_chunks, device=device)[None, :] < num_boundaries[:, None]
```

For sequences with fewer than `max_chunks` boundaries, the extra positions are marked
as invalid.

**Output**: chunked_states (B, M, D), chunk_mask (B, M), max_chunks (int)

### Why this approach

We use sorting + gathering instead of a simple `masked_select` because `masked_select`
produces variable-length outputs that cannot be batched. The sort-and-gather approach
maintains the batch dimension and handles variable boundary counts via the validity mask.

---

## 13. Stage 1: Chunk-Level Processing

Stage 1 consists of ConMamba layers 6 through 11. These are the same ConMamba layers as
in the baseline model — they are not modified in any way. The only difference is that
they now process M frames instead of L frames, where M is the number of boundaries.

```python
for layer in self.stage1_layers:
    chunked_states = layer(chunked_states)
```

Each layer applies the same FFN → BiMamba → ConvModule → FFN → Norm structure. The
BiMamba SSM processes the compressed sequence, building context over the M boundary
frames. The ConvModule's kernel_size=31 still applies, but now 31 consecutive frames
in the compressed sequence correspond to a much larger span of the original audio.

### No mask passed to Stage 1

The chunk_mask (which marks valid positions in the compressed sequence) is not passed
to the Stage 1 layers. This is an intentional design choice documented in the known
issues (Section 33):

- ConmambaEncoderLayer ignores the mask anyway — it explicitly sets `conv_mask = None`
  at line 316 of Conmamba.py
- BiMamba does not use attention masks (it is a recurrent model, not attention)
- The padded positions in the compressed sequence contain real gathered features, not
  noise, because they are gathered from the hidden_states tensor
- Dynamic batching groups sequences of similar length, so padding is minimal

### How the decoder receives the encoder output

Both the Transformer decoder and Mamba decoder receive the **full-length** encoder
output (B, L, D) — not the compressed (B, M, D) output. This is because the DeChunk
expansion (Section 14) restores the sequence to its original length before the output
leaves the encoder. The decoder is completely unaware that compression happened.

**Transformer decoder** (used in `conmamba_small`, `conmamba_large`, all H-Mamba runs):
Standard cross-attention between decoder tokens and encoder output. The encoder output
serves as key and value in the attention computation. Since the output is length L, the
attention matrix is (T_dec × L) — identical to the baseline without DC.

**Mamba decoder** (used in `conmambamamba_small`, `conmambamamba_large`):
The Mamba decoder replaces cross-attention with a concatenate-and-truncate mechanism:

```python
# In MambaDecoderLayer.forward():
# 1. Self-Mamba over target sequence
tgt2 = self.self_mamba(tgt)              # (B, T_dec, D)
tgt = tgt + dropout(tgt2)

# 2. Cross-Mamba: concatenate encoder output with target, run Mamba, keep only target
tgt2 = self.cross_mamba(
    torch.cat([memory, tgt], dim=1)      # (B, L + T_dec, D)
)[:, -tgt.shape[1]:]                     # Take last T_dec tokens → (B, T_dec, D)

tgt = tgt + dropout(tgt2)

# 3. Feed-forward
tgt2 = self.pos_ffn(tgt)
tgt = tgt + dropout(tgt2)
```

The concatenate-and-truncate trick works because Mamba processes left-to-right: by
placing the encoder output before the target tokens, the Mamba SSM state accumulates
information from the full encoder output before processing the target. The truncation
`[:, -T_dec:]` then extracts only the target positions, which now carry encoder context
in their SSM hidden state.

This is an implicit cross-attention — the encoder information flows into the decoder
through the SSM recurrence rather than through explicit key-value attention. The
self_mamba and cross_mamba are separate Mamba instances (unidirectional, not BiMamba)
with independent parameters.

**Important for H-Mamba**: Since both decoder types receive length-L output from the
encoder, no decoder modifications are needed for Dynamic Chunking. The DC mechanism
is fully encapsulated within the encoder.

---

## 14. The DeChunk Layer

The DeChunkLayer (`HMambaEncoder.py`, lines 250-370) expands the compressed sequence
back to the original length. It has no learnable parameters — it uses the boundary
probabilities from the routing module to interpolate between boundary frames via
Exponential Moving Average (EMA).

### Step-by-step operation

**Input**: chunked_states (B, M, D), boundary_mask (B, L), boundary_prob (B, L, 2)

**Step 1: Extract boundary probabilities**

```python
p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1-1e-4)  # (B, L)
```

The boundary probability at each position determines how the EMA weights work.
Clamping prevents numerical issues (log(0) in the EMA kernel).

**Step 2: Reorder probabilities to match compressed sequence**

The probabilities are reordered using the same sort-and-gather trick as ChunkLayer:

```python
token_idx = torch.arange(L, device=device)[None, :].expand(B, -1)
token_idx = token_idx + (~boundary_mask).long() * L
sorted_indices = torch.argsort(token_idx, dim=1)
p_chunked = torch.gather(p, dim=1, index=sorted_indices[:, :M])  # (B, M)
```

Now p_chunked contains the boundary probabilities for the M boundary frames in the
same order as chunked_states.

**Step 3: Apply EMA**

The EMA computation produces expanded (smoothed) hidden states at the M boundary
positions. The formula is:

```
out[0] = x[0]
out[t] = p[t] * x[t] + (1 - p[t]) * out[t-1]    for t = 1, ..., M-1
```

This is implemented either via the optimized Mamba kernel or the PyTorch fallback
(see Section 15).

**Step 4: Map back to original positions**

```python
chunk_indices = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
chunk_indices = chunk_indices.clamp(min=0, max=M-1)
expanded_states = ema_output.gather(
    dim=1,
    index=chunk_indices.unsqueeze(-1).expand(-1, -1, D)
)  # (B, L, D)
```

The cumulative sum of boundary_mask maps each original position to its corresponding
chunk index. For example:

```
boundary_mask:   [T, F, F, T, F, T, F, F]
cumsum:          [1, 1, 1, 2, 2, 3, 3, 3]
chunk_indices:   [0, 0, 0, 1, 1, 2, 2, 2]   (subtract 1, clamp to [0, M-1])
```

Positions 0, 1, 2 all map to chunk 0 (the first boundary frame).
Positions 3, 4 map to chunk 1.
Positions 5, 6, 7 map to chunk 2.

The gather operation copies the EMA output at each chunk index to all positions that
belong to that chunk.

---

## 15. The EMA Expansion Math

The Exponential Moving Average is the core of the DeChunk expansion. It uses the
boundary probability as the mixing coefficient.

### The formula

For the M compressed frames:

```
out[0] = x[0]                                     (first frame passed through)
out[t] = p[t] * x[t] + (1 - p[t]) * out[t-1]     (for t = 1, ..., M-1)
```

Where:
- `x[t]` is the Stage 1 output at compressed position t (D-dimensional vector)
- `p[t]` is the boundary probability at compressed position t (scalar in (0, 1))
- `out[t]` is the EMA output at compressed position t (D-dimensional vector)

### Intuition

- When p[t] is high (near 1.0): `out[t] ≈ x[t]`. The current frame dominates.
  This happens at strong boundaries where the frame content changed significantly.
- When p[t] is low (near 0.0): `out[t] ≈ out[t-1]`. The previous output is carried
  forward. This happens at weak boundaries (similar frames) — the smoothing carries
  forward the previous context.

The effect is that the EMA output at each position is a weighted average of the current
frame and all previous frames, with exponentially decaying weights. Strong boundaries
reset the average; weak boundaries let it drift.

### PyTorch implementation

```python
def _ema_pytorch(self, x, p):
    outputs = [x[:, 0]]                                         # (B, D)
    for t in range(1, x.shape[1]):
        prev = outputs[-1]                                       # (B, D)
        curr = p[:, t:t+1] * x[:, t] + (1 - p[:, t:t+1]) * prev # (B, D)
        outputs.append(curr)
    return torch.stack(outputs, dim=1)                           # (B, M, D)
```

This is a simple sequential loop. It is O(M * D) and cannot be parallelized because
each step depends on the previous output. This is the fallback for CPU or non-CUDA
devices.

### Mamba kernel implementation

The optimized implementation uses the Mamba2 kernel (`mamba_chunk_scan_combined`), which
reformulates the EMA as a State Space Model (SSM) scan:

```python
dt = log(1 / (1 - p))        # Convert probability to "delta time"
x_scaled = x / dt             # Scale input
A = -ones(nheads)              # State transition: decay by exp(-1) per step
B = p                          # Input gate: boundary probability
C = ones_like(p)               # Output gate: pass through
```

The Mamba kernel computes the SSM scan in O(M) work using the parallel scan algorithm,
which is much faster than the sequential loop on GPU. The kernel processes chunks of
`block_size=256` frames in parallel.

### Head dimension

The EMA kernel requires reshaping the D-dimensional vectors into (nheads, headdim)
format:

- Small model: D=144, headdim=36, nheads=4
- Large model: D=512, headdim=128, nheads=4

The EMA is applied independently to each head dimension, then the heads are concatenated.

---

## 16. The Residual Connection

After DeChunk expansion, the expanded states are combined with the pre-DC residual:

```python
residual = self.residual_proj(stage0_out.float()).to(stage0_out.dtype)
# ... (Stage 1 and DeChunk happen here) ...
output = expanded_states.float() * ste_func(router_output.selected_probs) + residual
output = output.to(src.dtype)
```

### Why a residual connection

Without the residual, all information about non-boundary frames would be lost after
chunking. The DeChunk expansion fills in non-boundary positions by copying from the
nearest boundary frame, but this is an approximation. The residual connection preserves
the original frame-level information and adds it to the expanded output.

### The residual projection

`residual_proj` is a Linear(d_model, d_model, bias=False) initialized to zero weights.
This means at the start of training, the residual is zero, and the output equals the
expanded states. As training progresses, the projection learns to extract useful
information from the Stage 0 output.

Zero initialization is critical: if the projection started with random weights, the
residual would add noise to the expanded states at the beginning of training, before
the routing module has learned meaningful boundaries.

### The STE multiplication

The `ste_func(selected_probs)` multiplication serves a specific purpose:

**Forward pass**: returns 1.0, so `output = expanded_states + residual`
**Backward pass**: connects gradients from the ASR loss to the selected_probs

This is the mechanism by which the routing module receives task-specific gradient
signals (see Section 11 for full explanation).

### Bug 2 fix context

The original code had:

```python
output = stage0_out + residual + expanded_states  # WRONG
```

This was wrong for two reasons:
1. It added stage0_out twice (once directly, once through residual_proj)
2. It did not include the STE multiplication, so the routing module received no
   gradient from the ASR loss

The fixed version:

```python
output = expanded_states.float() * ste_func(router_output.selected_probs) + residual
```

This was verified to restore gradient flow to the routing module parameters.

---

## 17. The Load Balancing Loss

The load balancing loss (`load_balancing_loss` in `HMambaEncoder.py`, lines 377-473)
controls the compression ratio. Without this loss, the routing module would have no
incentive to compress — it could keep all frames (boundary_bias stays at 1.0) and get
zero compression.

### The core tension

There are two conflicting objectives:

1. **ASR loss**: wants to keep all frames (more information → better WER)
2. **DC loss**: wants to match a target compression ratio (e.g., 50% for N=2)

The total loss is:

```python
total_loss = asr_loss + dc_loss_weight * dc_loss
```

The dc_loss_weight controls the balance. Higher weight → stricter compression at the
cost of potential WER degradation. Lower weight → less compression, better WER.

### The target ratio

The target ratio `r = 1/N` specifies what fraction of frames should be boundaries:

| N | Target Ratio r | Meaning |
|---|----------------|---------|
| 1 | 1.0 | Keep all frames (control, no compression) |
| 2 | 0.5 | Keep 50% of frames |
| 3 | 0.333 | Keep 33% of frames |
| 4 | 0.25 | Keep 25% of frames |

---

## 18. The Five Loss Terms Explained

### Loss 1: BCE (Binary Cross-Entropy)

```python
bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, target_probs)
```

**Weight**: 1.0

**What it does**: treats each frame independently and pushes its boundary probability
toward the target ratio. If target_ratio=0.5, every frame is encouraged to have a 50%
boundary probability. This does not mean every frame should be ambiguous — the other
loss terms push toward decisive (0 or 1) probabilities. The BCE just ensures the
overall mean probability is correct.

**Why logits not probabilities**: `binary_cross_entropy_with_logits` is numerically
stable under bf16 mixed precision. Using `binary_cross_entropy` with raw probabilities
can produce NaN under autocast.

### Loss 2: Mean probability matching

```python
avg_boundary_prob = boundary_probs.mean()
mean_loss = (avg_boundary_prob - target_ratio) ** 2
```

**Weight**: 5.0

**What it does**: penalizes the squared difference between the average boundary
probability across all frames in the batch and the target ratio. This is a stronger
signal than BCE because it operates on the global mean, not per-frame.

### Loss 3: Variance regularization

```python
prob_variance = boundary_probs.var()
target_variance = target_ratio * (1 - target_ratio)  # Bernoulli variance
variance_loss = F.relu(target_variance * 0.5 - prob_variance)
```

**Weight**: 0.5

**What it does**: penalizes low variance in boundary probabilities. If all probabilities
are near 0.5 (the model is uncertain about everything), variance is low and this loss
fires. If some probabilities are near 0 and others near 1 (the model is making clear
decisions), variance is high and this loss is zero.

The target variance is `r * (1-r)`, which is the variance of a Bernoulli distribution
with probability r. The ReLU ensures this loss only fires when variance is below half
the target — we do not penalize high variance.

### Loss 4: Entropy regularization

```python
entropy = -(boundary_prob_float * (boundary_prob_float + 1e-8).log()).sum(dim=-1).mean()
max_entropy = -2 * (0.5 * log(0.5))  # = log(2) ≈ 0.693
entropy_loss = F.relu(0.5 * max_entropy - entropy)
```

**Weight**: 0.5

**What it does**: penalizes low entropy in the boundary probability distribution. Low
entropy means the model has collapsed to always predicting the same class (either
always boundary or always not-boundary). This loss keeps the model exploring.

The ReLU ensures this loss only fires when entropy drops below half of maximum entropy.
Above that threshold, the loss is zero.

### Loss 5: Ratio loss (the most important term)

```python
if boundary_hard is not None:
    actual_ratio = boundary_hard[..., 1].mean()   # Training: STE gradients
else:
    actual_ratio = boundary_mask.float().mean()    # Inference: no gradients needed

ratio_diff = actual_ratio - target_ratio
ratio_loss = ratio_diff.abs() + ratio_diff ** 2    # Huber-style
```

**Weight**: 10.0 (highest weight)

**What it does**: directly penalizes the difference between the actual compression
ratio (fraction of frames selected as boundaries) and the target ratio. This is the
most direct signal for compression control.

The Huber-style loss (`|x| + x^2`) provides:
- Strong constant gradient when far from target (from the |x| term)
- Smooth optimization near the target (from the x^2 term)

**Critical implementation detail**: This loss uses `boundary_hard[..., 1]` (the
Gumbel-Softmax hard output with STE gradients), NOT `boundary_mask` (which is boolean
and has no gradients). This was Bug 1 — the original code used `boundary_mask.float().mean()`,
which always gave zero gradient to boundary_bias, preventing the routing module from
learning to adjust the compression ratio.

### Combined loss

```python
loss = (1.0 * bce_loss + 5.0 * mean_loss + 0.5 * variance_loss +
        0.5 * entropy_loss + 10.0 * ratio_loss)
```

The ratio_loss dominates (weight 10.0), followed by mean_loss (5.0), then BCE (1.0),
then variance and entropy (0.5 each). This weighting was chosen empirically during
the 100h pilot study.

---

## 19. The HMambaEncoder Class

`HMambaEncoder` (`HMambaEncoder.py`, lines 499-628) is the standalone encoder class
that combines all components:

```python
class HMambaEncoder(nn.Module):
    def __init__(self, d_model=144, stage0_layers=None, stage1_layers=None,
                 headdim=36, target_ratio=2.0, device=None, dtype=None):
```

### Initialization

- Creates RoutingModule, ChunkLayer, DeChunkLayer
- Creates residual_proj (Linear, d_model → d_model, zero-initialized)
- Defines residual_func as a lambda:
  `lambda out, residual, p: out * ste_func(p) + residual`

### Forward pass

The forward method implements the full pipeline: Stage 0 → Route → Chunk → Stage 1 →
DeChunk → Residual → Output.

### compute_dc_loss

Calls `load_balancing_loss(router_output, self.target_ratio)` and returns the scalar loss.

---

## 20. The HMambaEncoderWrapper

`HMambaEncoderWrapper` (`HMambaEncoderWrapper.py`, lines 46-304) is the wrapper used
in actual training. It takes an existing ConmambaEncoder, splits its layers, and inserts
the DC mechanism. This is the class instantiated by `create_hmamba_from_conmamba()`.

### Key differences from HMambaEncoder

1. **Takes an existing encoder**: does not create new layers; splits the existing
   ConMamba layers into Stage 0 and Stage 1
2. **Copies the final LayerNorm**: from the original encoder
3. **Stores DC loss**: in `self.last_dc_loss` and `self.last_compression_ratio` after
   every forward pass, so the training script can access them without recomputation
4. **Has API compatibility**: accepts the same arguments as ConmambaEncoder's forward
   method (src_mask, src_key_padding_mask, pos_embs, dynchunktrain_config) even though
   most are unused

### The forward method

The wrapper's forward (lines 103-223) follows the same pipeline as HMambaEncoder
but also:

- Converts src_key_padding_mask to a validity mask (inverts the convention: SpeechBrain
  uses True=padded, we use True=valid)
- Stores `self.last_dc_loss` after every forward pass
- Stores `self.last_compression_ratio` for logging
- Returns (output, stats_dict_or_None) to match the ConmambaEncoder API

### The factory function

```python
def create_hmamba_from_conmamba(conmamba_encoder, d_model=144, split_idx=6,
                                target_compression_N=2.0, headdim=36):
    return HMambaEncoderWrapper(conmamba_encoder, d_model, split_idx,
                                 target_compression_N, headdim)
```

This is the entry point called by the training script.

---

## 21. The Training Script

`train_s2s_hmamba.py` (758 lines) defines the `HMambaASR` Brain class, which extends
SpeechBrain's Brain class with H-Mamba-specific functionality.

### Key methods

**on_fit_start()** (lines 42-100):
- Wraps the ConMamba encoder with HMambaEncoderWrapper BEFORE super().on_fit_start()
  so that DC parameters are included in the optimizer
- Verifies that routing_module parameters are in the optimizer's parameter groups
- If DC params are missing (happens with some SpeechBrain optimizer configurations),
  adds them manually to a new parameter group
- Initializes HMambaLogger with experiment configuration

**_wrap_encoder_with_hmamba()** (lines 102-144):
- Gets the ConMamba encoder from self.modules.Transformer.encoder
- Reads hyperparameters: d_model, hmamba_split_idx, hmamba_target_N, hmamba_headdim
- Creates HMambaEncoderWrapper via create_hmamba_from_conmamba()
- Moves wrapper to correct device
- Replaces the encoder in self.modules.Transformer
- Logs all routing module parameters with shapes and requires_grad status

**compute_forward()** (lines 186-255):
- Standard SpeechBrain compute_forward with two additions:
  1. Updates target_compression_N during training (warm-up schedule)
  2. Tracks audio duration for RTF calculation
- Computes features, applies augmentation, runs through CNN frontend and Transformer
- Returns (p_ctc, p_seq, wav_lens, hyps, src)

**compute_objectives()** (lines 257-344):
- Computes base ASR loss = ctc_weight * CTC_loss + (1 - ctc_weight) * seq2seq_loss
- During training only: gets DC loss from self.hmamba_encoder.last_dc_loss
- Total loss = asr_loss + dc_loss_weight * dc_loss
- Stores current losses and DC stats for batch logging

**on_stage_start()** (lines 346-378):
- For TRAIN: resets DC metrics, updates Gumbel temperature, starts HMambaLogger epoch
- Logs current DC parameters (target_N, gumbel_tau, temperature, boundary_bias)

**on_stage_end()** (lines 380-451):
- For TRAIN: computes average DC metrics
- For VALID: saves checkpoints, updates early stopping metric, logs to HMambaLogger
- For TEST: writes WER to file

**on_evaluate_start()** (lines 453-473):
- Ensures HMambaEncoderWrapper is initialized BEFORE loading averaged checkpoints
  (Bug 3 fix — the original code loaded checkpoints before wrapping, causing key
  mismatch errors)

**fit_batch()** (lines 475-489):
- OOM handler: wraps super().fit_batch() in try/except
- On CUDA OOM: clears gradients, empties cache, returns zero loss
- This is necessary because DC produces variable-length sequences, so some batches
  may use more memory than others

**on_fit_batch_end()** (lines 491-542):
- Computes gradient norm for logging
- Applies Noam learning rate schedule
- Logs batch metrics via HMambaLogger (losses, DC stats, timing, VRAM)

---

## 22. Warm-Up Schedules

### Compression warm-up

The target compression ratio starts at N=1 (no compression) and linearly increases to
the final target over `hmamba_warmup_epochs` epochs:

```python
def _get_current_target_N(self):
    warmup_epochs = getattr(self.hparams, 'hmamba_warmup_epochs', 20)
    target_N = getattr(self.hparams, 'hmamba_target_N', 2.0)
    current_epoch = self.hparams.epoch_counter.current

    if current_epoch < warmup_epochs:
        current_N = 1.0 + (target_N - 1.0) * (current_epoch / warmup_epochs)
    else:
        current_N = target_N
    return current_N
```

For N=2 with warmup_epochs=15:

| Epoch | current_N | Target ratio |
|-------|-----------|-------------|
| 0 | 1.0 | 100% kept |
| 3 | 1.2 | 83% kept |
| 7 | 1.47 | 68% kept |
| 10 | 1.67 | 60% kept |
| 15 | 2.0 | 50% kept |
| 16+ | 2.0 | 50% kept |

### Why warm-up

If we start with aggressive compression from epoch 0, the model has not learned
meaningful representations yet, and the routing module makes random boundary decisions.
This can cause the ASR loss to spike early and the model may never recover.

By starting with N=1 (keep everything), the model first learns to do ASR well, then
gradually learns to compress. The routing module has time to learn what cosine similarity
patterns correspond to meaningful boundaries.

### boundary_bias initialization

The boundary_bias is initialized at 1.0, which means nearly all frames are kept at the
start. This is consistent with the warm-up starting at N=1. As the warm-up increases
N, the DC loss pushes boundary_bias downward to achieve the target compression.

---

## 23. Gumbel Temperature Annealing

The Gumbel-Softmax temperature (tau) controls the sharpness of the discrete boundary
decisions during training:

```python
def _update_gumbel_temperature(self):
    gumbel_start = getattr(self.hparams, 'hmamba_gumbel_start', 1.0)
    gumbel_end = getattr(self.hparams, 'hmamba_gumbel_end', 0.5)
    gumbel_anneal_epochs = getattr(self.hparams, 'hmamba_gumbel_anneal_epochs', 30)
    current_epoch = self.hparams.epoch_counter.current

    if current_epoch < gumbel_anneal_epochs:
        gumbel_tau = gumbel_start - (gumbel_start - gumbel_end) * (
            current_epoch / gumbel_anneal_epochs)
    else:
        gumbel_tau = gumbel_end
    self.hmamba_encoder.routing_module.gumbel_tau = gumbel_tau
```

### Schedule per experiment

| Config | Start | End | Anneal Epochs |
|--------|-------|-----|---------------|
| N=1 (control) | 1.0 | 1.0 | 1 |
| N=2 | 1.0 | 0.3 | 30 |
| N=3 | 1.0 | 0.3 | 35 |
| N=4 | 1.0 | 0.3 | 35 |

### Why anneal

- **High tau (1.0)**: soft decisions, lots of exploration. The Gumbel noise dominates
  and boundaries are placed semi-randomly. This is good early in training when the
  routing module is still learning.
- **Low tau (0.3)**: sharp decisions, less exploration. The routing module's learned
  probabilities dominate and boundaries are placed at acoustically meaningful positions.
  This is good later in training when the routing module has learned what to look for.

The N=1 control does not anneal (tau stays at 1.0, anneal_epochs=1) because no
compression happens — there is no boundary to make sharp.

---

## 24. The OOM Handler

Dynamic Chunking creates variable memory usage because different utterances produce
different numbers of chunks. A batch with many short chunks might use significantly
more memory than expected (Stage 1 processes M frames, but M varies per utterance).

The OOM handler in `fit_batch()`:

```python
def fit_batch(self, batch):
    try:
        loss = super().fit_batch(batch)
        return loss
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("[H-Mamba] CUDA OOM — skipping batch, clearing cache.")
            for p in self.modules.parameters():
                if p.grad is not None:
                    del p.grad
            torch.cuda.empty_cache()
            import gc; gc.collect()
            return torch.tensor(0.0, device=self.device)
        raise
```

### What it does

1. Catches CUDA out-of-memory RuntimeErrors
2. Deletes all accumulated gradients (frees GPU memory)
3. Empties the CUDA cache (returns freed memory to the allocator)
4. Runs Python garbage collection (frees any remaining references)
5. Returns a zero loss (this batch is skipped)

### Why return zero loss

Returning zero means this batch contributes no gradient update. The optimizer step
still happens (with zero gradients), but the model parameters do not change. This is
equivalent to skipping the batch entirely. The alternative — crashing — would lose
all training progress since the last checkpoint.

### When it triggers

OOM typically happens with:
- Very long utterances (30+ seconds → 750+ frames → many chunks)
- High compression (N=4 produces many small chunks with high memory overhead per chunk)
- Large models (d_model=512 uses 4x more memory per frame than d_model=144)

---

## 25. The HMamba Logger

`hmamba_logger.py` (884 lines) provides comprehensive logging for H-Mamba training.
It tracks metrics that are specific to Dynamic Chunking and are not available in
standard SpeechBrain logging.

### Metrics tracked

**Per-batch** (logged every N batches, default N=10):
- Losses: total, CTC, seq2seq, DC
- DC stats: compression ratio, number of chunks, average chunk size, boundary
  probability mean, target N
- Timing: batch time, data load time, forward time, backward time (milliseconds)
- GPU: VRAM used, VRAM allocated
- Optimization: gradient norm

**Per-epoch**:
- Aggregated training losses (mean over batches)
- Validation: loss, accuracy, WER, CER
- DC stats: average compression ratio, std, target ratio, average chunks,
  average boundary probability
- Performance: Real-Time Factor (RTF), epoch time, total audio processed
- GPU: peak VRAM, average VRAM
- Optimization: learning rate, average/max gradient norm

**Per-inference-run**:
- Dataset name, number of samples
- WER, CER
- Total inference time, total audio duration
- RTF, average latency per sample
- Average compression ratio, peak VRAM

### Output files

The logger creates several files in the log directory:

| File | Format | Content |
|------|--------|---------|
| batch_metrics.csv | CSV | Per-batch metrics (one row per logged batch) |
| epoch_metrics.csv | CSV | Per-epoch metrics (one row per epoch) |
| training.log | Text | Human-readable training log |
| experiment_config.json | JSON | Full experiment configuration |
| experiment_summary.txt | Text | Summary report generated at end of training |
| events.out.tfevents.* | TensorBoard | TensorBoard event files |

### GPU monitoring

The `GPUMonitor` class attempts to use pynvml (NVIDIA Management Library) for detailed
GPU metrics (utilization, temperature, power draw). If pynvml is not available, it
falls back to PyTorch's CUDA memory tracking.

### Timer

The `Timer` class provides high-precision timing with optional CUDA synchronization:

```python
with Timer(cuda_sync=True) as t:
    # code to time
elapsed_ms = t.elapsed_ms
```

CUDA synchronization ensures that GPU operations complete before the timer stops,
giving accurate wall-clock times for GPU-intensive operations.

---

## 26. The Eight Experiments

### The experimental grid

| # | Name | d_model | target_N | Compression | DC Loss Weight |
|---|------|---------|----------|-------------|----------------|
| 1 | hmamba_small_N1 | 144 | 1.0 | 0% (control) | 0.0 |
| 2 | hmamba_small_N2 | 144 | 2.0 | 50% | 5.0 |
| 3 | hmamba_small_N3 | 144 | 3.0 | 67% | 6.5 |
| 4 | hmamba_small_N4 | 144 | 4.0 | 75% | 7.5 |
| 5 | hmamba_large_N1 | 512 | 1.0 | 0% (control) | 0.0 |
| 6 | hmamba_large_N2 | 512 | 2.0 | 50% | 5.0 |
| 7 | hmamba_large_N3 | 512 | 3.0 | 67% | 6.5 |
| 8 | hmamba_large_N4 | 512 | 4.0 | 75% | 7.5 |

### Why N=1 as control

N=1 means target_ratio=1.0 (keep all frames). This is equivalent to running the full
encoder with the DC mechanism present but inactive. It tests whether the DC architecture
itself (the split, the routing module, the residual connection) introduces any WER
degradation, independent of compression.

If N=1 matches the ConMamba baseline WER, then any WER change at N=2/3/4 is due to
the compression, not the architectural modification.

### Why these DC loss weights

Higher compression targets need stronger DC loss to enforce the ratio:

- N=1: weight=0.0 (no DC loss — no compression target)
- N=2: weight=5.0 (moderate enforcement)
- N=3: weight=6.5 (stronger enforcement)
- N=4: weight=7.5 (strongest enforcement)

These values were chosen based on the 100h pilot study (Section 30). The general
principle: more aggressive compression requires stronger loss to overcome the ASR
loss's natural preference for keeping all frames.

---

## 27. Small Model Configurations

All small models share these base parameters (matching conmamba_small.yaml):

| Parameter | Value |
|-----------|-------|
| d_model | 144 |
| nhead | 4 (unused by BiMamba) |
| num_encoder_layers | 12 (split: 6 + 6) |
| num_decoder_layers | 4 |
| d_ffn | 1024 |
| dropout | 0.1 |
| Mamba: d_state / expand / d_conv | 16 / 2 / 4 |
| Mamba: bidirectional | True |
| Learning rate | 0.001 |
| Optimizer | Adam (betas: 0.9, 0.98) |
| Batch size (per GPU) | 16 |
| Grad accumulation | 1 |
| Effective batch | 16 * 2 GPUs = 32 |
| Dynamic batching | max_batch_length_train=1050 |
| Scheduler | Noam (warmup=25,000 steps) |
| CTC weight (training) | 0.3 |
| CTC weight (decoding) | 0.40 |
| LM weight (decoding) | 0.60 |
| Label smoothing | 0.0 |
| Precision | bf16 |
| Seed | 7775 |
| Epochs | 300 |
| Early stopping | patience=30, warmup=50 |

### H-Mamba-specific parameters (small)

| Parameter | N=1 | N=2 | N=3 | N=4 |
|-----------|-----|-----|-----|-----|
| hmamba_split_idx | 6 | 6 | 6 | 6 |
| hmamba_target_N | 1.0 | 2.0 | 3.0 | 4.0 |
| hmamba_headdim | 36 | 36 | 36 | 36 |
| hmamba_dc_loss_weight | 0.0 | 5.0 | 6.5 | 7.5 |
| hmamba_warmup_epochs | 0 | 15 | 20 | 20 |
| hmamba_gumbel_start | 1.0 | 1.0 | 1.0 | 1.0 |
| hmamba_gumbel_end | 1.0 | 0.3 | 0.3 | 0.3 |
| hmamba_gumbel_anneal_epochs | 1 | 30 | 35 | 35 |

### SLURM overrides (small)

The SLURM scripts override certain YAML parameters:

| Parameter | YAML value | SLURM override |
|-----------|-----------|----------------|
| batch_size | 16 | 24 |
| max_batch_length_train | 1050 | 1200 |
| max_batch_length_val | 100 | 120 |

---

## 28. Large Model Configurations

All large models share these base parameters (matching conmamba_large.yaml):

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| nhead | 8 (unused by BiMamba) |
| num_encoder_layers | 12 (split: 6 + 6) |
| num_decoder_layers | 6 |
| d_ffn | 2048 |
| dropout | 0.1 |
| Mamba: d_state / expand / d_conv | 16 / 2 / 4 |
| Mamba: bidirectional | True |
| Learning rate | 0.0008 |
| Optimizer | AdamW (betas: 0.9, 0.98) |
| Batch size (per GPU) | 16 |
| Grad accumulation | 8 |
| Effective batch | 16 * 2 GPUs * 8 = 256 |
| Dynamic batching | max_batch_length_train=600 |
| Scheduler | Noam (warmup=3,750 steps) |
| CTC weight (training) | 0.3 |
| CTC weight (decoding) | 0.40 |
| LM weight (decoding) | 0.60 |
| Label smoothing | 0.1 |
| Precision | bf16 |
| Seed | 3407 |
| Epochs | 300 |
| Early stopping | patience=30, warmup=50 |

### H-Mamba-specific parameters (large)

| Parameter | N=1 | N=2 | N=3 | N=4 |
|-----------|-----|-----|-----|-----|
| hmamba_split_idx | 6 | 6 | 6 | 6 |
| hmamba_target_N | 1.0 | 2.0 | 3.0 | 4.0 |
| hmamba_headdim | 128 | 128 | 128 | 128 |
| hmamba_dc_loss_weight | 0.0 | 5.0 | 6.5 | 7.5 |
| hmamba_warmup_epochs | 0 | 15 | 20 | 20 |
| hmamba_gumbel_start | 1.0 | 1.0 | 1.0 | 1.0 |
| hmamba_gumbel_end | 1.0 | 0.3 | 0.3 | 0.3 |
| hmamba_gumbel_anneal_epochs | 1 | 30 | 35 | 35 |

Note: hmamba_headdim is 128 for large (512/128 = 4 heads) vs 36 for small (144/36 = 4 heads).
Both give 4 EMA heads.

### SLURM overrides (large)

| Parameter | YAML value | SLURM override |
|-----------|-----------|----------------|
| batch_size | 16 | 16 (no change) |
| max_batch_length_train | 600 | 600 (no change) |
| max_batch_length_val | 100 | 100 (no change) |

Large models are more memory-constrained, so no overrides are needed.

---

## 29. SLURM Job Configuration

### Common SLURM parameters

| Parameter | Small Models | Large Models |
|-----------|-------------|-------------|
| Partition | preempt | preempt |
| GPUs | 2x A6000 | 2x A6000 |
| CPUs | 16 per task | 16 per task |
| Memory | 128 GB | 256 GB |
| Max walltime | 14 days | 14 days |
| Requeue | Enabled | Enabled |
| Email | anshulk@andrew.cmu.edu | anshulk@andrew.cmu.edu |

### Environment setup (all scripts)

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB__SERVICE_WAIT=300
export WANDB_MODE=offline
```

`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` limits the maximum memory block size
in PyTorch's CUDA allocator to 512 MB. This reduces memory fragmentation, which is
important for DC because the variable sequence lengths create many differently-sized
allocations.

`WANDB_MODE=offline` saves WandB logs locally instead of uploading to the cloud. This
avoids network issues on cluster nodes without internet access.

### Training command (example: small N=2)

```bash
torchrun --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    --nnodes=1 \
    --rdzv_endpoint=localhost:$((RANDOM + 20000)) \
    train_s2s_hmamba.py hparams/S2S/hmamba_small_N2.yaml \
    --distributed_launch \
    --data_folder /data/user_data/anshulk/hnet_asr/LibriSpeech \
    --output_folder /data/user_data/anshulk/hnet_asr/results/hmamba_small_N2 \
    --batch_size 24 \
    --max_batch_length_train 1200 \
    --max_batch_length_val 120 \
    --use_wandb True \
    --precision bf16
```

### Result directories

```
/data/user_data/anshulk/hnet_asr/results/
    hmamba_small_N1/
    hmamba_small_N2/
    hmamba_small_N3/
    hmamba_small_N4/
    hmamba_large_N1/
    hmamba_large_N2/
    hmamba_large_N3/
    hmamba_large_N4/
```

---

## 30. The 100-Hour Pilot Study

Before running the full 960h experiments, a pilot study was conducted on LibriSpeech
train-clean-100 (100 hours) using the small model.

### Pilot results

| Config | Compression | test-clean WER | test-other WER |
|--------|-------------|---------------|---------------|
| H-Mamba Small N=2 (100h) | 50% | 5.96 | 16.35 |
| H-Mamba Small N=3 (100h) | 67% | 7.80 | 21.36 |
| H-Mamba Small N=4 (100h) | 75% | 7.35 | 19.71 |

### The N=3 anomaly

N=3 was the worst performing of all three compression levels — it underperformed both
N=2 (5.96/16.35) and even the more aggressive N=4 (7.35/19.71) by 0.45–1.65% absolute
on test-clean. This is counterintuitive: less compression (N=3) should give better WER
than more compression (N=4).

The anomaly is attributed to **inconsistent learned boundaries**. N=3's intermediate
compression target (keep 1-in-3 frames) sits in an awkward regime where the routing
module struggles to find stable boundary positions. This gives Stage-1 Mamba noisier
inputs than either N=2 (where half the frames are kept, giving more context) or N=4
(where the stronger DC loss weight forces more decisive boundary placement).

Additional context from the error breakdown:
- N=3 test-clean: 514 ins / 623 del / 2,964 sub (SER 56.56%)
- N=3 test-other: 1,373 ins / 1,722 del / 8,088 sub (SER 80.54%)
- RTF: 0.0007, peak VRAM: 14,142 MB

Two critical bugs discovered in the code audit (see Section 31) also contributed:

1. **Bug 1 (ratio_loss zero gradient)**: The ratio_loss used `boundary_mask.float().mean()`,
   which is a boolean-to-float conversion with zero gradient. This meant `boundary_bias`
   received no gradient from the strongest loss term (weight=10.0). The compression ratio
   drifted instead of converging to the target.

2. **Bug 2 (wrong residual)**: The residual connection was `stage0_out + residual +
   expanded_states` instead of `expanded_states * ste_func(selected_probs) + residual`.
   This meant the routing module received no gradient from the ASR loss.

With both bugs, the routing module was barely learning. Compression was inconsistent
and the boundary positions were semi-random.

Both bugs are now fixed and gradient flow is verified (see Section 32). The N=3 anomaly
is flagged as something worth watching in the 960h runs — if it persists with fixed
code, it becomes a real finding worth discussing in the paper (boundary consistency
vs. compression depth).

### Pilot takeaways

1. The DC mechanism works — it learns to compress speech sequences
2. N=2 achieves reasonable WER on 100h (5.96 test-clean, vs ~4.5 for ConMamba baseline)
3. The gap between N=2 and N=4 is ~1.4% absolute on test-clean
4. N=3 is the worst at 7.80/21.36 — worse than both N=2 and N=4
5. The N=3 anomaly suggests boundary consistency matters more than compression depth;
   if it persists at 960h (with bug fixes), it is a paper-worthy finding

### Why 960h will be different

The 100h pilot used train-clean-100 only. The 960h experiments use the full training
set (960h), which provides:
- 10x more training data for the routing module to learn from
- More diverse speaking styles and acoustic conditions
- Better generalization to test-other (which contains noisy speech)

We expect the WER gap between compressed and uncompressed models to be smaller at
960h because the routing module has more data to learn from.

---

## 31. Bug Fixes and Code Audit

Two independent code audits were conducted in March-April 2026. Nine bugs were found
and fixed. The three most critical bugs directly affected the Dynamic Chunking
mechanism.

### Bug 1: ratio_loss zero gradient (HIGH severity)

**File**: `HMambaEncoder.py`, load_balancing_loss function

**Problem**: The ratio_loss computed `actual_ratio = boundary_mask.float().mean()`.
`boundary_mask` is created by `boundary_hard[..., 1] > 0.5`, which is a comparison
operation that produces a boolean tensor. Converting a boolean to float gives a tensor
with no gradient history. This means `boundary_bias` (the parameter that controls the
compression ratio) received zero gradient from the ratio_loss, which has the highest
weight (10.0) in the combined loss.

**Fix**: Added `boundary_hard` field to RoutingModuleOutput. In load_balancing_loss,
use `boundary_hard[..., 1].mean()` (which has STE gradients) instead of
`boundary_mask.float().mean()`.

**Verification**: After fix, boundary_bias gradient norm went from 0.0 to ~2.58.

### Bug 2: Wrong residual connection (HIGH severity)

**File**: `HMambaEncoderWrapper.py`, forward method

**Problem**: The residual connection was:
```python
output = stage0_out + residual + expanded_states
```
This was wrong because:
1. stage0_out was added directly (it should only appear via residual_proj)
2. The STE multiplication was missing, so the routing module received no gradient
   from the ASR loss

**Fix**:
```python
output = expanded_states.float() * ste_func(router_output.selected_probs) + residual
```

**Verification**: After fix, selected_probs receives non-zero gradients from the ASR loss.

### Bug 3: Eval crash with standalone evaluation (HIGH severity)

**File**: `train_s2s_hmamba.py`, on_evaluate_start method

**Problem**: When running evaluation with `--skip_train`, on_evaluate_start called
`find_checkpoints()` and `load_state_dict()` before _wrap_encoder_with_hmamba(). The
checkpoint contains HMambaEncoderWrapper parameters (routing_module, residual_proj, etc.)
but the model still has the original ConMamba encoder. The state_dict keys do not match,
causing a crash.

**Fix**: Check `_hmamba_initialized` flag and call `_wrap_encoder_with_hmamba()` before
loading checkpoints.

### Bug 4: tokenizer global variable (MEDIUM severity)

**Problem**: `tokenizer.decode_ids()` used a global variable instead of `self.tokenizer`.
**Fix**: Changed to `self.tokenizer.decode_ids()`.

### Bug 5: Early stopping never triggers (MEDIUM severity)

**Problem**: `epoch_counter.update_metric()` was never called, so EpochCounterWithStopper
never updated its patience counter.
**Fix**: Added `self.hparams.epoch_counter.update_metric(stage_stats["ACC"])` after
checkpoint save.

### Bug 6: Missing OOM handler (MEDIUM severity)

**Problem**: No OOM recovery. DC variable memory usage could crash training.
**Fix**: Added fit_batch() override with try/except (see Section 24).

### Bug 7: Incomplete optimizer param fallback (MEDIUM severity)

**Problem**: When DC params were added manually to the optimizer, only routing_module
parameters were included. residual_proj and dechunk_layer were missing.
**Fix**: Include all DC params:
```python
dc_params = list(self.hmamba_encoder.routing_module.parameters()) + \
            list(self.hmamba_encoder.residual_proj.parameters()) + \
            list(self.hmamba_encoder.dechunk_layer.parameters())
```

### Bug 8: Dead variable (LOW severity)

**Problem**: A redundant `dc_loss_weight` variable was computed but unused.
**Fix**: Removed the dead variable.

### Bug 9: Missing dependencies (ENV)

**Problem**: tensorboard and psutil were not installed in the conda environment.
**Fix**: `pip install tensorboard psutil`.

---

## 32. Gradient Flow Verification

After fixing all bugs, gradient flow was verified end-to-end. A single training step
was run with a small batch, and the gradient norms for all routing module parameters
were checked:

```
Parameter              Gradient Norm    Status
─────────────────────  ──────────────  ──────
q_proj_layer.weight    95.28           OK (receives gradients from both ASR + DC loss)
k_proj_layer.weight    93.14           OK (receives gradients from both ASR + DC loss)
temperature            10.49           OK (controls decision sharpness)
boundary_bias           2.67           OK (was ZERO before Bug 1 fix)
residual_proj.weight   2878.5          OK (high norm expected — large projection)
```

All five routing parameters receive non-zero gradients from both the ASR loss (via STE)
and the DC loss (via Gumbel-Softmax). This confirms that the routing module can now
learn from both signals.

### What the gradients mean

- **q_proj and k_proj** (large norms ~93-95): These are the largest parameters and
  receive the strongest gradients. The ASR loss pushes them to produce similarity
  patterns that preserve recognition accuracy. The DC loss pushes them to produce
  patterns that match the target compression ratio.

- **temperature** (norm ~10.5): Learning how sharp to make boundary decisions. The
  gradient from BCE loss and entropy loss compete to set the optimal sharpness.

- **boundary_bias** (norm ~2.67): This is the parameter that most directly controls
  the compression ratio. Before Bug 1 fix, this was ZERO — the bias was frozen at its
  initial value of 1.0, meaning the model could never learn to compress beyond whatever
  the initial bias allowed.

- **residual_proj** (norm ~2878.5): Large norm because this is a D*D weight matrix
  that processes every frame. The gradient tells it what information from Stage 0 is
  useful to preserve in the residual.

---

## 33. Known Deferred Issues

### chunk_mask not passed to Stage 1

The chunk_mask (validity mask for the compressed sequence) is not passed to the Stage 1
layers. This means Stage 1 layers process padded positions as if they were real frames.

Why this is acceptable:
1. ConmambaEncoderLayer explicitly sets `conv_mask = None` at line 316 of Conmamba.py,
   ignoring any mask that would be passed
2. BiMamba does not use attention masks — it is a recurrent model that processes all
   positions sequentially
3. The padded positions contain real gathered features (from the sort-and-gather in
   ChunkLayer), not random noise
4. Dynamic batching groups sequences of similar length, minimizing padding

### BCE loss includes position 0

The BCE loss in load_balancing_loss includes position 0, which is always a boundary
(hardcoded probability 1.0). This frame contributes zero gradient (log(1/(1-1)) is
handled by clamping, but the gradient with respect to the routing module is zero
because position 0 bypasses the cosine similarity computation). The inclusion
inflates the loss value slightly but has no training impact.

### Positional encoding and compression interaction

The system uses `fixed_abs_sine` positional encoding, which is added to the encoder
input **before** Stage 0:

```python
# In TransformerASR.forward():
src = self.custom_src_module(src)               # CNN frontend + linear projection
src = src + self.positional_encoding(src)       # Add sinusoidal position embeddings
encoder_out, _ = self.encoder(src=src, ...)     # Encoder processes pos-encoded input
```

The positional information is baked into the hidden states before the DC mechanism sees
them. This means:

1. **Stage 0** processes frames with correct positional information — each frame knows
   its absolute position in the utterance.
2. **The routing module** computes cosine similarity on position-aware hidden states.
   This is beneficial: transitions at position 10 vs. position 500 may have different
   significance, and the routing module can learn this.
3. **Stage 1** processes the compressed sequence with positional information inherited
   from the boundary frames' original positions. The compressed frames retain their
   original positional encoding, but the *relative* positions are distorted — frame 5
   might be adjacent to frame 15 in the compressed sequence. BiMamba is tolerant of
   this because it is a recurrent model (not attention-based) and does not explicitly
   use positional encodings internally. The SSM state simply processes whatever sequence
   it receives.
4. **DeChunk expansion** copies each boundary frame's features (including its baked-in
   positional encoding) to all positions in its chunk. Non-boundary positions get the
   nearest boundary frame's positional information, not their own. The **residual
   connection** mitigates this: the residual from Stage 0 output carries the correct
   positional encoding for every frame and is added back to the expanded output.

This is not an explicit design decision but a consequence of how SpeechBrain adds
positional encoding outside the encoder. If positional encoding were added per-layer
(as in some Transformer implementations), the interaction with DC would require
explicit handling.

### Streaming / online ASR incompatibility

The H-Mamba Dynamic Chunking mechanism is **incompatible with streaming ASR**:

1. **BiMamba is bidirectional**: every encoder layer processes the full utterance in both
   forward and backward directions. There is no causal variant.
2. **The routing module uses look-ahead**: the cosine similarity between frame t and
   frame t+1 requires the full utterance to be available. There is no streaming-compatible
   boundary detection.
3. **The DeChunk EMA expansion operates on the full compressed sequence**: it requires
   all boundary frames to be known before expansion can begin.
4. **The Transformer decoder's cross-attention** attends to all L encoder output frames.

This is by design — the project targets offline ASR (the full utterance is available).
A streaming variant would require replacing BiMamba with unidirectional Mamba, using
CIF-style left-to-right boundary detection instead of cosine similarity, and chunk-based
processing in the DeChunk layer. This is out of scope for the current work.

---

## 34. The Alternative Loss Formula

The official H-Net implementation uses a simpler single-term load balancing loss:

```python
# Official H-Net formula (hnet/utils/train.py):
true_ratio = boundary_mask.float().mean()
average_prob = tokenized_prob.float().mean()
loss = ((1 - true_ratio) * (1 - average_prob) +
        true_ratio * average_prob * (N-1)) * N / (N-1)
```

This formula has been proven to work across text (English, Chinese), code, and DNA in
the H-Net paper. It is cleaner and provides a single gradient signal.

### Why we use the 5-term loss instead

Our 5-term loss was developed before the H-Net code was available. It works correctly
now that Bug 1 is fixed (gradients flow through all terms). The 5-term loss provides
more diverse gradient signals, which may help in the speech domain where the input
sequences are much longer than in text (hundreds of frames vs. dozens of tokens).

### Fallback plan

If compression convergence is slow or unstable during 960h training, the first
intervention is to switch to the official H-Net formula. This is a single-line change
in `load_balancing_loss()`.

---

## 35. Package Upgrade and API Patches

### Why we upgraded

The original codebase used mamba-ssm 1.1.3 + causal-conv1d 1.1.3 + PyTorch 2.0.1. This
stack only included Mamba-1 ops. The DeChunkLayer's EMA expansion tried to use
`mamba_chunk_scan_combined` (a Mamba-2 SSD kernel), but it doesn't exist in mamba-ssm 1.x.
The code fell back to a pure PyTorch sequential loop.

We upgraded to get the Mamba-2 kernel and newer CUDA optimizations:

| Package | Before | After |
|---------|--------|-------|
| torch | 2.0.1+cu118 | 2.1.1+cu118 |
| torchaudio | 2.0.2+cu118 | 2.1.1+cu118 |
| triton | 2.0.0 | 2.1.0 |
| mamba-ssm | 1.1.3.post1 | 2.0.3 |
| causal-conv1d | 1.1.3.post1 | 1.4.0 |

### causal-conv1d 1.4.0 API changes

causal-conv1d 1.4.0 changed the C++ CUDA kernel signatures. The file
`Mamba-ASR/modules/mamba/selective_scan_interface.py` was patched at all 9 call sites.

**Forward (`causal_conv1d_fwd`)** — 6 call sites:

```python
# Old (5 args):
causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, True)

# New (7 args — added initial_states, final_states_out):
causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, None, None, None, True)
```

**Backward (`causal_conv1d_bwd`)** — 3 call sites:

```python
# Old (7 args, 3 returns):
dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
    x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, True
)

# New (10 args, 4 returns — added initial_states, dfinal_states, return_dinitial_states):
dx, dconv1d_weight, dconv1d_bias, _ = causal_conv1d_cuda.causal_conv1d_bwd(
    x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
)
```

The 4th return value (`dinitial_states`) is discarded with `_` since we don't use
initial states.

### Mamba-2 Triton kernel crash

After the upgrade, `mamba_chunk_scan_combined` imported successfully but crashed
at runtime with a Triton JIT assertion:

```
python: /project/lib/Analysis/Allocation.cpp:40:
Assertion `!(srcMmaLayout && dstMmaLayout) &&
"Unexpected mma -> mma layout conversion"' failed.
Aborted (core dumped)
```

This is a known Triton 2.1.0 bug in the MLIR→PTX lowering pass. It triggers on
Ampere GPUs (A6000 sm_86, L40S sm_89) when JIT-compiling the SSD combined kernel.
The fix requires triton >= 2.2.0, which needs PyTorch >= 2.2.

**Resolution**: Disabled the Mamba-2 Triton kernel in `HMambaEncoder.py`
(`MAMBA_KERNEL_AVAILABLE = False`). The DeChunk EMA uses the PyTorch fallback instead.
This only affects the expansion step — the main encoder Mamba-1 layers continue to
use optimized CUDA kernels from causal-conv1d and selective_scan.

### What uses optimized kernels vs. fallbacks

| Component | Kernel | Status |
|-----------|--------|--------|
| BiMamba encoder (Stage 0 + Stage 1) | causal_conv1d_cuda + selective_scan_cuda | Optimized CUDA |
| Mamba decoder | causal_conv1d_cuda + selective_scan_cuda | Optimized CUDA |
| DeChunk EMA expansion | mamba_chunk_scan_combined (Triton) | PyTorch fallback |
| RoutingModule, ChunkLayer | Pure PyTorch | N/A (no kernel needed) |

### OOM protection

The training script includes OOM handling in `fit_batch()` (train_s2s_hmamba.py):

```python
def fit_batch(self, batch):
    try:
        loss = super().fit_batch(batch)
        return loss
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning("[H-Mamba] CUDA OOM — skipping batch, clearing cache.")
            for p in self.modules.parameters():
                if p.grad is not None:
                    del p.grad
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.device)
        raise
```

This catches OOM errors from variable-length batches (DC adds memory overhead that
depends on the number of boundary frames). The batch is skipped and training continues.

### GPU utilization (smoke test observations)

From the smoke test (Small N=2, single A6000, train-clean-100):

| Metric | Value |
|--------|-------|
| Peak VRAM | 12,332 MB / 49,140 MB (25%) |
| Avg batch time | 157 ms |
| RTF | 0.0005 (2140x realtime) |
| Epoch time (100h data) | 185 seconds |

The small model (14.1M params) underutilizes a single A6000. The SLURM scripts
override `max_batch_length_train` to 1200 (vs yaml default 1050) to increase
throughput. The large model (115M params, d_model=512) will use significantly
more VRAM. With DDP across 2 GPUs, the effective per-GPU batch is halved.

---

## 36. Tensor Shape Reference

This section documents every tensor shape through the entire H-Mamba pipeline. All
shapes assume batch size B, original sequence length L, compressed length M, model
dimension D, and 2-class boundary distribution.

### CNN Frontend

| Tensor | Shape | Notes |
|--------|-------|-------|
| Raw audio | (B, T) | T = samples at 16kHz |
| Mel features | (B, T/160, 80) | 80 mel bins, 10ms hop |
| After CNN | (B, L, 640) | L ≈ T/640, 4x downsample |
| After projection | (B, L, D) | D = 144 or 512 |

### Stage 0

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input | (B, L, D) | From CNN frontend |
| After each layer | (B, L, D) | Same shape, richer features |
| stage0_out | (B, L, D) | Output of layer 5 |
| residual | (B, L, D) | residual_proj(stage0_out) |

### Routing Module

| Tensor | Shape | Notes |
|--------|-------|-------|
| q (query) | (B, L-1, D) | Projected frame t |
| k (key) | (B, L-1, D) | Projected frame t+1 |
| cos_sim | (B, L-1) | Cosine similarity |
| boundary_logits | (B, L-1) | After bias and temperature |
| boundary_prob_single | (B, L) | After sigmoid + pad |
| boundary_prob | (B, L, 2) | [P(no-boundary), P(boundary)] |
| boundary_hard | (B, L, 2) | Gumbel-Softmax hard output |
| boundary_mask | (B, L) | Boolean: True = boundary |
| selected_probs | (B, L, 1) | Probability of selected class |

### Chunk Layer

| Tensor | Shape | Notes |
|--------|-------|-------|
| token_idx | (B, L) | Position indices |
| sorted_indices | (B, L) | Boundaries sorted to front |
| chunked_states | (B, M, D) | M = max boundaries in batch |
| chunk_mask | (B, M) | Boolean: True = valid chunk |

### Stage 1

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input | (B, M, D) | From ChunkLayer |
| After each layer | (B, M, D) | Same compressed shape |

### DeChunk Layer

| Tensor | Shape | Notes |
|--------|-------|-------|
| p_chunked | (B, M) | Boundary probs for chunks |
| dt | (B, M) | log(1/(1-p)), EMA kernel input |
| EMA output | (B, M, D) | After EMA interpolation |
| chunk_indices | (B, L) | Maps positions to chunks |
| expanded_states | (B, L, D) | Back to original length |

### Output

| Tensor | Shape | Notes |
|--------|-------|-------|
| output | (B, L, D) | expanded * STE(probs) + residual |
| After LayerNorm | (B, L, D) | Final encoder output |

---

## 37. File Reference

### Core H-Mamba modules

| File | Lines | Purpose |
|------|-------|---------|
| `Mamba-ASR/modules/HMambaEncoder.py` | ~800 | RoutingModule, ChunkLayer, DeChunkLayer, load_balancing_loss, STE, HMambaEncoder |
| `Mamba-ASR/modules/HMambaEncoderWrapper.py` | ~440 | HMambaEncoderWrapper, create_hmamba_from_conmamba |
| `Mamba-ASR/modules/hmamba_logger.py` | ~884 | HMambaLogger, GPUMonitor, Timer, BatchMetrics, EpochMetrics |

### Training

| File | Lines | Purpose |
|------|-------|---------|
| `Mamba-ASR/train_s2s_hmamba.py` | ~758 | HMambaASR Brain class, dataio_prepare, main |

### Configurations

| File | Purpose |
|------|---------|
| `Mamba-ASR/hparams/S2S/hmamba_small_N1.yaml` | Small N=1 (control) |
| `Mamba-ASR/hparams/S2S/hmamba_small_N2.yaml` | Small N=2 (50% compression) |
| `Mamba-ASR/hparams/S2S/hmamba_small_N3.yaml` | Small N=3 (67% compression) |
| `Mamba-ASR/hparams/S2S/hmamba_small_N4.yaml` | Small N=4 (75% compression) |
| `Mamba-ASR/hparams/S2S/hmamba_large_N1.yaml` | Large N=1 (control) |
| `Mamba-ASR/hparams/S2S/hmamba_large_N2.yaml` | Large N=2 (50% compression) |
| `Mamba-ASR/hparams/S2S/hmamba_large_N3.yaml` | Large N=3 (67% compression) |
| `Mamba-ASR/hparams/S2S/hmamba_large_N4.yaml` | Large N=4 (75% compression) |

### SLURM scripts

| File | Purpose |
|------|---------|
| `slurm/hmamba_small_N1.sh` | Submit Small N=1 |
| `slurm/hmamba_small_N2.sh` | Submit Small N=2 |
| `slurm/hmamba_small_N3.sh` | Submit Small N=3 |
| `slurm/hmamba_small_N4.sh` | Submit Small N=4 |
| `slurm/hmamba_large_N1.sh` | Submit Large N=1 |
| `slurm/hmamba_large_N2.sh` | Submit Large N=2 |
| `slurm/hmamba_large_N3.sh` | Submit Large N=3 |
| `slurm/hmamba_large_N4.sh` | Submit Large N=4 |

### Encoder modules (shared with baselines)

| File | Purpose |
|------|---------|
| `Mamba-ASR/modules/Conmamba.py` | ConMamba encoder + Mamba decoder |
| `Mamba-ASR/modules/Conformer.py` | Conformer encoder |
| `Mamba-ASR/modules/TransformerASR.py` | Top-level model, Transformer decoder |
| `Mamba-ASR/modules/mamba/bimamba.py` | Bidirectional Mamba |
| `Mamba-ASR/modules/mamba/selective_scan_interface.py` | Selective scan ops, patched for causal-conv1d 1.4.0 |

---

## 38. What This Stage Will Establish

When all 8 experiments complete and are evaluated, this stage will answer:

### Primary questions

1. **Can H-Mamba match ConMamba WER with 50% compression?**
   - Success criterion: hmamba_small_N2 within 0.3% absolute of conmamba_small (2.22/5.56)
   - Success criterion: hmamba_large_N2 within 0.3% absolute of conmamba_large (2.27/5.12)

2. **What is the WER-compression Pareto frontier?**
   - N=1 (0% compression) → N=2 (50%) → N=3 (67%) → N=4 (75%)
   - At what compression ratio does WER start degrading significantly?

3. **Does the N=1 control match the baseline?**
   - If hmamba_small_N1 ≠ conmamba_small, the DC architecture itself has overhead
   - If hmamba_small_N1 ≈ conmamba_small, all WER changes are due to compression

### Secondary questions

4. **Does the compression pattern hold at both scales?**
   - If small and large show the same relative WER-compression tradeoff, the result
     is robust to model scale.

5. **Does the actual compression ratio match the target?**
   - N=2 should achieve ~50% compression. If it achieves 40% or 60%, the loss function
     needs adjustment.

6. **What do the learned boundaries look like?**
   - The boundary analysis (MFA alignment comparison, phone-class compression heatmap)
     will reveal whether the model learns linguistically meaningful boundaries.

### Results table (to be filled)

| Model | target_N | Actual Comp. | With LM (c/o) | No LM (c/o) | Status |
|-------|----------|-------------|---------------|-------------|--------|
| hmamba_small_N1 | 1.0 | — | — / — | — / — | Pending |
| hmamba_small_N2 | 2.0 | — | — / — | — / — | Pending |
| hmamba_small_N3 | 3.0 | — | — / — | — / — | Pending |
| hmamba_small_N4 | 4.0 | — | — / — | — / — | Pending |
| hmamba_large_N1 | 1.0 | — | — / — | — / — | Pending |
| hmamba_large_N2 | 2.0 | — | — / — | — / — | Pending |
| hmamba_large_N3 | 3.0 | — | — / — | — / — | Pending |
| hmamba_large_N4 | 4.0 | — | — / — | — / — | Pending |

### Baseline reference (from baseline_reproduction.md)

| Model | With LM (c/o) | No LM (c/o) |
|-------|---------------|-------------|
| conmamba_small | 2.22 / 5.56 | 3.34 / 8.47 |
| conmamba_large | 2.27 / 5.12 | 2.82 / 6.60 |

---

*Document last updated: April 1, 2026*
*All code references are from the current codebase after the April 2026 audit.*
*Smoke test v2 in progress (job 6921845). Full 960h training awaiting smoke test completion.*
