# Interpretable Hierarchical Speech Recognition via Dynamic Chunking: A Mamba–H-Net ASR

*Research proposal & experimental plan · drafted 2026-07-01 · LibriSpeech-960h*

---

## 1. Problem Statement — and Why It Matters

**The problem.** Every mainstream end-to-end ASR system hard-codes two granularities that the model is never allowed to question:

1. **A fixed acoustic frame rate.** A convolutional frontend downsamples audio to one vector every 40 ms (25 Hz) and *keeps it there*. Conformer [8] runs its entire stack at that single rate; even Zipformer [9] — which does use multiple rates — follows a **hand-designed, fixed** downsampling schedule chosen by the architect, not by the data.
2. **A fixed output vocabulary.** A BPE/word-piece or character inventory, frozen before training, defines the target units.

Neither granularity is learned from the speech signal, yet speech is intrinsically hierarchical and *variable-rate*: a burst consonant occupies ~20 ms while a stressed vowel spans 200 ms, and the linguistically meaningful units (phones → syllables → words) are nested and of wildly varying duration. Forcing a uniform rate spends equal compute on silence and on dense transitions, and divorces the model's internal representation from any interpretable linguistic tier.

**Why it matters.** If an ASR encoder could **learn where its own boundaries are** — compressing the sequence adaptively at multiple nested levels driven only by the recognition objective — we would gain three things at once: (i) compute allocated by information content rather than by the clock; (ii) an encoder whose intermediate representations *are* candidate linguistic units, opening a direct interpretability window ("what did the model decide a phone/word is?"); and (iii) a step toward genuinely tokenizer-free speech recognition. The open scientific question is whether such a **self-learned hierarchy can match the recognition accuracy of a hand-designed one** — because if it can, learned dynamic chunking becomes a strictly better default: same WER, less hand-tuning, more interpretability.

**What we propose.** DC-ASR inserts **H-Net-style dynamic chunking** [1] — a fully differentiable, content-based boundary predictor — between stacks of **Mamba-2** [6] state-space blocks, yielding an ASR encoder that learns its acoustic hierarchy end-to-end. We build two architecture types (1-stage and 2-stage), hold them at *matched overall compression*, evaluate three decoder read-outs of one hybrid model (CTC, AED, and joint CTC+AED), each under greedy and beam search, with and without an external LM, and benchmark WER against Conformer [8] and Zipformer [9]. We then dissect the learned boundaries against forced-aligned phone/word references to ask what linguistic structure emerges on its own.

---

## 2. Background

### 2.1 State-space models and Mamba
A **structured state-space model (SSM)** [7] maps an input sequence $x_t$ to output $y_t$ through a latent state $h_t$ governed by a linear dynamical system. In continuous time,
$$h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t).$$
Discretising with step $\Delta$ (zero-order hold) gives the recurrence
$$h_t = \bar A\,h_{t-1} + \bar B\,x_t, \qquad y_t = C\,h_t, \qquad \bar A = \exp(\Delta A),\; \bar B = (\Delta A)^{-1}(\exp(\Delta A) - I)\,\Delta B.$$
S4 [7] made this efficient for long sequences via a structured (diagonal-plus-low-rank) $A$. **Mamba** [5] made the parameters $(\bar A, \bar B, C, \Delta)$ **input-dependent** ("selective"), so the model can choose what to remember or forget per token, and provided a hardware-aware parallel scan giving $O(L)$ scaling with a global receptive field. **Mamba-2** [6] reformulated selective SSMs as a form of masked attention (the state-space duality, SSD), simplifying the layer and speeding it up. For ASR, $O(L)$ scaling with global context is well-suited to long acoustic sequences, and two recent systems confirm Mamba works for speech:
- **Samba-ASR** [4] uses Mamba as *both* encoder and decoder and reports strong WERs versus Transformer baselines.
- **Mamba streaming ASR + unimodal aggregation (UMA)** [3] pairs a Mamba encoder with a *single-level*, CTC-triggered frame-aggregation scheme. UMA is the closest existing "learned aggregation" in ASR — and its single-level, CTC-driven nature is exactly the contrast that motivates a *multi-level, self-learned* mechanism.

### 2.2 H-Net dynamic chunking
H-Net [1] makes discrete segmentation differentiable and trainable end-to-end. Per chunking stage:
- a **routing module** predicts a boundary probability from the cosine dissimilarity of adjacent hidden states,
$$p_t = \tfrac12\Big(1 - \tfrac{q_t^\top k_{t-1}}{\lVert q_t\rVert\,\lVert k_{t-1}\rVert}\Big) \in [0,1];$$
- a **downsampler** keeps positions where $p_t > 0.5$, compressing $L$ vectors to $L' < L$ chunk vectors;
- a **smoothing module** — an exponential moving average $\bar z_t = P_t \hat z_t + (1-P_t)\bar z_{t-1}$ — together with a **straight-through estimator** [18] makes the hard selection differentiable;
- a **ratio loss** $\mathcal L_{\text{ratio}}$ pushes the average boundary rate toward a target compression, preventing the trivial keep-all / drop-all solutions;
- an **upsampler / dechunker** restores the original length for the residual and output paths.

Stacked, this yields a U-Net-like **encoder → main network → decoder** hierarchy in which the main network runs on the compressed sequence. Critically, **H-Net already uses Mamba-2 as its backbone**, which is why the Mamba↔H-Net marriage is natural rather than forced. H-Net grew out of the byte-level / tokenizer-free LM line — MambaByte [21], SpaceByte [20], and Byte Latent Transformer [19] — which removes the tokenizer for *text*; DC-ASR carries the same idea to *acoustic frames*.

### 2.3 Decoders, LMs, and the interpretability toolkit we build on
End-to-end ASR heads come in three standard forms: **CTC** [10] (conditionally-independent frame labels with a blank), **attention/AED** [12] (autoregressive), and **transducer/RNN-T** [11]. A **hybrid CTC/attention** setup [12] trains both from one encoder and is the efficient way to obtain both heads. External **language models** are applied at decode time via shallow fusion or rescoring; Conformer [8] improves 2.1/4.3 → 1.9/3.9 (test-clean/other) with an LM, so LM and no-LM numbers must both be reported. For interpretability we lean on the *unsupervised acoustic-unit-discovery* tradition and its evaluation protocols — boundary precision/recall/F1 and R-value against forced alignment [15], and linear-probing of representations as in wav2vec 2.0 [16] / HuBERT [17] analyses.

---

## 3. Data

**LibriSpeech-960h only** [13]. All training uses the full 960-hour LibriSpeech training set (train-clean-100 + train-clean-360 + train-other-500); all WERs are reported on the four standard partitions **dev-clean / dev-other / test-clean / test-other**. This is the field's most-reported benchmark, which is exactly what the "competitive-with-Conformer/Zipformer" claim requires — both baselines publish LibriSpeech-960h numbers, so the comparison is apples-to-apples.

- **Input features.** 80-dim log-Mel filterbanks @ 100 Hz, global CMVN, SpecAugment [14].
- **Output units.** A 500-token BPE vocabulary for the attention head and a shared BPE/char set for CTC (unit choice itself is an ablation, since one motivation is reduced tokenizer dependence).
- **LM data.** The official LibriSpeech LM corpus (810M-word text) for an external Transformer/n-gram LM used only at decode time.
- **Forced alignment.** Montreal Forced Aligner [15] produces phone- and word-level time boundaries on the training/dev audio, providing the ground truth for the interpretability analyses (Section 7).

*Scope note.* The earlier draft proposed a cross-lingual morphology study (Turkish/Mandarin). Under the 960h-only constraint that is explicitly **out of scope** and moved to Limitations/Future Work (Section 11); it does not affect the core claims, which are all monolingual-English on LibriSpeech.

---

## 4. Architecture and Math

### 4.1 The Mamba–chunk sandwich
```
      80-dim logmel @ 100 Hz
              │
        [ Conv subsampling ×4 ]  → 25 Hz frame sequence, length L0
              │
   ┌──────────▼───────────────────────────────┐
   │  ENCODER stage (at length L_s)             │
   │    k_enc × Mamba-2 blocks                  │
   │        │                                   │
   │   [ Router ]  p_t = ½(1 − cos(q_t,k_{t−1}))│
   │   [ Downsample ] keep p_t>0.5 → L_{s+1}    │
   └──────────┬────────────────────────────────┘
              │   (Type B: repeat this stage once more; Type A: skip)
   ┌──────────▼───────────┐
   │  MAIN network         │  m × Mamba-2 blocks at the coarsest rate
   └──────────┬───────────┘
   ┌──────────▼────────────────────────────────┐
   │  DECODER stage (mirror)                    │
   │   [ Upsample / dechunk ] → L_s             │
   │   [ EMA smoothing + STE residual ]         │
   │    k_dec × Mamba-2 blocks                  │
   └──────────┬────────────────────────────────┘
              │  frame-rate encoder output
        [ ASR head: CTC and/or attention ]
```
"A few blocks of Mamba, then H-Net, then a few blocks of Mamba" is precisely the encoder-stage box; Type B nests two such stages.

### 4.2 Two architecture types (each Large and Small → 4 base encoders)
- **Type A — 1-stage:** `Mamba → H-Net → Mamba`. One chunking block. Hypothesis: boundaries near **phone/syllable** granularity; the acoustic analogue of UMA [3] but self-learned and not CTC-triggered.
- **Type B — 2-stage:** `Mamba → H-Net → Mamba → H-Net → Mamba`. Two chunking blocks: frames → units → word-ish chunks, main stack at the coarsest rate. H-Net's headline configuration ported to audio. Hypothesis: **stage-1 ≈ phone/syllable, stage-2 ≈ word**.

### 4.3 Compression level $N$ — the iso-compression design
$N$ is the **overall** downsampling factor: the encoder keeps a fraction $1/N$ of its input frames end-to-end (enforced by $\mathcal L_{\text{ratio}}$; the router places boundaries freely within that budget). The two types hit the same $N$ differently, so they are always compared at **matched overall compression**:
- **Type A** — its single block runs at per-block factor $N$.
- **Type B** — its two blocks each run at per-block factor $\sqrt N$, so the keep-fractions multiply to $1/N$. Per-block factors: $\{1,\sqrt2,\sqrt3,2\}$.

| $N$ (overall) | overall frames kept | Type A per-block | Type B per-block ($\sqrt N$) | Type B kept / block |
|---|---|---|---|---|
| $N=1$ | 100% | 1 (no-op) | 1 | 100% |
| $N=2$ | 50%  | 2 | $\sqrt2\approx1.41$ | 70.7% |
| $N=3$ | 33%  | 3 | $\sqrt3\approx1.73$ | 57.7% |
| $N=4$ | 25%  | 4 | 2 | 50% |

Two consequences stated up front:
- **$N=1$ is a no-compression control inside the architecture** — every H-Net block passes everything through, so both types reduce to a pure-Mamba encoder. A "chunking off" reference at zero extra engineering.
- **Iso-compression is the experimental point.** At each $N$, Type A applies one hard $N\times$ compression while Type B applies two gentle $\sqrt N\times$ compressions to the *same total budget*. A WER or interpretability gap between them therefore isolates the effect of **staging** (coarse-to-fine, two learned boundary sets), not of **how much** is discarded. Type B's gentler per-block ratios ($\le 2\times$) should also make its stages less collapse-prone than a per-block-$N$ scheme.

### 4.4 Training objective
The total loss combines the recognition loss and the per-stage ratio loss:
$$\mathcal L = \underbrace{\lambda_{\text{ctc}}\,\mathcal L_{\text{CTC}} + (1-\lambda_{\text{ctc}})\,\mathcal L_{\text{AED}}}_{\text{hybrid recognition [12]}} \;+\; \beta \sum_{s} \mathcal L_{\text{ratio}}^{(s)},$$
with $\mathcal L_{\text{CTC}}$ the CTC loss [10], $\mathcal L_{\text{AED}}$ the label-smoothed cross-entropy of the attention decoder, and $\mathcal L_{\text{ratio}}^{(s)}$ H-Net's compression-target regulariser at stage $s$. Gradients flow through the discrete boundary selection via the straight-through estimator [18]. Training one hybrid model thus yields **both** a CTC and an attention head; the external LM is added only at decoding.

### 4.5 Model sizes
Two capacity variants per type. The encoder is **bidirectional** (each Mamba-2 block runs a forward *and* a reversed scan — the right choice for offline recognition), which roughly doubles the per-block parameters, so sizes are set by the architecture and are **not matched to the Zipformer parameter counts**. Zipformer/Conformer serve as published **WER references**, not a size-controlled comparison: the goal is a working **H-Net×Mamba ASR** encoder — competitive WER *plus* an interpretable, self-learned acoustic hierarchy — not a win at equal parameters. Layer counts are a free hyperparameter (tuned for WER/compute); final sizes are reported as trained.

| Variant | $d_{\text{model}}$ (outer / main) | Mamba-2 layers (enc / main / dec) | ~Enc params (bidirectional) |
|---|---|---|---|
| **Small** | 384 / 512 | 4 / 12 / 4 | ~62M (measured) |
| **Large** | 512 / 768 | 6 / 18 / 6 | ~185M (measured) |

(Type B splits the encoder/decoder Mamba budget across its two stages at the same total depth, so Types A and B stay matched **to each other** at each size.)


---

## 5. Hypotheses

We state falsifiable hypotheses so the experiments have clear pass/fail conditions.

- **H1 (Competitive WER).** At a suitable compression level, DC-ASR reaches WER competitive with strong published LibriSpeech systems — within a small margin (target: ≤ 0.3 abs on test-clean, ≤ 0.5 on test-other) of published Zipformer [9] — in **both** the no-LM and +LM decoding settings. Zipformer is a published WER *reference*, not a size-matched control; DC-ASR is not constrained to equal parameters, so H1 is a **viability check** — the primary contributions are the learned-beats-fixed result (H2) and the emergent linguistic structure (H4). *Falsified if* the best DC-ASR trails published Zipformer by > 1.0 abs.
- **H2 (Learned beats fixed at equal compression).** At the same overall $N$, learned dynamic chunking yields lower WER than fixed-stride pooling to the same rate. *Falsified if* fixed pooling matches or beats learned chunking across all $N$.
- **H3 (Staging helps the hierarchy, not necessarily the WER).** At matched overall compression, Type B's two stages produce boundaries that align better with the phone→word hierarchy than Type A's single stage (higher word-boundary F1 at stage-2), even if WER is comparable. *Falsified if* Type B shows no interpretability advantage and no WER advantage over Type A.
- **H4 (Emergent linguistic structure).** Learned boundaries align with forced-aligned phone/word boundaries above a random-rate baseline, and stage depth correlates with linguistic tier (stage-1→phone/syllable, stage-2→word). *Falsified if* boundary F1 is at chance, i.e. the model chunks on acoustic energy/silence alone.
- **H5 (A compression sweet spot exists).** WER is roughly flat or improving from $N{=}1$ to a moderate $N$ (compute saved with little accuracy loss), then degrades past a threshold. *Falsified if* WER degrades monotonically from $N{=}1$ (any compression hurts).

---

## 6. Experiments & Interpretability

### 6.1 The experimental grid
Core grid: **{Type A, Type B} × {N=1,2,3,4} × {Small, Large}**, and each trained model is then **decoded** along **{CTC, AED, CTC+AED} × {greedy, beam} × {no-LM, +LM}** (the decoding protocol, §6.3). The three decoders are **read-outs of one hybrid model**, not three trainings — CTC and AED heads come from the same run (§4.4), so a single training fills an entire decode sub-block. Screening is at **Small** across all $N$; only the winning $(type, N)$ cells are promoted to **Large** for the headline table. That is 8 Small encoders (2 types × 4 $N$) + ~4 Large encoders (promoted) = **~12 training runs**, and each run is read out under the full decode matrix below (~10 meaningful WER numbers per model per test set).

### 6.2 Baselines — cited, not re-trained
Per the locked protocol we do **not** re-train any external system; DC-ASR is compared against **published** LibriSpeech-960h WERs, reporting **both no-LM and +LM** wherever the source gives them (exactly as we report DC-ASR). All numbers were cross-checked against the cited papers on 2026-07-01. `n/r` = the setting is not reported by that source — note that several modern encoders publish their *headline* WER **without** an external LM, so a blank +LM cell is often a design choice, not a gap.

**Table A — End-to-end supervised, LibriSpeech-960h, no extra data (the like-for-like comparison set).** WER = test-clean / test-other.

| Model | Ref | Enc. type | Decoder | Params | no-LM | + LM |
|---|---|---|---|---|---|---|
| QuartzNet-15×5 | [28] | 1D-conv (CTC) | CTC | 19M | 3.90 / 11.28 | 2.69 / 7.25 |
| Jasper DR-10×5 | [29] | 1D-conv (CTC) | CTC | 333M | 3.86 / — | 2.95 / — |
| Transformer-AED | [30,31] | Transformer | AED | 270M | 2.89 / 6.98 | 2.33 / 5.17 |
| LSTM-AED | [30,31] | LSTM | AED | 360M | 2.6 / 6.0 | 2.2 / 5.2 |
| Transformer (SpeechBrain) | [–] | Transformer | CTC+AED | 165M | n/r | 2.46 / 5.77 |
| ContextNet-L | [24] | 1D-conv + SE | RNN-T | ~112M | 2.1 / 4.6 | 1.9 / 4.1 |
| **Conformer-L** (orig.) | [8] | conv+attn (flat) | RNN-T | 118.8M | **2.1 / 4.3** | **1.9 / 3.9** |
| Conformer-L (reproduced) | [9] | conv+attn (flat) | — | ~122M | 2.46 / 5.55 | n/r |
| Conformer-M | [8] | conv+attn (flat) | RNN-T | 30.7M | 2.3 / 5.0 | n/r |
| Branchformer | [26] | parallel MLP/attn | AED | 116.2M | 2.4 / 5.5 | n/r |
| Squeezeformer-M | [25] | conv+attn (U-Net) | CTC | 55.6M | 2.56 / 6.50 | n/r |
| Squeezeformer-L | [25] | conv+attn (U-Net) | CTC | — | 2.47 / 5.97 | n/r |
| E-Branchformer-L | [27] | parallel MLP/attn | AED | ~149M | n/r | **1.81 / 3.65** |
| Zipformer-S | [9] | **hand-designed hier.** | CTC/AED | 23.2M | 2.42 / 5.73 | n/r |
| **Zipformer-M** | [9] | **hand-designed hier.** | CTC/AED | 65.6M | **2.21 / 4.79** | n/r |
| **Zipformer-L** | [9] | **hand-designed hier.** | CTC/AED | 148.4M | **2.00 / 4.38** | n/r |

**Table B — Self-supervised / extra-data (context only — NOT a like-for-like comparison; these use large unlabeled or weakly-labeled corpora beyond LS-960).**

| Model | Ref | Extra data | Params | no-LM | + LM |
|---|---|---|---|---|---|
| wav2vec 2.0 Large | [16] | LibriLight-60k (unlab.) | 317M | 2.1 / 4.6* | 1.8 / 3.3 |
| HuBERT X-Large | [17] | LibriLight-60k (unlab.) | ~1B | n/r | ~1.9 / 3.3 |
| WavLM Large | [32] | 94k h (unlab.) | 316M | n/r | ~1.8 / 3.2 |
| Whisper large-v3 | [33] | 680k–5M h (weak sup.) | 1.55B | ~2.0–2.7 (zero-shot) | n/r |

\*wav2vec2 "no-LM" here is the supervised-from-scratch Large; the 1.8/3.3 is with self-training + LM. Table-B rows marked `~` are the models' commonly-reported figures (arXiv IDs verified; exact WER not re-checked this session) — kept for landscape context, not used as a comparison bar.

**Two caveats we will state in the paper.**
- **Reproduction gap.** The *original* Conformer-L is 2.1/4.3 (no-LM); independent open-source re-implementations land near **2.46/5.55** [9]. When we say "match Conformer," the honest bar is the reproduced range — which the stronger encoders (Zipformer, E-Branchformer) clear, and which DC-ASR targets.
- **No-LM is the headline for the strongest encoders.** Zipformer and Squeezeformer report their best numbers *without* an external LM. DC-ASR should therefore target a strong **no-LM** result first, treating the LM as a separately-reported additional gain — precisely why every DC-ASR cell is run in both settings.

**Internal controls (configurations of DC-ASR itself, not external baselines — no re-implementation).** These are the *same* DC-ASR model with one knob changed, so they are part of our own grid rather than re-trained third-party systems: the **$N{=}1$ no-chunk** config (= pure-Mamba encoder, isolates the chunking contribution) and a **fixed-stride-pooling** variant at each rate. They stay in the grid because **H2** ("learned beats fixed at equal compression") is defined against them; a Mamba+UMA [3] control is included if budget allows.

### 6.3 Training and decoding protocol
**Training.** LibriSpeech-960h [13]; 80-d logmel @ 100 Hz; SpecAugment [14]; hybrid CTC/attention loss ($\lambda_{\text{ctc}}=0.3$) [12]; AdamW with warmup-decay; ~100–120 epochs; speed perturbation ×3. One hybrid model is trained per $(type, N, size)$ cell — **we do not train pure-CTC or pure-AED models separately** (the hybrid loss converges better and keeps attention aligned [12]). Compute on Babel (interactive `srun --gres=gpu:1`). Because baselines are **cited, not re-trained**, the entire compute budget is DC-ASR's own grid; the Large-960h DC-ASR runs are the budget driver (a few GPU-days each), which is why the Small variant screens the grid first.

**Decoding — three read-outs of the one trained model.** At test time the same hybrid model is decoded three ways, under both search strategies and both LM settings. WER = test-clean / test-other is reported for every meaningful cell.

| Read-out | Greedy | Beam ($B{=}10$) | +LM (beam only) | What it is for |
|---|---|---|---|---|
| **CTC head** | ✓ argmax-per-frame + collapse (fast/no-search ref) | ✓ prefix-beam | ✓ shallow fusion | speed/latency (non-autoregressive) **and** frame **spikes** for the interpretability cross-check vs. H-Net boundaries |
| **AED head** | (optional; weak, diagnostic only) | ✓ label-synchronous beam | ✓ shallow fusion / rescoring | isolates the attention head's fluency contribution |
| **Joint CTC+AED** | — (beam by nature) | ✓ one-pass joint beam [12] | ✓ shallow fusion / rescoring | **best-WER headline** vs. Conformer/Zipformer |

Conventions (settled): **greedy = beam width 1**, kept as the fast reference and a *peakiness diagnostic* (the greedy→beam gap measures how confident/peaky the model is); greedy is only meaningful for the **CTC** head, so AED and joint are **beam-only**. **Beam width $B{=}10$** by default (a small $B\in\{4,8,10,16\}$ sweep on dev to confirm the knee, then fixed). The external Transformer-LM enters via **shallow fusion** during beam search, so **+LM lives on the beam side only** — "greedy +LM" is not a standard cell; the no-LM column already covers greedy and beam. Report GFLOPs and RTF alongside WER (CTC-greedy gives the fastest RTF; joint-beam+LM the best WER — the accuracy/latency trade-off is itself a result).

### 6.4 Interpretability program (the scientific core — "analyse these learned things")
For each stage $s$ and compression $N$, the learned boundaries $\{t: p_t>0.5\}$ and chunk embeddings are the data:

1. **Boundary alignment.** Against MFA [15] phone/syllable/word boundaries, compute precision/recall/**F1** and **R-value** (tolerance ±20 ms). *Which tier does each stage snap to, and how sharply?* Baseline: random boundaries at the same rate.
2. **Chunk-identity probing.** Linear probes [16,17] on chunk embeddings predicting phone identity, phone class (voicing/place/manner), and word identity. *What information concentrates at each level?*
3. **Staging at fixed compression (H3).** Because Type A and Type B are held at the same overall $N$, directly compare their boundaries: *does splitting one $N\times$ step into two $\sqrt N\times$ steps yield cleaner phone-then-word tiers?*
4. **Emergence curve (H4).** Track boundary-F1 and probe accuracy over training epochs vs. the WER curve. *Does linguistic structure appear before, with, or after WER convergence — i.e. is it truly self-learned from the recognition signal?*
5. **Compression–accuracy–linguistics surface (H5).** Plot realised compression vs. WER vs. boundary-F1 across $N$. *Is there a rate where units become maximally linguistic at minimal WER cost?*
6. **Boundary robustness.** Measure how boundaries move under noise / speed-perturbation / silence insertion — do they track linguistic content or acoustic artefacts?

### 6.5 Deliverable figures
(F1) WER-vs-$N$ curves, DC-ASR vs. fixed-pooling vs. Zipformer, per type; (F2) boundary-F1-by-tier bar chart, Type A vs. Type B; (F3) an example utterance with learned stage-1/stage-2 boundaries overlaid on the spectrogram and the MFA phone/word tiers; (F4) emergence curves (boundary-F1 & WER vs. epoch); (F5) probe-accuracy-by-stage heatmap.

---

## 7. Possible Results

We describe the outcomes we expect and how each maps onto the hypotheses (illustrative targets, not measurements). The headline table would look like:

WER = test-clean / test-other; DC-ASR is reported in **both** no-LM and +LM columns, matching how each cell is actually decoded. Competitor rows are **published** values (Table A, §6.2), not re-runs.

| Row | Params | no-LM | + LM | role |
|---|---|---|---|---|
| Conformer-L (orig.) [8] | 118.8M | 2.1 / 4.3 | 1.9 / 3.9 | flat-encoder anchor |
| Conformer-L (reproduced) [9] | ~122M | 2.46 / 5.55 | n/r | honest reproduction bar |
| Zipformer-M [9] | 65.6M | 2.21 / 4.79 | n/r | hand-designed-hier foil |
| Zipformer-L [9] | 148.4M | 2.00 / 4.38 | n/r | strongest foil |
| E-Branchformer-L [27] | ~149M | n/r | 1.81 / 3.65 | +LM SOTA |
| **DC-ASR Type A, $N{=}2$, Large** | ~185M | *target ≈ Zipformer-M* | *target ≤ Zipformer-M* | **H1** |
| **DC-ASR Type B, $N{=}2$, Large** | ~185M | *target ≈ Zipformer-M* | *target ≤ Zipformer-M* | **H1 / H3** |
| DC-ASR $N{=}1$ (no-chunk), Large | ~185M | internal Mamba ref | internal Mamba ref | control |
| DC-ASR fixed-pool $N{=}2$, Large | ~185M | internal fixed-rate ref | internal fixed-rate ref | **H2** |

The claim succeeds if the two bold DC-ASR rows land within the small margin of H1 (≤0.3 test-clean, ≤0.5 test-other vs. published Zipformer-M, used as a WER reference — not a size-matched control) in **both** the no-LM and +LM columns, while beating the fixed-pool control (H2).

**Expected qualitative outcomes.**
- **On WER (H1/H2/H5):** DC-ASR at moderate compression ($N{=}2$) is competitive with Zipformer at matched params and cheaper in GFLOPs; learned chunking beats fixed pooling at equal rate; WER is flat-to-slightly-better from $N{=}1$→$2$, then degrades by $N{=}4$ as too many frames are dropped — locating a sweet spot.
- **On structure (H3/H4):** boundary-F1 is well above the random-rate baseline; Type B's stage-2 aligns with word boundaries better than Type A's single stage; probes recover phone identity best from stage-1 chunks and word identity best from stage-2 chunks; emergence curves show boundary structure sharpening *alongside* WER, evidencing self-learned segmentation.
- **A negative result is still publishable:** if boundaries track silence/energy rather than phones (H4 falsified), that is an informative finding about what the ASR objective does and does not induce — reported honestly against the MFA baseline.

---

## 8. Novelty, Overlap, Previous Literature & Related Work

*Novelty search performed 2026-07-01 over arXiv full-text (multiple query families), Semantic Scholar (H-Net's full 64-paper forward-citation set enumerated), and OpenAlex.*

### 8.1 The exact combination does not exist
- `"H-Net" AND speech` on arXiv → **0 hits**. `"Mamba" AND "H-Net"` → **1 hit**, and it is a finance model (orderbook events), not speech.
- Of the **64 papers citing H-Net** [1], **8 touch audio — and all eight are generation-side**: neural audio codecs / TTS / variable-frame-rate tokenizers (e.g. DyCAST, Elastic Time, audio-token compression) and omnimodal token compression (DASH). **None is an ASR encoder that learns a multi-level acoustic hierarchy**, and none makes the learned units an object of interpretability study.
- H-Net's dynamic chunking has already been ported to **genomics** (dnaHNet, LDARNet) and **finance** — speech is the conspicuous open gap, which both confirms reach and argues for moving promptly.

### 8.2 Naming-collision warning (address head-on in the paper)
"**Dynamic chunk(ing)**" already has an established, **different** meaning in streaming ASR — the WeNet-style *variable streaming chunk size for latency control* (e.g. Context-Aware Dynamic Chunking for Tibetan ASR [22]; TC-BiMamba [23]). Those vary a fixed window's size; they are **not** H-Net's learned *content-based* chunking. The paper must disambiguate on page 1 (we use **"learned acoustic chunking"** / "content-based hierarchical chunking") and cite these precisely to draw the line.

### 8.3 Closest legitimate neighbours and how DC-ASR differs
| Prior work | What it does | How DC-ASR differs |
|---|---|---|
| **H-Net** [1] | Learned dynamic chunking for text/DNA/code; Mamba-2 backbone | First application to **speech/ASR**; acoustic-frame boundaries; interpretability of learned units |
| **H-Net++** [2] | H-Net for morphologically-rich **text** (Persian) | Acoustic, not text (motivates future cross-lingual work) |
| **Mamba+UMA ASR** [3] | **Single-level**, CTC-triggered frame aggregation | **Multi-level**, self-learned, differentiable; UMA is a key baseline |
| **Samba-ASR** [4] | Mamba encoder+decoder ASR, **no hierarchy** | Adds learned hierarchical compression + its analysis (= our $N{=}1$ baseline) |
| **Zipformer** [9] | SOTA ASR; **hand-designed** U-Net multi-rate encoder | Same hierarchical idea, but **learned & content-adaptive**; the key WER foil |
| **Conformer** [8] | SOTA conv-augmented Transformer; **flat** rate | The flat-encoder WER anchor |
| Byte-level LMs [19,20,21] | Tokenizer-free **text** modelling | The lineage DC-ASR extends into audio |
| DyCAST / Elastic Time / DASH | Variable-rate audio **codec / TTS / token compression** | Recognition-side; interpretability, not generation efficiency |
| Tibetan-DC / TC-BiMamba | **Streaming** variable *window* size | Content-based *learned boundaries* — a different mechanism (§8.2) |

### 8.4 Net assessment
- **Architecture novelty: high** — no H-Net×Mamba ASR exists; the Mamba-sandwich-per-stage instantiation for audio is new.
- **Interpretability novelty: high** — "what linguistic hierarchy does an ASR encoder self-discover when allowed to chunk freely?" is unasked in this form; it links H-Net to the acoustic-unit-discovery literature.
- **Competitiveness angle: sharp & defensible** — because Zipformer already proved a *hand-designed* hierarchy beats a flat encoder at lower FLOPs, "a *learned* hierarchy matches the hand-designed one" is a crisp, falsifiable claim with an obvious baseline, not a vague SOTA chase.
- **Scoop risk: real** — genomics/finance ports appeared within a year; publish the analysis and the learned-vs-fixed comparison, not just a WER number.

---

## 9. Contributions
1. **DC-ASR** — the first ASR encoder built from Mamba-2 blocks interleaved with H-Net dynamic chunking, in two types (1-stage `Mamba–H-Net–Mamba`, 2-stage `Mamba–H-Net–Mamba–H-Net–Mamba`), each in Large and Small variants, read out three ways (CTC, AED, joint CTC+AED) under greedy/beam search, with and without an LM.
2. **An iso-compression study** — via the $\sqrt N$ split, Type A and Type B are compared at *matched overall compression*, isolating the effect of *staging* from the amount of information discarded — a controlled comparison prior hierarchical-ASR work has not offered.
3. **A "no-WER-cost" result on LibriSpeech-960h** — evidence that a learned, content-adaptive hierarchy matches Conformer/Zipformer at matched params, making self-learned chunking a better default than hand-set frame rates + tokenizers.
4. **An interpretability suite** — boundary-alignment, chunk-probing, and emergence-curve analyses showing what linguistic tiers the stages discover and whether the structure is self-learned from the recognition objective.

---

## 10. Limitations
- **English-only / single corpus.** LibriSpeech-960h only; no cross-lingual or morphology-rich evaluation (the H-Net++ [2] angle) and no far-field/conversational/noisy data. Generality is therefore untested; the cross-lingual study is deferred to future work.
- **Read speech.** LibriSpeech is clean read audiobooks; learned boundaries may exploit its regular prosody and clean silences, so the interpretability findings may not transfer to spontaneous speech.
- **Two types, not a full architecture search.** We fix 1- and 2-stage designs and one router (cosine) to keep the study controlled; deeper hierarchies and alternative routers are out of scope.
- **Non-streaming.** The router as specified is non-causal; a streaming/causal variant is future work (and connects to [3]).
- **Interpretability ≠ causation.** Probes and boundary-F1 show *correlation* with linguistic tiers; they do not prove the model *uses* units the way humans do. We report them as evidence, not proof.
- **Compute-bounded grid.** Only promising Small cells are promoted to Large; some conclusions rest on Small-scale trends assumed to hold at scale.

---

## 11. References

[1] S. Hwang, B. Wang, A. Gu. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net). arXiv:2507.07955, 2025. https://arxiv.org/abs/2507.07955

[2] M. Zakershahrak, S. Ghodratnama. *H-Net++: Hierarchical Dynamic Chunking for Tokenizer-Free Language Modelling in Morphologically-Rich Languages.* arXiv:2508.05628, 2025. https://arxiv.org/abs/2508.05628

[3] Y. Fang, X. Li. *Mamba for Streaming ASR Combined with Unimodal Aggregation.* arXiv:2410.00070, 2024. https://arxiv.org/abs/2410.00070

[4] S. A. G. Shakhadri, K. KR, K. B. Angadi. *Samba-ASR: State-of-the-Art Speech Recognition Leveraging Structured State-Space Models.* arXiv:2501.02832, 2025. https://arxiv.org/abs/2501.02832

[5] A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752, 2023. https://arxiv.org/abs/2312.00752

[6] T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). arXiv:2405.21060, 2024. https://arxiv.org/abs/2405.21060

[7] A. Gu, K. Goel, C. Ré. *Efficiently Modeling Long Sequences with Structured State Spaces* (S4). arXiv:2111.00396, 2022. https://arxiv.org/abs/2111.00396

[8] A. Gulati, J. Qin, C.-C. Chiu, et al. *Conformer: Convolution-augmented Transformer for Speech Recognition.* arXiv:2005.08100, 2020. https://arxiv.org/abs/2005.08100

[9] Z. Yao, L. Guo, X. Yang, et al. *Zipformer: A Faster and Better Encoder for Automatic Speech Recognition.* arXiv:2310.11230 (ICLR 2024). https://arxiv.org/abs/2310.11230

[10] A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. *Connectionist Temporal Classification.* ICML, 2006.

[11] A. Graves. *Sequence Transduction with Recurrent Neural Networks* (RNN-T). arXiv:1211.3711, 2012. https://arxiv.org/abs/1211.3711

[12] S. Watanabe, T. Hori, S. Kim, J. R. Hershey, T. Hayashi. *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition.* IEEE JSTSP, 2017.

[13] V. Panayotov, G. Chen, D. Povey, S. Khudanpur. *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books.* ICASSP, 2015.

[14] D. S. Park, W. Chan, Y. Zhang, et al. *SpecAugment: A Simple Data Augmentation Method for ASR.* arXiv:1904.08779 (Interspeech 2019). https://arxiv.org/abs/1904.08779

[15] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, M. Sonderegger. *Montreal Forced Aligner.* Interspeech, 2017.

[16] A. Baevski, H. Zhou, A. Mohamed, M. Auli. *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* arXiv:2006.11477, 2020. https://arxiv.org/abs/2006.11477

[17] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction.* arXiv:2106.07447, 2021. https://arxiv.org/abs/2106.07447

[18] Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons* (straight-through estimator). arXiv:1308.3432, 2013. https://arxiv.org/abs/1308.3432

[19] A. Pagnoni, R. Pasunuru, et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* arXiv:2412.09871, 2024. https://arxiv.org/abs/2412.09871

[20] K. Slagle. *SpaceByte: Towards Deleting Tokenization from Large Language Modeling.* arXiv:2404.14408, 2024. https://arxiv.org/abs/2404.14408

[21] J. Wang, T. Gangavarapu, J. N. Yan, A. M. Rush. *MambaByte: Token-free Selective State Space Model.* arXiv:2401.13660, 2024. https://arxiv.org/abs/2401.13660

[22] Context-Aware Dynamic Chunking for Streaming Tibetan Speech Recognition. arXiv:2511.09085, 2025. https://arxiv.org/abs/2511.09085 *(naming-collision anchor: streaming variable-window, not learned chunking).*

[23] TC-BiMamba: Trans-Chunk BiMamba for Unified Streaming and Non-Streaming ASR. arXiv:2602.11546, 2026. https://arxiv.org/abs/2602.11546 *(naming-collision anchor).*

[24] W. Han, Z. Zhang, Y. Zhang, et al. *ContextNet: Improving Convolutional Neural Networks for ASR with Global Context.* arXiv:2005.03191 (Interspeech 2020). https://arxiv.org/abs/2005.03191

[25] S. Kim, A. Gholami, A. Shaw, et al. *Squeezeformer: An Efficient Transformer for Automatic Speech Recognition.* arXiv:2206.00888 (NeurIPS 2022). https://arxiv.org/abs/2206.00888

[26] Y. Peng, S. Dalmia, I. Lane, S. Watanabe. *Branchformer: Parallel MLP-Attention Architectures to Capture Local and Global Context for Speech Recognition and Understanding.* arXiv:2207.02971 (ICML 2022). https://arxiv.org/abs/2207.02971

[27] K. Kim, F. Wu, Y. Peng, et al. *E-Branchformer: Branchformer with Enhanced Merging for Speech Recognition.* arXiv:2210.00077 (SLT 2022). https://arxiv.org/abs/2210.00077

[28] S. Kriman, S. Beliaev, B. Ginsburg, et al. *QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions.* arXiv:1910.10261 (ICASSP 2020). https://arxiv.org/abs/1910.10261

[29] J. Li, V. Lavrukhin, B. Ginsburg, et al. *Jasper: An End-to-End Convolutional Neural Acoustic Model.* arXiv:1904.03288 (Interspeech 2019). https://arxiv.org/abs/1904.03288

[30] S. Karita, N. Chen, T. Hayashi, et al. *A Comparative Study on Transformer vs RNN in Speech Applications.* arXiv:1909.06317 (ASRU 2019). https://arxiv.org/abs/1909.06317

[31] Conformer paper Table 2 comparison set (Transformer-AED [Synnaeve et al.]; LSTM-AED [Park et al.]), as tabulated in Gulati et al. 2020 [8]. Original figures: WER numbers cited via [8].

[32] S. Chen, C. Wang, Z. Chen, et al. *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* arXiv:2110.13900 (2021). https://arxiv.org/abs/2110.13900

[33] A. Radford, J. W. Kim, T. Xu, et al. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). arXiv:2212.04356 (2022). https://arxiv.org/abs/2212.04356
