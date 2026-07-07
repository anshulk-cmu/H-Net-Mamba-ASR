# Interpretable Hierarchical Speech Recognition via Dynamic Chunking: A Mamba–H-Net ASR Proposal

**Working title (candidate):** *DC-ASR: Learned Hierarchy at No Cost to Accuracy — A Mamba–H-Net Speech Recognizer that Chunks Itself*

**Author:** (you) · **Drafted:** 2026-07-01 · **Status:** research proposal / idea-shaping document

---

## 0. One-paragraph pitch

Modern ASR still bakes in a fixed acoustic-to-symbol granularity: a frontend downsamples audio to a fixed frame rate (e.g. one vector / 40 ms), and a fixed subword/BPE or character vocabulary defines the output units. We propose to remove *both* fixed granularities from the **encoder** by inserting **H-Net-style dynamic chunking** (learned, content-based boundary prediction, [Hwang, Wang & Gu 2025](https://doi.org/10.48550/arXiv.2507.07955)) between stacks of **Mamba-2** state-space blocks ([Samba-ASR](https://doi.org/10.48550/arXiv.2501.02832); [Mamba streaming ASR](https://doi.org/10.48550/arXiv.2410.00070) establish Mamba as a strong ASR backbone). The result is a recognizer whose encoder *learns where the boundaries are*, rather than being told. We build exactly **two architecture types**, each in a **Large and a Small** variant: a **1-stage** model `Mamba → H-Net → Mamba` (one chunking block) and a **2-stage** model `Mamba → H-Net → Mamba → H-Net → Mamba` (two chunking blocks). We sweep a **compression level $N\in\{1,2,3,4\}$** defined as the *overall* downsampling factor — the encoder keeps a fraction $1/N$ of its frames end-to-end ($N{=}1$ keeps 100%, a no-compression control; then 50/33/25%). The 1-stage model reaches this with its single block at factor $N$; the **2-stage model splits it across two blocks at factor $\sqrt{N}$ each** ({$1,\sqrt2,\sqrt3,2$}), so both types keep the **same overall fraction** at every $N$ — an iso-compression comparison of one hard step vs. two soft steps. The proposal carries **two payloads**. (1) A **competitiveness claim**: DC-ASR should reach WER on par with the strongest fixed-architecture encoders (Conformer, Zipformer), demonstrating that letting the model *self-learn* its acoustic hierarchy costs little-to-no accuracy while removing hand-set frame rates and tokenizers — i.e. it is a *better default*, not merely an equal one. (2) An **interpretability payload**: we treat the learned boundaries as objects of study and ask what linguistic structure the compression discovers on its own — do stage-1 chunks align with phones/syllables and stage-2 chunks with words? — and how much of ASR's supervised signal is really needed for that structure to emerge. The framing sharpens against **Zipformer**, which is *already* a hierarchical multi-frame-rate encoder but with a **hand-designed** downsampling schedule; DC-ASR asks whether a **learned, content-adaptive** hierarchy can match it.

---

## 1. Where this came from (the raw idea, cleaned up)

Your original framing:

> build a new ASR as a combination of H-Net and Mamba-ASR — a 1-stage and a 2-stage H-Net-ASR; a few blocks of Mamba, then H-Net, then a few blocks of Mamba; see how compression helps in self-learning at $N=1,2,3,4$ levels; then analyse these learned things. Search arXiv for overlap / uniqueness / novelty.

Cleaned into research terms, that is three separable contributions:

1. **An architecture** — a Mamba↔dynamic-chunk *sandwich* encoder for ASR, in **two types**, each with a **Large and a Small** variant: **1-stage** `Mamba–H-Net–Mamba` (one chunking block) and **2-stage** `Mamba–H-Net–Mamba–H-Net–Mamba` (two chunking blocks). H-Net calls these "stages."
2. **A competitiveness + compression study** — show WER competitive with Conformer/Zipformer, and characterise how recognition quality and the learned segmentation change with the **overall compression level $N\in\{1,2,3,4\}$** (100/50/33/25% frames kept end-to-end) and with the decoder/LM choice. Because both types are held at the *same overall compression*, differences isolate the effect of *staging*. The thesis: *self-learned chunking does not materially hurt WER*, so it is a better default than hand-set frame rates + tokenizers.
3. **An interpretability study** — decode what the learned chunk boundaries and chunk representations *mean* linguistically and acoustically at each stage.

The novelty search (Section 6) says contributions (1) and (3) are essentially open in speech; (2) is what turns it from a demo into a paper — and the SOTA-WER bar is what makes the "better default" argument land.

---

## 2. Background you're standing on

### 2.1 Mamba / selective state-space models as an ASR backbone
A Mamba-2 layer is a selective SSM: for input $x_t$ it maintains a hidden state $h_t$ evolving as a discretized linear system $h_t=\bar{A}_t h_{t-1}+\bar{B}_t x_t,\; y_t=C_t h_t$, where $(\bar A_t,\bar B_t,C_t,\Delta_t)$ are **input-dependent** (the "selective" part). It gives $O(L)$ sequence-length scaling with a global receptive field, which is exactly what long acoustic sequences want. Two of your four read papers establish that this works for ASR:
- **Samba-ASR** ([Shakhadri et al. 2025](https://doi.org/10.48550/arXiv.2501.02832)) uses Mamba as *both* encoder and decoder and reports strong WERs vs. Transformer baselines.
- **Mamba streaming ASR + UMA** ([Fang & Li 2024](https://doi.org/10.48550/arXiv.2410.00070)) uses a Mamba encoder plus **unimodal aggregation (UMA)** — a *single-level*, CTC-triggered frame aggregation. UMA is the closest existing "learned aggregation" in ASR, and its single-level, CTC-driven nature is precisely the contrast that motivates a *multi-level, self-learned* mechanism.

### 2.2 H-Net dynamic chunking
H-Net ([Hwang, Wang & Gu 2025](https://doi.org/10.48550/arXiv.2507.07955)) makes discrete segmentation differentiable and end-to-end. Its core loop, per stage:
- a **routing module** predicts a boundary probability from the cosine dissimilarity of adjacent hidden states,
$$p_t=\tfrac12\Big(1-\tfrac{q_t^\top k_{t-1}}{\lVert q_t\rVert\,\lVert k_{t-1}\rVert}\Big)\in[0,1];$$
- a **downsampler** keeps positions where $p_t>0.5$, compressing $L$ vectors to $L'<L$ chunk vectors;
- a **smoothing module** (an EMA, $\bar z_t=P_t\hat z_t+(1-P_t)\bar z_{t-1}$) plus a **straight-through estimator** make the hard selection trainable;
- a **ratio loss** $\mathcal L_{\text{ratio}}$ targets a desired compression rate so the model neither keeps everything nor collapses;
- an **upsampler/dechunker** restores the original length for the residual/output path.

Stack this and you get a U-Net-like **encoder → main network → decoder** hierarchy where the *main* network runs on the compressed sequence. H-Net uses **Mamba-2 as its encoder/decoder backbone** already — which is exactly why the Mamba↔H-Net marriage is natural rather than forced.

### 2.3 The speech-segmentation / interpretability literature you'll connect to
There is a deep line on *unsupervised* acoustic unit and word discovery — segmental CPC for word segmentation, phoneme-boundary detection with learnable segmental features, "what do self-supervised speech models know about words," and human-like phonetic-categorization biases in neural speech models. These matter because they give you **ground-truth probes and evaluation protocols** (boundary F1, phone-purity, cluster NMI) to interpret H-Net's learned boundaries against — but none of them use a *fully-differentiable, jointly-trained, multi-level* chunker driven by the ASR objective. That is the seam you're entering.

---

## 3. Proposed architecture

### 3.1 The Mamba–chunk sandwich (one stage)
```
        waveform / filterbank (e.g. 80-dim logmel @ 100 Hz)
                     │
              [ Conv subsampling ]  →  frame rate f0  (e.g. 25–50 Hz)
                     │
   ┌─────────────────▼──────────────────┐
   │  ENCODER stage s (operates at len L_s)              │
   │    k_enc × Mamba-2 blocks   ("few blocks of Mamba")  │
   │            │                                         │
   │       [ Router ]  → boundary prob p_t (cosine)        │
   │       [ Downsample ]  → keep p_t>0.5   (len L_{s+1})  │
   └─────────────────┬──────────────────┘
                     │  (compressed chunk vectors)
             … recurse for N stages …
                     │
         ┌───────────▼───────────┐
         │  MAIN network          │   m × Mamba-2 blocks at the
         │  (coarsest level L_N)  │   most compressed rate
         └───────────┬───────────┘
                     │
   ┌─────────────────▼──────────────────┐
   │  DECODER stage s (mirror)                           │
   │       [ Upsample / dechunk ] → len L_s               │
   │       [ EMA smoothing + STE residual ]               │
   │    k_dec × Mamba-2 blocks   ("few blocks of Mamba")  │
   └─────────────────┬──────────────────┘
                     │  frame-rate encoder output
             ┌───────▼────────┐
             │  ASR head      │   CTC  and/or  attention/transducer
             └────────────────┘
```
The phrase "a few blocks of Mamba, then H-Net, then a few blocks of Mamba" is exactly the encoder-stage box: Mamba blocks build local acoustic context → router+downsample compresses → (recurse) → main Mamba stack → upsample → Mamba blocks. This is H-Net's own encoder/main/decoder skeleton with Mamba-2 backbones, retargeted from bytes to acoustic frames.

### 3.2 The two architecture types, and the compression level $N$
We commit to exactly **two encoder types**, each built in a **Large** and a **Small** variant (four base encoders total):

- **Type A — 1-stage** `Mamba → H-Net → Mamba`: one H-Net chunking block between two Mamba stacks. One learned compression. Hypothesis: boundaries land near **phone / syllable** granularity. This is the acoustic analogue of UMA ([Fang & Li 2024](https://doi.org/10.48550/arXiv.2410.00070)) but *self-learned and not CTC-triggered*.
- **Type B — 2-stage** `Mamba → H-Net → Mamba → H-Net → Mamba`: two chunking blocks. The first compresses frames → units, the second compresses units → word-ish chunks, with the innermost Mamba stack running at the coarsest rate. This is H-Net's headline configuration ported to audio. Hypothesis: **stage-1 ≈ phone/syllable, stage-2 ≈ word**.

**Compression level $N$ = overall downsampling factor.** $N$ is the *end-to-end* target: the encoder keeps a fraction $1/N$ of its input frames overall (enforced by the ratio loss, Section 2.2, which sets the target; the router is free to place boundaries *within* that budget). Crucially, the two types reach the same $N$ differently, so they are compared at **matched overall compression**:

- **Type A** has one block, run at per-block factor $N$.
- **Type B** has two blocks, each run at per-block factor $\sqrt{N}$ (so the two keep-fractions multiply to $1/N$): the per-block factors are $\{1,\sqrt2,\sqrt3,2\}$.

| $N$ (overall) | overall frames kept | Type A per-block | Type B per-block ($\sqrt N$) | Type B kept / block |
|---|---|---|---|---|
| $N=1$ | 100% | 1 (no-op) | 1 | 100% |
| $N=2$ | 50% | 2 | $\sqrt2\approx1.41$ | 70.7% |
| $N=3$ | 33% | 3 | $\sqrt3\approx1.73$ | 57.7% |
| $N=4$ | 25% | 4 | 2 | 50% |

Two consequences worth stating up front:
- **$N=1$ is a no-compression control living inside the architecture** — every H-Net block passes everything through, so Type A and Type B both reduce to a pure-Mamba encoder. A clean "chunking off" reference at zero extra engineering.
- **Iso-compression is the whole point of the $\sqrt N$ split.** At each $N$, Type A applies one hard $N\times$ compression while Type B applies two gentle $\sqrt N\times$ compressions to the *same total budget*. So a WER or interpretability gap between them isolates the effect of **staging the compression** (coarse-to-fine, two learned boundary sets) rather than **how much** is discarded. Type B's gentler per-block ratios should also make its stages *less* prone to collapse than a naive per-block-$N$ scheme would.

So the study is a **2 types × 4 compression levels × 2 sizes** grid, crossed with the decoder/LM matrix below. **Whether Type B's two stages stay clean and non-collapsing (and whether staging helps or hurts at fixed compression) is an empirical result, not a risk to hide.**

### 3.3 Decoders and language models (the evaluation matrix)
Every architecture variety is evaluated under **both decoders, each with and without an external LM** — a 2×2 output matrix, because the "no-WER-cost" claim must hold across the standard decoding regimes, not just one:

| Decoder | without LM | with LM |
|---|---|---|
| **CTC** | greedy / prefix-beam | + n-gram or neural-LM shallow fusion / rescoring |
| **Attention (AED)** | beam search | + LM shallow fusion / rescoring |

- **CTC head** on the restored frame sequence is the simplest and gives a natural interpretability cross-check: CTC spike positions vs. learned chunk boundaries (UMA already relates aggregation to CTC).
- **Attention decoder** (autoregressive; Samba-ASR-style Mamba or a small Transformer decoder) typically yields the stronger WER and is where you most expect to match Conformer/Zipformer.
- **LM ablation** matters for the headline: Conformer improves ~2.1→1.9 / 4.3→3.9 test-clean/other with an external LM, so DC-ASR's with-LM and without-LM numbers must *both* be reported against the corresponding competitor column. A joint **CTC/attention** setup (hybrid loss + one-pass rescoring, the ESPnet/WeNet convention) is the natural way to get both heads from one model and is worth including as a third row.

---

## 4. The interpretability program (the scientific core)

For each stage $s$ (one in Type A, two in Type B) and each compression level $N$, the learned boundaries $\{t: p_t>0.5\}$ and chunk embeddings are the **data** of the study. Concrete analyses:

1. **Boundary alignment.** Force-align the audio (e.g. Montreal Forced Aligner) to get phone/word boundaries; measure boundary **precision/recall/F1 and R-value** of learned level-$s$ boundaries against phone, syllable, morpheme, and word references. *Question: which level "snaps to" which linguistic tier, and how sharply?*
2. **Chunk-identity probing.** Train linear probes on chunk embeddings to predict phone identity, phone class (voicing/place/manner), word identity, part-of-speech. *Question: what information concentrates at each level?*
3. **Compression vs. linguistics trade-off, and staging at fixed compression.** Sweep $N$; plot compression rate vs. WER vs. boundary-F1. Because Type A and Type B are held at the *same overall $N$*, directly compare their boundaries: *does splitting one $N\times$ step into two $\sqrt N\times$ steps (Type B) produce cleaner phone-then-word tiers than a single hard step (Type A)?* *Question: is there a compression sweet spot where the units become maximally linguistic, and does staging move it?*
4. **Emergence / self-learning curve.** Track boundary-F1 and probe accuracy over training. *Question: does linguistic structure emerge before, with, or after WER convergence? Is it "self-learned" (present with weak/no explicit boundary supervision)?*
5. **Cross-lingual / morphology transfer.** Because H-Net++ ([Zakershahrak & Ghodratnama 2025](https://doi.org/10.48550/arXiv.2508.05628)) shows morphology-rich languages benefit from learned chunking, test whether learned acoustic chunks track **morphological** structure in an agglutinative language (Turkish/Finnish) vs. an isolating one (Mandarin).
6. **Robustness signature.** H-Net's text version is more robust to perturbation; test whether dynamic acoustic chunking is more robust to noise / speaking-rate / disfluency than fixed-rate baselines, and whether boundaries *move* sensibly under time-warp.

This is the "analyse these learned things" step, made falsifiable.

---

## 5. Experimental design

**Data.** LibriSpeech (100h→960h) for the core; add a morphology-contrast pair (e.g. Turkish from Common Voice vs. Mandarin AISHELL) for analysis 5. (You already worked with LibriSpeech + AISHELL, so infrastructure exists.)

**SOTA reference targets (LibriSpeech test-clean / test-other, verified 2026-07-01).** These are the bar the "no-WER-cost" claim is measured against:

| System | Params | test-clean | test-other | Note |
|---|---|---|---|---|
| Conformer-L, no LM | ~118M | 2.1 | 4.3 | orig. paper |
| Conformer-L, + LM | ~118M | 1.9 | 3.9 | orig. paper |
| Conformer-S, no LM | 10M | 2.7 | 6.3 | small-model ref |
| Zipformer-S (transducer) | 23.2M | 2.42 | 5.73 | icefall |
| Zipformer-M (transducer) | 65.5M | 2.21 | 4.79 | 63 GFLOPs |
| Zipformer-L (transducer) | 148.4M | 2.00 | 4.38 | strongest |

The honest competitive bar is **Zipformer**, and it is the *right* bar for a second reason: Zipformer is itself a **hand-designed hierarchical multi-frame-rate encoder** (a U-Net downsampling schedule). Matching it means a *learned* hierarchy is as good as a *hand-tuned* one — the paper's whole point.

**Baselines you must run yourself (fair, compute-matched — reviewers will check).**
- **Conformer/CTC and a Zipformer-style baseline** re-trained in your own pipeline at matched params/FLOPs (don't only cite published numbers; recipe/data differences move WER by 0.2–0.5 abs).
- **Mamba encoder without chunking** (Samba-ASR-style) — isolates the chunking contribution.
- **Mamba encoder + UMA** ([Fang & Li 2024](https://doi.org/10.48550/arXiv.2410.00070)) — the single-level learned-aggregation baseline; **matching/interpreting-beyond UMA is a crux comparison.**
- **Fixed 2×/4× pooling** at the same average compression as learned chunking, and a **fixed Zipformer-schedule** run — the two "hand-designed hierarchy" controls that isolate *learned* vs *fixed* boundaries.

**Model scale.** Each type is built **Large** (≈65–100M enc params, next to Zipformer-M/L) and **Small** (≈25M, next to Zipformer-S) so the WER comparison is params-matched. Report GFLOPs like H-Net/Zipformer do.

**The full grid.** {Type A 1-stage, Type B 2-stage} × {compression $N$=1,2,3,4} × {Large, Small} × {CTC, attention} × {no-LM, +LM}. It factorises cleanly: $N$, decoder and LM are cheap axes once an encoder trains, and $N{=}1$ is the built-in no-chunk control. Use the **Small** variant to screen the full $N$-and-decoder grid; promote only the winning cells to the **Large** variant for the headline WER table.

**Metrics.** WER (test-clean/other, ± LM); boundary P/R/F1 + R-value; probe accuracy; realised compression rate vs. target $N$ (does the model use its full $1/N$ budget?); RTF/latency; and a training-dynamics panel. Define WER $= (S+D+I)/N$ explicitly.

**Ablations.** compression-level $N$ sweep (100/50/33/25% kept); STE vs. Gumbel-softmax routing; CTC vs. attention vs. joint CTC/attention; router cosine vs. learned-MLP boundary predictor; learned vs. fixed (Zipformer-schedule) downsampling.

**Compute.** This is GPU work — your Babel access (interactive `srun --gres=gpu:1`, the `hnetasr`-style conda env) is the natural home; a ~65–100M model on 960h LibriSpeech is a few-GPU-day regime per config, so the {2 types × 4 compression levels × 2 sizes} grid (decoder/LM axes are cheap add-ons on a trained encoder) is the budget driver. *(Note: the earlier `h-mamba_asr` repo/data and its conda env were deleted from Babel and the GitHub remote is gone, so this would start from a fresh scaffold.)*

---

## 6. Novelty, overlap, and prior art (grounded in a citation-graph search)

I searched arXiv (full-text API, multiple query families), Semantic Scholar (H-Net's full forward-citation set: **64 citing papers**), and OpenAlex. Findings:

### 6.1 The exact combination does not exist
- `"H-Net" AND speech` on arXiv → **0 hits**. `"Mamba" AND "H-Net"` → **1 hit**, and it is [ByteGen](https://doi.org/10.48550/arXiv.2508.02247) (financial orderbook events), not speech.
- Of the **64 papers citing H-Net**, **8 touch audio** — and *all eight are generation-side*: neural audio **codecs / TTS / variable-frame-rate tokenizers** (DyCAST "Beyond Fixed Frames", Elastic Time, TLDR audio-token compression) or **omnimodal token compression** (DASH). **None is an ASR encoder that learns a multi-level acoustic hierarchy**, and none makes the learned units the object of an interpretability study.
- H-Net's dynamic chunking *has* been ported to other modalities already: **genomics** (dnaHNet, LDARNet, DNACHUNKER) and **finance** (ByteGen). Speech is the conspicuous gap — which both confirms the idea's reach *and* means you should move before someone else fills it.

### 6.2 Naming-collision warning (must be addressed head-on in the paper)
"**Dynamic chunk(ing)**" already has an **established, different** meaning in streaming ASR — the WeNet-style *variable streaming chunk size for latency control*. Recent examples: [Context-Aware Dynamic Chunking for Tibetan ASR](https://doi.org/10.48550/arXiv.2511.09085) and [TC-BiMamba](https://doi.org/10.48550/arXiv.2602.11546) (dynamic chunk-size training for unified streaming BiMamba). These are **not** H-Net's learned *content-based* chunking; they vary a fixed window's size. Your paper must disambiguate in the first page (propose a distinct term, e.g. *learned acoustic chunking* or *content-based hierarchical chunking*) or a reviewer will wave these as "prior art" incorrectly — and you'll want to cite them precisely to draw the line.

### 6.3 Closest legitimate neighbors and how you differ
| Prior work | What it does | How your proposal differs |
|---|---|---|
| **H-Net** ([2507.07955](https://doi.org/10.48550/arXiv.2507.07955)) | Learned dynamic chunking for **text/DNA/code**; Mamba-2 backbone | First application to **speech/ASR**; acoustic-frame boundaries; interpretability of acoustic units |
| **H-Net++** ([2508.05628](https://doi.org/10.48550/arXiv.2508.05628)) | H-Net for morphologically-rich **text** (Persian) | Acoustic, not text; but directly motivates your morphology-transfer analysis |
| **Mamba+UMA ASR** ([2410.00070](https://doi.org/10.48550/arXiv.2410.00070)) | **Single-level**, CTC-triggered frame aggregation | **Multi-level**, self-learned, differentiable hierarchy; UMA is your key baseline |
| **Samba-ASR** ([2501.02832](https://doi.org/10.48550/arXiv.2501.02832)) | Mamba encoder+decoder ASR, **no hierarchy** | Adds learned hierarchical compression + its analysis |
| **Zipformer** ([2310.11230](https://doi.org/10.48550/arXiv.2310.11230)) | SOTA ASR; **hand-designed** U-Net multi-frame-rate encoder | Same hierarchical *idea*, but **learned & content-adaptive** rather than a fixed schedule — the key WER foil |
| **Conformer** ([2005.08100](https://doi.org/10.48550/arXiv.2005.08100)) | SOTA conv-augmented Transformer; **flat** frame rate | The flat-encoder WER anchor (2.1/4.3 no-LM) |
| DyCAST / Elastic Time / TLDR / DASH | Variable-rate audio **codec/TTS/token compression** | Recognition-side; interpretability, not generation efficiency |
| Tibetan-DC / TC-BiMamba | **Streaming** variable *window* size | Content-based *learned boundaries*, a different mechanism (see 6.2) |

### 6.4 Net assessment
- **Architecture novelty: high** — no H-Net×Mamba ASR exists; the specific Mamba-sandwich-per-stage instantiation for audio is new.
- **Interpretability novelty: high** — "what linguistic hierarchy does an ASR model self-discover when you let it chunk freely at $N$ levels" is unasked in this form; it links H-Net to the unsupervised-unit-discovery literature in a way nobody has.
- **Competitiveness angle: defensible and sharp** — because Zipformer already proved a *hand-designed* hierarchical encoder beats a flat one at lower FLOPs, "a *learned* hierarchy matches the hand-designed one" is a crisp, falsifiable claim with an obvious baseline, not a vague SOTA-chase.
- **Risk of being scooped: real** — genomics/finance ports appeared within a year; audio codecs already use the cosine-boundary trick. The ASR + interpretability framing is your moat; publish the *analysis and the learned-vs-fixed comparison*, not just a WER number.

---

## 7. Contributions, as they'd appear in the paper

1. **DC-ASR**, the first ASR encoder built from Mamba-2 blocks interleaved with H-Net dynamic chunking, in **two types** — 1-stage `Mamba–H-Net–Mamba` and 2-stage `Mamba–H-Net–Mamba–H-Net–Mamba` — each in **Large and Small** variants and evaluated with **CTC and attention decoders, with and without an LM**.
2. **A "no-WER-cost" result**: DC-ASR reaches WER competitive with Conformer and Zipformer on LibriSpeech, showing that a **learned, content-adaptive** hierarchy matches a **hand-designed** one (Zipformer's fixed multi-rate schedule) — so self-learned chunking is a better *default*, not an accuracy sacrifice. Includes the **compression-level $N$=1,2,3,4** (100/50/33/25% frames kept) trade-off curve, with $N{=}1$ as the built-in no-chunk control.
3. An **interpretability suite** (boundary alignment, chunk probing, emergence curves) showing what linguistic tiers the 1- and 2-stage models discover and whether the structure is *self-learned* from the ASR objective.
4. A **cross-linguistic** analysis (morphology-rich vs. isolating) tying learned acoustic chunking to H-Net++'s morphology findings.

## 8. Risks and honest unknowns
- **Stage collapse at high compression:** in Type B at high $N$ (e.g. $N{=}4$: two $2\times$ blocks, 25% kept overall), a stage may still degenerate (all-keep or all-drop). Mitigation: per-stage ratio-loss targets, curriculum on sequence length (H-Net++ trick); if it persists, that ceiling on usable compression is itself a reportable finding.
- **Not actually matching Zipformer:** the WER gap may not close at small scale. Mitigation: params-match to Zipformer-S/M, re-train baselines in-pipeline, and lead with the *learned-vs-fixed-hierarchy* comparison (which holds even at a modest absolute WER) rather than a raw SOTA claim.
- **Boundaries may be acoustic, not linguistic:** learned chunks might track energy/silence, not phones. That is itself a *publishable finding* if measured honestly against forced alignment.
- **Streaming vs. global chunking tension:** H-Net's router is bidirectional-ish; a causal/streaming router is its own mini-contribution (and connects to the streaming-ASR papers).
- **Compute:** the $N$×size grid is the budget driver; start at $N\le2$, LibriSpeech-100h to de-risk before scaling.

## 9. Minimal first experiment (2–3 weeks)
**Type A (1-stage), Small, $N{=}2$ (50% kept), CTC head, LibriSpeech-100h**, vs. three controls at matched params: (a) the same model at $N{=}1$ (no-chunk, i.e. pure Mamba — free control), (b) fixed-2×-pool, (c) a small Conformer/CTC. Deliverable: a WER table (no-LM) + one boundary-alignment-F1 plot vs. forced-aligned phones. **Go/no-go:** if learned boundaries beat fixed pooling on WER *and* show above-chance phone alignment, scale to 960h and add Type B, the Large variant, the attention decoder, the LM, and the full $N$-sweep.

---

## 10. Key references (verified, non-retracted)
- Hwang, Wang, Gu (2025), *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net). https://doi.org/10.48550/arXiv.2507.07955
- Zakershahrak & Ghodratnama (2025), *H-Net++: Hierarchical Dynamic Chunking … Morphologically-Rich Languages*. https://doi.org/10.48550/arXiv.2508.05628
- Fang & Li (2024), *Mamba for Streaming ASR Combined with Unimodal Aggregation*. https://doi.org/10.48550/arXiv.2410.00070
- Shakhadri et al. (2025), *Samba-ASR: SOTA Speech Recognition Leveraging Structured State-Space Models*. https://doi.org/10.48550/arXiv.2501.02832
- Gulati et al. (2020), *Conformer: Convolution-augmented Transformer for Speech Recognition* (flat-encoder WER anchor). https://doi.org/10.48550/arXiv.2005.08100
- Yao et al. (2024), *Zipformer: A faster and better encoder for ASR* (hand-designed hierarchical-encoder foil). https://doi.org/10.48550/arXiv.2310.11230
- Zhang et al. (2026), *TC-BiMamba: Trans-Chunk BiMamba for unified streaming/non-streaming ASR* (naming-collision anchor). https://doi.org/10.48550/arXiv.2602.11546
- *Context-Aware Dynamic Chunking for Streaming Tibetan Speech Recognition* (2025) (naming-collision anchor). https://doi.org/10.48550/arXiv.2511.09085

*Novelty search performed 2026-07-01 over arXiv full-text, Semantic Scholar (64 H-Net citations enumerated), and OpenAlex. "Dynamic chunking" in ASR prior art overwhelmingly denotes streaming variable-window sizing, a distinct mechanism from H-Net's learned content-based boundaries; the recognition-side, multi-level, interpretability framing proposed here is unoccupied at time of search.*
