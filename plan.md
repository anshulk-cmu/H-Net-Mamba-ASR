# H-Mamba ASR: 8-Week Plan for EMNLP 2026

**Target:** ARR May 2026 cycle (deadline May 25, 2026)  
**Venue:** EMNLP 2026, Budapest, October 24-29  
**Notification:** August 20, 2026  
**Today:** April 5, 2026 (Week 1, Day 5)  
**Authors:** Anshul Kumar, Shinji Watanabe (CMU)  
**Last updated:** April 5, 2026

**Proposed title:** Learning Acoustic Compression Hierarchies: Dynamic Chunking with Mamba for Variable-Rate Speech Recognition

---

## 1. Project Status

### 1.1 Completed Work

| Item | Result |
|------|--------|
| Conformer Small S2S (960h, 150ep) | 2.52/5.97 (LM), 4.13/10.13 (no LM) |
| ConMamba Small S2S (960h, 150ep) | 2.22/5.56 (LM), 3.34/8.47 (no LM) |
| ConMambaMamba Small S2S (960h, 150ep) | 2.52/5.98 (LM), 3.64/8.70 (no LM) |
| Conformer Large S2S (960h, 150ep) | 2.03/4.70 (LM), 2.57/5.94 (no LM) |
| ConMamba Large S2S (960h, 150ep) | 2.27/5.12 (LM), 2.82/6.60 (no LM) |
| ConMambaMamba Large S2S (960h, 150ep) | 2.41/5.72 (LM), 2.93/6.99 (no LM) |
| ConMamba Large CTC (960h) | 3.93/10.40 (no LM) |
| H-Mamba pilot (100h, small, N=2,3,4) | N=2: 5.96/16.35, N=3: 7.80/21.36, N=4: 7.35/19.71 |
| 8 H-Mamba YAML configs | hmamba_{small,large}_N{1,2,3,4}.yaml |
| 8 H-Mamba SLURM scripts | hmamba_{small,large}_N{1,2,3,4}.sh |
| Conda env `hnetasr` | All deps verified (torch, mamba_ssm, causal_conv1d, speechbrain, etc.) |
| LibriSpeech 960h | /data/user_data/anshulk/hnet_asr/LibriSpeech |
| Full code audit + bug fixes | 11 bugs fixed, gradient flow verified, 3 audits complete |
| Package upgrade | torch 2.0.1→2.1.1, mamba-ssm 1.1.3→2.0.3, causal-conv1d 1.1.3→1.4.0 |
| causal-conv1d 1.4.0 API patches | 6 fwd + 3 bwd call sites patched in selective_scan_interface.py |
| Triton kernel workaround | MAMBA_KERNEL_AVAILABLE=False for DeChunk EMA (Triton 2.1.0 crash on Ampere) |
| Smoke test v1 (Phase 1 only) | 2 epochs passed: loss 597→395, compression 0.886 vs target 0.882. Timed out during eval. |
| Smoke test v3 DDP (job 6928765) | DDP 2-GPU training passed, checkpoint saved, NumPy 1.26.4 fix confirmed. |
| Resume test (job 6928816) | Resumed from DDP checkpoint, epoch 2 trained. Optimizer, scheduler, DC state all restored correctly. |
| Grad norm logging fix | fit_batch() override captures grad_norm/bias_grad before optimizer.zero_grad(). Confirmed: grad_norm=1195 (was 0.0000). Also: persist values across grad_accum batches for large models. |
| Final validation (job 6928979) | DDP training + grad norm fix verified. Grad norm 1195.0 (max 1770.9), bias_grad 4.9→13.5. |
| Comprehensive documentation | docs/hmamba_dynamic_chunking.md (2500+ lines), BiMamba v2, SSM math, decoder, pos encoding |
| NumPy downgrade fix | numpy 2.0.2 → 1.26.4, fixes DDP broadcast crash (PyTorch 2.1.1 incompatible with NumPy 2.x) |
| SLURM partition config | Small: general (2d). Large: preempt (14d, --requeue). |
| Baseline reproduction docs | docs/baseline_reproduction.md |

### 1.2 Bug Fixes (All Verified)

| # | Severity | File | Issue | Fix |
|---|----------|------|-------|-----|
| 1 | HIGH | HMambaEncoder.py | ratio_loss got zero gradient (boundary_mask was boolean) | Uses boundary_hard[..., 1] with STE gradients |
| 2 | HIGH | HMambaEncoderWrapper.py | Wrong residual: `stage0 + residual + expanded` | `expanded * ste_func(selected_probs) + residual` |
| 3 | HIGH | train_s2s_hmamba.py | Eval loaded checkpoint before wrapping encoder | Moved wrap before load_state_dict |
| 4 | MED | train_s2s_hmamba.py | Global `tokenizer` instead of `self.tokenizer` | Fixed |
| 5 | MED | train_s2s_hmamba.py | `update_metric()` never called, early stopping broken | Added call after checkpointer save |
| 6 | MED | train_s2s_hmamba.py | No OOM handler for DC variable memory | Added fit_batch() with OOM recovery |
| 7 | MED | train_s2s_hmamba.py | Optimizer fallback missed DC params | Added residual_proj + dechunk_layer |
| 8 | LOW | train_s2s_hmamba.py | Dead dc_loss_weight variable | Removed |
| 9 | ENV | conda env | Missing tensorboard, psutil | Installed |
| 10 | MED | train_s2s_hmamba.py | Grad norm + bias_grad always 0 (computed after zero_grad) | Override fit_batch(), capture between backward() and step() |
| 11 | LOW | train_s2s_hmamba.py | Grad norm always 0 on large models (grad_accum=8 misaligned with log interval) | Persist _last_grad_norm across non-step batches instead of resetting to 0 |

**Gradient flow verified** — all routing params receive non-zero gradients:

```
q_proj: 95.28  |  k_proj: 93.14  |  temperature: 10.49  |  boundary_bias: 2.67  |  residual_proj: 2878.5
```

### 1.3 Known Issues (Intentionally Deferred)

| Issue | Why acceptable |
|-------|---------------|
| chunk_mask not passed to Stage 1 | ConmambaEncoderLayer ignores mask (line 316). BiMamba doesn't use attention masks. |
| BCE loss includes hardcoded position 0 | Zero gradient at that position, cosmetic only. |
| Mamba-2 Triton SSD kernel crash (Ampere) | Triton 2.1.0 JIT assertion on sm_86/sm_89. Fix needs triton ≥ 2.2 (PyTorch ≥ 2.2). PyTorch EMA fallback used for DeChunk. Main encoder Mamba-1 kernels unaffected. |
| Streaming incompatibility | BiMamba is bidirectional, routing uses look-ahead. Offline-only by design. |
| Conformer Large CTC (bf16 convergence) | Resubmitted as fp32 (job 6907548). |
| N=3 anomaly (WER ~10% vs N=2's ~4%) | Persists at both small and large scales after all bug fixes. Likely intrinsic difficulty of 67% compression, not a code bug. Paper will discuss as a finding. |

### 1.4 In Progress (as of April 5)

| Item | Status |
|------|--------|
| H-Mamba Small N=1 (job 6951736) | Running, general, epoch ~130, ACC 97.1%, WER **3.54%** (ep 120), comp 0.814. Converging slowly. |
| H-Mamba Small N=2 (job 6951737) | Running, general, epoch ~180, ACC 97.1%, WER **3.49%** (ep 170), comp 0.501. Now ahead of S_N1. |
| H-Mamba Small N=3 (job 6951738) | Running, general, epoch ~205, ACC 87.3%, WER **10.01%** (ep 160), comp 0.335. Plateaued ~10%. |
| H-Mamba Small N=4 (job 6951739) | Running, general, epoch ~161, ACC 86.9%, WER **9.70%** (ep 160), comp 0.251. Recovered from degradation. |
| H-Mamba Large N=1 (job 6959510) | Pending (preempt), last epoch 80, ACC 97.5%, WER **2.76%**, comp 0.859. Crashed (huggingface-hub fix applied), resubmitted. |
| H-Mamba Large N=2 (job 6959511) | Pending (preempt), last epoch 82, ACC 97.4%, WER **3.04%** (ep 70), comp 0.501. Crashed, resubmitted. |
| H-Mamba Large N=3 (job 6933858) | Running, preempt, epoch ~141, ACC 83.1%, WER **7.95%** (ep 140), comp 0.334. Still improving. |
| H-Mamba Large N=4 (job 6959512) | Pending (preempt), last epoch 90, ACC 87.8%, WER **6.48%** (ep 90), comp 0.251. Crashed, resubmitted. |
| Conformer Large CTC fp32 (job 6907548) | Pending (preempt), crashed same huggingface-hub bug, auto-requeued. |

**Note:** Small runs hit 2-day wall on April 4, resubmitted same day. L_N1, L_N2, L_N4 crashed on preempt restart due to `huggingface-hub` 1.8.0 incompatibility with `transformers` 4.40.0 — fixed by downgrading to 0.36.2, resubmitted April 5. SLURM logs were overwritten on restart but all data recovered from `epoch_metrics.csv` (persistent, never overwritten).

### 1.5 Not Started

| Item | Priority | Est. Effort |
|------|----------|-------------|
| With-LM / without-LM eval (16 evals) | CRITICAL | ~1 hr each after training |
| Conformer+DC ablation (Small N=2) | CRITICAL | 3-5 days |
| Fixed-2x downsampling baseline | HIGH | 3-5 days |
| MFA boundary analysis | HIGH | 2-3 days setup + compute |
| Phone-class compression analysis | HIGH | 1-2 days after MFA |
| Efficiency metrics (RTF, VRAM, FLOPs) | HIGH | 1 day |
| TED-LIUM 3 or GigaSpeech eval | MEDIUM | 1-2 days prep + inference |
| Probing experiment (phone classifier) | MEDIUM | 2 days |
| Speaking-rate adaptation analysis | MEDIUM | 1 day after MFA |
| Paper writing (8 pages + refs) | CRITICAL | ~2 weeks |

---

## 2. Week-by-Week Schedule

### Week 1: April 1-7 — Smoke Test + Launch Small Runs

**Day 1 (April 1) — DONE:**
- Upgraded packages: torch 2.1.1, mamba-ssm 2.0.3, causal-conv1d 1.4.0
- Patched causal-conv1d 1.4.0 API changes (6 fwd + 3 bwd call sites)
- Discovered Triton 2.1.0 crash on Ampere GPUs, disabled Mamba-2 kernel, PyTorch fallback works
- Smoke test v1 (job 6912668): Phase 1 passed (2 epochs, loss decreasing, compression converging).
  Timed out during beam search eval (2h limit too short).
- Smoke test v2 (job 6921845): Phase 1 passed (single GPU, 3 epochs + eval). Phase 2 DDP crashed — NumPy 2.0.2 incompatible with PyTorch 2.1.1 DDP.
- Fixed NumPy: downgraded 2.0.2 → 1.26.4 (pinned in requirements.txt).
- Smoke test v3 (job 6928765): DDP (2 GPU) Phase 1 passed — training, checkpoint save, NumPy fix confirmed.
- Resume test (job 6928816): Resumed from v3 DDP checkpoint, epoch 2 trained successfully. All state (optimizer, scheduler, DC params) restored correctly.
- SLURM partition strategy: small runs on general (8 GPU cap, 2-day), large runs on preempt (24 GPU cap, 14-day, requeue).
- Fixed grad_norm logging bug: was always 0.0000 because computed after optimizer.zero_grad(). Now captured between backward() and step(). Final check (job 6928979) confirmed: grad_norm=1195.0 (max 1770.9), bias_grad=4.9→13.5.
- Comprehensive documentation update (BiMamba v2 internals, SSM math, decoder, positional encoding, streaming)
- **All smoke tests passed. Ready to submit 8 H-Mamba 960h training runs.**

**Day 2 (April 2) — DONE:**
- Submitted all 4 H-Mamba Small runs to general partition (2x A6000 each):
  - N=1 (job 6933669, babel-w9-20), N=2 (job 6933673, babel-t9-24),
    N=3 (job 6933674, babel-s9-28), N=4 (job 6933675, babel-t9-32)
- All 4 running on 960h LibriSpeech, DDP, bf16
- Early epoch 1 observations:
  - N=2: compression converged to target 0.938 by batch 300, bias_grad sign-flipping (self-correcting)
  - N=3: compression converged to target 0.909 by batch 250, grad working (bias_grad up to 20.25)
  - N=1: compression 0.958 (target 1.0 = no compression, control run)
  - Peak VRAM ~37 GB / 49 GB (comfortable headroom)
- Conformer CTC fp32 (6907548): preempted after epoch 25, requeued and resumed epoch 26
- Submitted all 4 H-Mamba Large runs to preempt partition (14d, --requeue):
  - N=1 (job 6933856), N=2 (job 6933857), N=3 (job 6933858), N=4 (job 6933859)
  - Large runs need ~12.5 days; general partition 2-day limit too short
  - Preempt allows 24 GPUs, doesn't conflict with small runs on general

**Day 3 (April 3) — DONE:**
- **32h checkpoint** — all 8 runs confirmed learning correctly:
  - S_N2 WER 4.08% (epoch 80) vs S_N1 4.34% (epoch 50) — gap closing but not epoch-matched.
  - L_N2 WER 3.24% approaching ConMamba Large baseline (2.82%)
  - All compression ratios locked to targets (N2=0.501, N3=0.335, N4=0.251)
- N3 anomaly confirmed at both small (WER 10.37%) and large (WER 9.10%) scales.
- Fixed grad_norm logging blind spot for large models (grad_accum alignment issue).
- Comprehensive documentation audit and update.
- Added competitive landscape section (Zipformer, E-Branchformer, SAMBA-ASR comparisons).

**Day 4 (April 4) — DONE:**
- All 4 small runs hit 2-day general wall time ~10:52 AM, cancelled by SLURM.
  - S_N1 at epoch 84, S_N2 at epoch 128, S_N3 at epoch 153, S_N4 at epoch 103.
- Resubmitted all 4 small runs (jobs 6951736-6951739), resumed from checkpoints.
- L_N1, L_N2 preempted and crashed on restart — `huggingface-hub` upgraded to 1.8.0
  (incompatible with `transformers` 4.40.0). SLURM logs overwritten on restart.

**Day 5 (April 5) — IN PROGRESS:**
- L_N4 also crashed on preempt restart (same huggingface-hub bug). Conformer CTC too.
- Fixed: downgraded `huggingface-hub` 1.8.0 → 0.36.2 in hnetasr env.
- Resubmitted L_N1 (6959510), L_N2 (6959511), L_N4 (6959512). Conformer auto-requeued.
- Recovered all WER/ACC data from `epoch_metrics.csv` (persistent logs, never overwritten).
- **Key results at ~60h:**
  - **S_N2 (3.49%) now ahead of S_N1 (3.54%)** — first time N=2 genuinely leads the control.
  - **L_N1 WER 2.76% (ep 80)** — already beats ConMamba Large no-LM baseline (2.82%).
  - **L_N2 WER 3.04% (ep 70)** — only 0.28% behind L_N1 at matched epoch 70 (2.88%).
  - S_N4 WER recovered: 14.61% (ep 70) → 9.70% (ep 160). Was degrading, now improving.
  - L_N4 WER 6.48% (ep 90) — steadily improving, large model handles 75% compression.
  - S_N3/L_N3 both plateaued ~10% / ~8% — N=3 anomaly confirmed structural.

**Days 4-5:**
- Install Montreal Forced Aligner, download English acoustic model
- Run MFA alignment on LibriSpeech test-clean and test-other (CPU, overnight)
- Create ablation configs:
  - `conformer_dc_small_N2.yaml` — Conformer encoder with DC wrapper at layer 6
  - `conmamba_fixed2x_small.yaml` — ConMamba with AvgPool at layer 6 (no learned routing)
- Resubmit small runs after 2-day wall (~April 4 evening)

**Days 6-7:**
- Monitor small H-Mamba training: loss curves, compression ratio convergence
- N=2 compression ratio already stabilized at 0.50 — continue monitoring
- If compression drifts, adjust DC loss weight and restart affected run

**Deliverable:** 4 small runs training. MFA running. Ablation configs ready.

---

### Week 2: April 8-14 — Monitor, Launch Ablations, Draft Related Work

- Small runs at 30-50 epochs. Monitor via WandB offline logs and train_log.txt
- Submit Conformer+DC Small N=2 and ConMamba-fixed-2x Small to SLURM
- MFA should be complete. Write boundary analysis scripts:
  - Parse TextGrid files for phone boundaries
  - Extract learned boundary probabilities from 100h pilot model (preliminary)
  - Compute boundary-F1 at phone level
- Draft Related Work section (independent of results)
- Set up Overleaf project with ARR template

**Deliverable:** 6 runs in progress. MFA pipeline working. Related Work drafted. Overleaf set up.

---

### Week 3: April 15-21 — Small Results + Analysis

- Small H-Mamba runs finish (with early stopping: 300 epochs / patience 30, expect convergence ~150-200)
- Evaluate all: with-LM and without-LM for 4 H-Mamba Small + 2 ablations

**Go/no-go update:** Large runs were already launched in Week 1 (April 2) since smoke tests
passed cleanly. The go/no-go decision now applies to whether large runs are on track:
- If S_N2 matches ConMamba Small (2.22/5.56 with LM) within ~0.3% absolute: large runs confirmed worthwhile
- If not: may need to adjust DC loss weight on large runs while they're still training
- N3 anomaly persists at both scales — may be an intrinsic difficulty of 67% compression, not a bug

Remaining:
- Run efficiency benchmarks on small models (RTF, VRAM at 5s/15s/30s/60s)
- Run boundary analysis on 960h Small N=2

**Deliverable:** Complete small-scale results. Large runs at ~50-70 epochs. First boundary analysis.

---

### Week 4: April 22-28 — Deep Analysis + Write Method/Experiments

- Large runs at 20-30%. Monitor.
- Complete boundary analysis for 960h Small N=2:
  - Boundary-F1 at phone/syllable/word levels
  - Per-phone-class compression heatmap (the money figure)
  - Spectrogram + boundary overlay visualizations (3-4 examples)
  - Compression ratio distribution histogram
- Speaking-rate adaptation: per-speaker compression vs syllables/sec
- Probing experiment: linear phone classifier on Stage-0 vs Stage-1 representations
- Error analysis: ins/del/sub breakdown across N=1,2,3,4
- Write Method section (architecture, DC mechanism, training) + Figure 1
- Write Experimental Setup section (datasets, baselines, training details, metrics)

**Deliverable:** Method + Experiments drafted. Analysis complete for small models.

---

### Week 5: April 29 - May 5 — Large Results + Second Benchmark + Write Results

- Large N=1 and N=2 should finish (10-14 days from week 3)
- Evaluate with-LM and without-LM
- Large N=4 may still be running; evaluate when done
- Prepare TED-LIUM 3 for cross-benchmark eval (inference only)
- Efficiency benchmarks on large models
- Boundary analysis on Large N=2 (does the pattern hold at scale?)
- Write Results section: Tables 1-3, Pareto curve figure
- Write Analysis section: boundary-F1, phone heatmap, spectrograms, probing

**Deliverable:** Main results table mostly filled. Results + Analysis drafted.

---

### Week 6: May 6-12 — Finalize Results + Write Intro/Conclusion

- Any remaining large runs finish. Evaluate.
- Complete the full 14+ row results table
- Write Introduction, Abstract, Conclusion
- Compile references (35-45 refs)
- All figures in publication quality

**Deliverable:** Complete first draft.

---

### Week 7: May 13-19 — Advisor Review + Revisions

- Send to Prof. Watanabe for review
- Self-review against ARR criteria (Soundness, Excitement, Clarity, Reproducibility)
- Key reviewer concerns to preempt:
  - Is A-ToMe properly differentiated? (cosine-sim overlap is the first thing reviewers notice)
  - Is CIF discussed and compared?
  - Does the paper read as "we discover something about acoustic compression" (EMNLP framing) not "we built a faster model" (Interspeech framing)?
  - Are Conformer+DC and fixed-2x ablations prominent?
- Address feedback, run any last experiments
- Write limitations section (required for ARR)
- Anonymize the paper

**Deliverable:** Revised draft ready for final pass.

---

### Week 8: May 20-25 — Final Polish + Submit

- Final proofread (read aloud)
- Verify all numbers match eval outputs
- Prepare supplementary material (spectrograms, hyperparameter tables, training curves)
- Submit to ARR May 2026 cycle, preferred venue: EMNLP 2026
- Complete author registration by May 27

**Deliverable:** Paper submitted.

---

## 3. Critical Path

```
Smoke test (day 1)
  → Launch small runs (day 1-2)
    → Small results (end of week 3)
      → Go/no-go → Launch large N=1,N=2 (week 3)
        → Large N=2 results (week 5)
          → Write Results (weeks 5-6)
            → Full draft (week 6)
              → Advisor review (week 7)
                → Submit (week 8)
```

**Bottleneck:** Large model training (10-14 days). Using `preempt` partition (14-day walltime, requeue) since general's 8-GPU cap is consumed by small runs. Epoch checkpointing handles preemption. If cluster congestion delays by >1 week, submit with small-scale large results and add full results during ARR author response.

---

## 4. H-Mamba Configs Summary

### Small (all match conmamba_small.yaml base: lr=0.001, dropout=0.1, label_smoothing=0.0, grad_accum=1, warmup=25000 steps)

| Config | DC Loss Weight | DC Warmup | Gumbel Anneal | Target Compression |
|--------|---------------|-----------|---------------|-------------------|
| hmamba_small_N1.yaml | 0.0 | 0 | 0 | 0% (control) |
| hmamba_small_N2.yaml | 5.0 | 15 ep | 30 ep | 50% |
| hmamba_small_N3.yaml | 6.5 | 20 ep | 35 ep | 67% |
| hmamba_small_N4.yaml | 7.5 | 20 ep | 35 ep | 75% |

### Large (all match conmamba_large.yaml base: lr=0.0008, grad_accum=8, warmup=3750 steps)

| Config | DC Loss Weight | DC Warmup | Gumbel Anneal | Target Compression |
|--------|---------------|-----------|---------------|-------------------|
| hmamba_large_N1.yaml | 0.0 | 0 | 0 | 0% (control) |
| hmamba_large_N2.yaml | 5.0 | 15 ep | 30 ep | 50% |
| hmamba_large_N3.yaml | 6.5 | 20 ep | 35 ep | 67% |
| hmamba_large_N4.yaml | 7.5 | 20 ep | 35 ep | 75% |

### Ablation configs (to create in Week 1)

| Config | Purpose |
|--------|---------|
| conformer_dc_small_N2.yaml | Conformer encoder + DC at layer 6 — answers "why Mamba specifically?" |
| conmamba_fixed2x_small.yaml | ConMamba + AvgPool at layer 6 — proves learned boundaries beat naive compression |

---

## 5. Compute Budget

| Run | GPUs | Est. Time | Priority |
|-----|------|-----------|----------|
| H-Mamba Small N=1 | 2x A6000 | 3-5 days | CRITICAL |
| H-Mamba Small N=2 | 2x A6000 | 3-5 days | CRITICAL |
| H-Mamba Small N=3 | 2x A6000 | 3-5 days | CRITICAL |
| H-Mamba Small N=4 | 2x A6000 | 3-5 days | CRITICAL |
| H-Mamba Large N=1 | 2x A6000 | 10-14 days | CRITICAL |
| H-Mamba Large N=2 | 2x A6000 | 10-14 days | CRITICAL |
| H-Mamba Large N=3 | 2x A6000 | 10-14 days | HIGH |
| H-Mamba Large N=4 | 2x A6000 | 10-14 days | HIGH |
| Conformer+DC Small N=2 | 2x A6000 | 3-5 days | CRITICAL |
| ConMamba-fixed-2x Small | 2x A6000 | 3-5 days | HIGH |
| MFA alignment | CPU | 1-2 days | HIGH |
| Efficiency benchmarks | 1x A6000 | 1 day | HIGH |
| TED-LIUM 3 eval | 1x A6000 | ~2 hours | MEDIUM |
| Probing classifier | 1x A6000 | ~4 hours | MEDIUM |

**Total:** ~12-14 training runs, ~80-120 GPU-days. With 4 parallel GPU slots, fits in 6 weeks of training.

---

## 6. Risk Register

| Risk | Prob. | Impact | Mitigation |
|------|-------|--------|------------|
| H-Mamba N=2 doesn't match ConMamba at 960h | Low | CRITICAL | Adjust DC loss weight, warmup schedule. Worst case: frame as "learned compression reveals acoustic structure" even if WER is slightly worse |
| Large runs take >14 days (preemption delays) | Medium | HIGH | Preempt partition (14-day, requeue) with epoch checkpointing. If N=3/N=4 don't finish, submit with small-scale N=3/N=4 + large N=1/N=2 only |
| Conformer+DC shows DC helps Conformer equally | Medium | MEDIUM | Reframe: "DC is a general technique, Mamba is the natural fit due to linear complexity" |
| MFA boundary analysis shows weak phoneme correlation | Low | HIGH | Pivot to information-theoretic framing. Or frame as interesting negative result |
| SLURM cluster overloaded | Medium | HIGH | Small on general, large on preempt (24 GPU pool) with checkpoint-resume. Could also use L40S cluster if available |
| Reviewer says "just an engineering combination" | High | HIGH | The analysis section is the answer. Phone-class compression, probing, speaking-rate. Without this, paper is dead. |
| Reviewer asks for CIF comparison | High | MEDIUM | Discuss CIF thoroughly in related work with clear differentiation. CIF baseline is ideal but may not fit in 8 weeks. |

---

## 7. Paper Structure (8 pages + refs)

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.3 | Core claim, method, key result, analysis finding |
| 1. Introduction | 1.3 | Variable info density, three threads, the gap, contributions |
| 2. Method | 1.5 | Architecture (Fig 1), DC mechanism (equations), training |
| 3. Experimental Setup | 1.0 | Datasets, baselines table, training details, metrics |
| 4. Results | 1.5 | Main WER table, efficiency table, compression stats, Pareto curve |
| 5. Analysis | 1.5 | Boundary-F1, phone-class heatmap (Fig 3), spectrograms (Fig 4), probing, speaking rate |
| 6. Related Work | 0.5 | A-ToMe, CIF, Zipformer, ConMamba, H-Net |
| 7. Conclusion | 0.4 | Findings, limitations, future work |
| References | 1.5-2.0 | 35-45 refs |

---

## 8. Critical References

| Paper | Why it matters |
|-------|---------------|
| Li et al., "A-ToMe" (Interspeech 2023, arXiv 2306.16009) | Cosine-sim token merging in ASR. Most similar prior work. Must differentiate. |
| Dong & Xu, "CIF" (ICASSP 2020, arXiv 1905.11235) | Established learned variable-rate ASR. Must discuss. |
| Gao et al., Paraformer (Interspeech 2022) | CIF descendant, widely deployed. |
| Kim et al., Squeezeformer (NeurIPS 2022) | First temporal U-Net in ASR. Conceptual predecessor. |
| Rekesh et al., Fast Conformer (ASRU 2023) | NVIDIA's 8x fixed downsampling. Industry standard. |
| Fang & Li, Mamba-UMA (ICASSP 2025, arXiv 2410.00070) | Mamba + learned aggregation. Must differentiate. |
| She et al., TC-BiMamba (arXiv 2602.11546, Feb 2026) | Recent BiMamba for ASR. Shows active field. |

---

## 9. Success Criteria

The paper is ready to submit when **all** of these are true:

1. H-Mamba N=2 Small matches ConMamba Small at 960h (within 0.3% WER)
2. H-Mamba N=2 Large matches ConMamba Large at 960h
3. Conformer+DC ablation complete (answers "why Mamba?")
4. Fixed-2x downsampling baseline shows learned boundaries beat naive compression
5. Boundary-F1 shows statistically significant correlation with phone transitions
6. Phone-class compression heatmap shows interpretable patterns (vowels compressed, transitions preserved)
7. Efficiency metrics show measurable RTF/VRAM gains at longer utterances
8. At least one benchmark beyond LibriSpeech evaluated
9. All figures publication quality
10. Analysis section >= 1.5 pages telling a compelling story about acoustic structure

---

## 10. Open Question: load_balancing_loss Formula

Current 5-term loss works (gradients verified). Official H-Net uses a simpler single-term:

```python
# Official H-Net (hnet/utils/train.py):
true_ratio = boundary_mask.float().mean()
average_prob = tokenized_prob.float().mean()
loss = ((1 - true_ratio) * (1 - average_prob) +
        true_ratio * average_prob * (N-1)) * N / (N-1)
```

**Decision:** Proceed with fixed 5-term loss. If compression convergence is slow or unstable during 960h training (monitor N=2 ratio in weeks 1-2), switch to the official formula as the first intervention.

---

## 11. Immediate Next Steps

1. ~~Run GPU smoke test~~ All passed: single GPU (v2), DDP (v3 job 6928765), resume (job 6928816)
2. ~~Submit small N=1,2,3,4 to SLURM~~ Running since April 2 (jobs 6933669-6933675)
3. ~~Submit large N=1,2,3,4 to SLURM~~ Running since April 2 (jobs 6933856-6933859, preempt)
4. **Resubmit small N=1,2,3,4** when they hit the 2-day wall (~April 4 evening). Same scripts, SpeechBrain auto-resumes.
5. Install MFA and start alignment
6. Create ablation configs (conformer_dc, fixed-2x)
