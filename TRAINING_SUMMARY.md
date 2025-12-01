# ConMamba Baseline Training - Ready to Run

## Configuration Summary

✅ **All setup complete** - Ready for training

### Data
- **Location:** `/home/anshulk/hnet_mamba_asr/data/LibriSpeech`
- **Training:** train-clean-100 (100 hours, 28,539 utterances)
- **Validation:** dev-clean (5.4 hours)
- **Test:** test-clean + test-other (10.4 hours)
- **Format:** 16kHz mono FLAC files ✓

### Model Configuration
- **Architecture:** ConMamba (12 encoder layers, 4 decoder layers)
- **Hidden dim:** 144
- **Encoder:** Bidirectional Mamba blocks
- **Decoder:** Transformer with CTC/Attention (weight: 0.3)

### Training Hyperparameters
- **GPUs:** 4x A6000 (48GB each)
- **Batch size:** 16 per GPU
- **Grad accumulation:** 2 steps
- **Global batch:** 16 × 4 × 2 = **128** ✓
- **Epochs:** 70
- **Precision:** BF16
- **Optimizer:** Adam (lr=0.001, warmup=15k steps)
- **Dropout:** 0.15
- **Label smoothing:** 0.1

### Expected Results
- **Training time:** ~4-5 hours on 4x A6000
- **Target WER (test-clean):** 7-9% (CTC-only, no LM)
- **With LM:** ~6% (comparable to Transformer baselines)

## How to Run

### Start Training
```bash
cd ~/hnet_mamba_asr
./run_baseline_training.sh
```

### Monitor Progress
Training logs: `Mamba-ASR/results/S2S/conmamba_S_S2S/7775/train_log.txt`

### Key Checkpoints
- **Epoch 1-10:** Warmup phase, loss decreases rapidly
- **Epoch 20-40:** Main convergence, WER improves steadily
- **Epoch 50-70:** Fine-tuning, WER plateaus

## Next Steps After Baseline

Once this completes with ~7-9% WER, you'll:

1. **Integrate H-Net Dynamic Chunking:**
   - Add DC layer after Mamba stage 0 (8 blocks)
   - Remaining 4 blocks operate on compressed chunks
   - Target compression ratio: 0.5 (50% reduction)

2. **Expected Improvements:**
   - **Speed:** 40-60% faster inference (fewer tokens processed)
   - **WER:** Maintain within 5-10% relative degradation
   - **Research contribution:** Learned adaptive segmentation

3. **Add Language Model for Final WER:**
   - External Transformer LM (already pretrained)
   - Should push WER below 6% on test-clean

---

**Config file:** `Mamba-ASR/hparams/S2S/conmamba_small_ls100.yaml`
**Training script:** `run_baseline_training.sh`
**Results directory:** `Mamba-ASR/results/S2S/conmamba_S_S2S/7775/`
