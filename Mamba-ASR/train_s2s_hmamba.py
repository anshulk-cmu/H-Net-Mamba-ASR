#!/usr/bin/env python3
"""
Recipe for training H-Mamba ASR system with Dynamic Chunking.

This extends the ConMamba S2S training to include:
- HMambaEncoderWrapper with learned compression
- DC load balancing loss
- Compression ratio warm-up
- DC metrics logging

Usage:
    python train_s2s_hmamba.py hparams/hmamba_S_S2S.yaml

Authors:
    Anshul Kumar 2024
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main, if_main_process

# Import H-Mamba wrapper
from modules.HMambaEncoderWrapper import HMambaEncoderWrapper, create_hmamba_from_conmamba

# Import H-Mamba logger
from modules.hmamba_logger import HMambaLogger, Timer

logger = logging.getLogger(__name__)

os.environ['WANDB__SERVICE_WAIT'] = '999999'


class HMambaASR(sb.core.Brain):
    """ASR Brain with H-Mamba encoder and Dynamic Chunking."""
    
    def on_fit_start(self):
        """Called at the beginning of fit(). Wrap encoder with H-Mamba."""
        # IMPORTANT: Wrap encoder BEFORE super().on_fit_start() so DC params are in optimizer
        if not hasattr(self, '_hmamba_initialized'):
            self._wrap_encoder_with_hmamba()
            self._hmamba_initialized = True
        
        # Now call super() which initializes optimizer with all modules (including DC)
        super().on_fit_start()
        
        # Verify DC params are in optimizer
        if hasattr(self, 'optimizer') and hasattr(self, 'hmamba_encoder'):
            dc_param_ids = {id(p) for p in self.hmamba_encoder.routing_module.parameters()}
            opt_param_ids = set()
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    opt_param_ids.add(id(p))
            
            dc_in_opt = dc_param_ids.issubset(opt_param_ids)
            logger.info(f"[H-Mamba] DC params in optimizer: {dc_in_opt}")
            if not dc_in_opt:
                logger.warning("[H-Mamba] DC params NOT in optimizer! Adding them now...")
                self.optimizer.add_param_group({
                    'params': list(self.hmamba_encoder.routing_module.parameters())
                })
        
        # Initialize H-Mamba logger
        if not hasattr(self, 'hmamba_logger'):
            use_hmamba_logger = getattr(self.hparams, 'use_hmamba_logger', True)
            if use_hmamba_logger:
                log_dir = os.path.join(self.hparams.output_folder, "hmamba_logs")
                experiment_name = getattr(self.hparams, 'experiment_name', 'hmamba_asr')
                log_every_n = getattr(self.hparams, 'hmamba_log_every_n_batches', 10)
                self.hmamba_logger = HMambaLogger(
                    log_dir=log_dir,
                    experiment_name=experiment_name,
                    use_tensorboard=getattr(self.hparams, 'use_tensorboard', True),
                    log_every_n_batches=log_every_n,
                    gpu_device_id=0,
                    verbose=True,
                )
                # Save experiment config
                config = {
                    'd_model': self.hparams.d_model,
                    'hmamba_split_idx': getattr(self.hparams, 'hmamba_split_idx', 6),
                    'hmamba_target_N': getattr(self.hparams, 'hmamba_target_N', 2.0),
                    'hmamba_warmup_epochs': getattr(self.hparams, 'hmamba_warmup_epochs', 20),
                    'hmamba_dc_loss_weight': getattr(self.hparams, 'hmamba_dc_loss_weight', 0.03),
                    'ctc_weight': self.hparams.ctc_weight,
                    'batch_size': getattr(self.hparams, 'batch_size', 'dynamic'),
                }
                self.hmamba_logger.save_experiment_config(config)
            else:
                self.hmamba_logger = None
        
        # Initialize batch timing tracking
        self.batch_start_time = None
        self.current_batch_idx = 0
    
    def _wrap_encoder_with_hmamba(self):
        """Replace ConMamba encoder with HMambaEncoderWrapper."""
        # Get the original encoder from TransformerASR
        original_encoder = self.modules.Transformer.encoder
        
        # Get H-Mamba hyperparameters with defaults
        d_model = self.hparams.d_model
        split_idx = getattr(self.hparams, 'hmamba_split_idx', 6)
        target_N = getattr(self.hparams, 'hmamba_target_N', 2.0)
        headdim = getattr(self.hparams, 'hmamba_headdim', 36)
        
        # Create wrapped encoder
        hmamba_encoder = create_hmamba_from_conmamba(
            conmamba_encoder=original_encoder,
            d_model=d_model,
            split_idx=split_idx,
            target_compression_N=target_N,
            headdim=headdim,
        )
        
        # Move to correct device
        hmamba_encoder = hmamba_encoder.to(self.device)
        
        # Replace encoder in Transformer
        self.modules.Transformer.encoder = hmamba_encoder
        
        # Store reference for easy access
        self.hmamba_encoder = hmamba_encoder
        
        # Initialize DC metrics tracking
        self.dc_metrics = {
            'compression_ratio': [],
            'dc_loss': [],
            'boundary_prob_mean': [],
        }
        
        logger.info(f"[H-Mamba] Encoder wrapped with DC (split_idx={split_idx}, target_N={target_N})")
        
        # Verify DC parameters are in optimizer (one-time check)
        dc_params = list(hmamba_encoder.routing_module.parameters())
        logger.info(f"[H-Mamba] DC RoutingModule has {len(dc_params)} parameters")
        for name, param in hmamba_encoder.routing_module.named_parameters():
            logger.info(f"[H-Mamba]   - {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    def _update_gumbel_temperature(self):
        """Anneal Gumbel-softmax temperature for sharper decisions over time."""
        if not hasattr(self, 'hmamba_encoder'):
            return
            
        # Gumbel temperature annealing schedule
        gumbel_start = getattr(self.hparams, 'hmamba_gumbel_start', 1.0)
        gumbel_end = getattr(self.hparams, 'hmamba_gumbel_end', 0.5)
        gumbel_anneal_epochs = getattr(self.hparams, 'hmamba_gumbel_anneal_epochs', 30)
        
        current_epoch = self.hparams.epoch_counter.current
        
        if current_epoch < gumbel_anneal_epochs:
            # Linear annealing from start to end
            progress = current_epoch / gumbel_anneal_epochs
            gumbel_tau = gumbel_start - (gumbel_start - gumbel_end) * progress
        else:
            gumbel_tau = gumbel_end
        
        # Update Gumbel temperature in routing module
        self.hmamba_encoder.routing_module.gumbel_tau = gumbel_tau
        
        return gumbel_tau
    
    def _get_current_target_N(self):
        """Get current target compression N with warm-up schedule."""
        warmup_epochs = getattr(self.hparams, 'hmamba_warmup_epochs', 20)
        target_N = getattr(self.hparams, 'hmamba_target_N', 2.0)
        
        current_epoch = self.hparams.epoch_counter.current
        
        if current_epoch < warmup_epochs:
            # Linear warm-up from N=1.0 (no compression) to target_N
            progress = current_epoch / warmup_epochs
            current_N = 1.0 + (target_N - 1.0) * progress
        else:
            current_N = target_N
        
        return current_N
    
    def compute_forward(self, batch, stage):
        """Forward computations with H-Mamba encoder."""
        # Start forward timing
        if stage == sb.Stage.TRAIN:
            self.forward_start_time = time.perf_counter()
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        
        # Store audio duration for RTF calculation
        if stage == sb.Stage.TRAIN:
            sample_rate = getattr(self.hparams, 'sample_rate', 16000)
            self.current_audio_duration = (wavs.shape[1] / sample_rate) * wavs.shape[0]

        # Compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Feature augmentation during training
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # CNN frontend
        src = self.modules.CNN(feats)

        # Update target compression ratio based on warm-up schedule
        if stage == sb.Stage.TRAIN and hasattr(self, 'hmamba_encoder'):
            self.hmamba_encoder.target_compression_N = self._get_current_target_N()

        # Forward through Transformer (with H-Mamba encoder)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # CTC output
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # Seq2Seq output
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Beam search for validation/test
        hyps = None
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), wav_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), wav_lens
                )

        # End forward timing
        if stage == sb.Stage.TRAIN:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.forward_time_ms = (time.perf_counter() - self.forward_start_time) * 1000
        
        return p_ctc, p_seq, wav_lens, hyps, src

    def compute_objectives(self, predictions, batch, stage):
        """Computes loss including DC load balancing loss."""
        p_ctc, p_seq, wav_lens, hyps, src = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Handle augmentation label replication
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(tokens_lens)
                tokens_eos = self.hparams.fea_augment.replicate_labels(tokens_eos)
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(tokens_eos_lens)

        # Seq2Seq loss
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        # CTC loss
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        # Base ASR loss
        loss_asr = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        # DC loss (only during training)
        loss_dc = torch.tensor(0.0, device=self.device)
        if stage == sb.Stage.TRAIN and hasattr(self, 'hmamba_encoder'):
            dc_loss_weight = getattr(self.hparams, 'hmamba_dc_loss_weight', 0.03)
            
            # Get DC loss stored during forward pass (avoids shape mismatch with raw CNN output)
            loss_dc = getattr(self.hmamba_encoder, 'last_dc_loss', torch.tensor(0.0, device=self.device))
            
            # Track metrics using stored values from forward pass
            compression_ratio = getattr(self.hmamba_encoder, 'last_compression_ratio', 0.0)
            self.dc_metrics['compression_ratio'].append(compression_ratio)
            self.dc_metrics['dc_loss'].append(loss_dc.item() if torch.is_tensor(loss_dc) else loss_dc)
        
        # Total loss
        loss = loss_asr + getattr(self.hparams, 'hmamba_dc_loss_weight', 0.03) * loss_dc
        
        # Store losses for batch logging
        if stage == sb.Stage.TRAIN:
            self.current_losses = {
                'total': loss.item(),
                'ctc': loss_ctc.item(),
                'seq': loss_seq.item(),
                'dc': loss_dc.item() if torch.is_tensor(loss_dc) else loss_dc,
            }
            
            # Get bias value and gradient for debugging
            bias_val = 0.0
            bias_grad = 0.0
            if hasattr(self, 'hmamba_encoder'):
                routing = self.hmamba_encoder.routing_module
                bias_val = routing.boundary_bias.item()
                if routing.boundary_bias.grad is not None:
                    bias_grad = routing.boundary_bias.grad.item()
            
            self.current_dc_stats = {
                'compression_ratio': compression_ratio,
                'num_chunks': getattr(self.hmamba_encoder, 'last_num_chunks', 0) if hasattr(self, 'hmamba_encoder') else 0,
                'avg_chunk_size': 0.0,
                'boundary_prob_mean': 0.0,
                'bias': bias_val,
                'bias_grad': bias_grad,
            }

        # Evaluation metrics
        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.wer_metric.append(ids, predicted_words, target_words)

            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

        return loss

    def on_stage_start(self, stage, epoch):
        """Called at the beginning of each stage."""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()
        else:
            # Reset DC metrics for new epoch
            self.dc_metrics = {
                'compression_ratio': [],
                'dc_loss': [],
                'boundary_prob_mean': [],
            }
            # Reset batch counter and start epoch timer
            self.current_batch_idx = 0
            self.epoch_start_time = time.time()
            
            # Update Gumbel temperature for this epoch (annealing)
            self._update_gumbel_temperature()
            
            # Log current DC parameters
            if hasattr(self, 'hmamba_encoder'):
                routing = self.hmamba_encoder.routing_module
                current_N = self._get_current_target_N()
                logger.info(
                    f"[H-Mamba] Epoch {epoch}: target_N={current_N:.2f}, "
                    f"gumbel_tau={routing.gumbel_tau:.3f}, "
                    f"temp={routing.temperature.item():.3f}, "
                    f"bias={routing.boundary_bias.item():.3f}"
                )
            
            # Start epoch in logger
            if hasattr(self, 'hmamba_logger') and self.hmamba_logger is not None:
                self.hmamba_logger.start_epoch(epoch)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Called at the end of each stage."""
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            
            # Add DC metrics to training stats
            if self.dc_metrics['compression_ratio']:
                avg_compression = sum(self.dc_metrics['compression_ratio']) / len(self.dc_metrics['compression_ratio'])
                avg_dc_loss = sum(self.dc_metrics['dc_loss']) / len(self.dc_metrics['dc_loss'])
                
                stage_stats['compression_ratio'] = avg_compression
                stage_stats['dc_loss'] = avg_dc_loss
                stage_stats['target_N'] = self._get_current_target_N()
                
                logger.info(f"[H-Mamba] Epoch {epoch}: compression={avg_compression:.3f}, "
                           f"dc_loss={avg_dc_loss:.4f}, target_N={self._get_current_target_N():.2f}")
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Logging and checkpointing
        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            
            # Log epoch to HMamba logger
            if hasattr(self, 'hmamba_logger') and self.hmamba_logger is not None:
                valid_wer = stage_stats.get('WER', None)
                self.hmamba_logger.log_epoch(
                    epoch=epoch,
                    valid_loss=stage_loss,
                    valid_acc=stage_stats.get('ACC', 0.0),
                    valid_wer=valid_wer,
                    valid_cer=None,
                    learning_rate=lr,
                )

            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint average if needed."""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model",
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")
        
        # Ensure H-Mamba wrapper is initialized for evaluation
        if not hasattr(self, '_hmamba_initialized'):
            self._wrap_encoder_with_hmamba()
            self._hmamba_initialized = True

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Apply learning rate scheduling and batch logging."""
        # Compute gradient norm before optimizer step
        grad_norm = 0.0
        if should_step:
            total_norm = 0.0
            for p in self.modules.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
        
        if should_step:
            self.hparams.noam_annealing(self.optimizer)
        
        # Update bias gradient info AFTER backward pass (more accurate)
        if hasattr(self, 'hmamba_encoder') and hasattr(self, 'current_dc_stats'):
            routing = self.hmamba_encoder.routing_module
            self.current_dc_stats['bias'] = routing.boundary_bias.item()
            if routing.boundary_bias.grad is not None:
                self.current_dc_stats['bias_grad'] = routing.boundary_bias.grad.item()
            else:
                self.current_dc_stats['bias_grad'] = 0.0
        
        # Log batch metrics
        if hasattr(self, 'hmamba_logger') and self.hmamba_logger is not None:
            # Get timing info
            batch_time_ms = getattr(self, 'forward_time_ms', 0.0)
            forward_time_ms = getattr(self, 'forward_time_ms', 0.0)
            
            # Get stored losses and DC stats
            losses = getattr(self, 'current_losses', {'total': loss.item(), 'ctc': 0, 'seq': 0, 'dc': 0})
            dc_stats = getattr(self, 'current_dc_stats', {'compression_ratio': 0, 'num_chunks': 0, 'avg_chunk_size': 0, 'boundary_prob_mean': 0})
            
            # Get audio duration
            audio_duration = getattr(self, 'current_audio_duration', 0.0)
            
            # Get current target N
            target_N = self._get_current_target_N()
            
            self.hmamba_logger.log_batch(
                epoch=self.hparams.epoch_counter.current,
                batch_idx=self.current_batch_idx,
                losses=losses,
                dc_stats=dc_stats,
                timing={'batch': batch_time_ms, 'forward': forward_time_ms, 'backward': 0.0, 'data_load': 0.0},
                grad_norm=grad_norm,
                audio_duration_sec=audio_duration,
                target_N=target_N,
            )
        
        self.current_batch_idx += 1


def dataio_prepare(hparams):
    """Prepare datasets for training."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    tokenizer = hparams["tokenizer"]

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides("wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # CLI parsing
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # DDP initialization
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation
    from librispeech_prepare import prepare_librispeech

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

    # Load pretrained LM and tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # WandB logging
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()

    if hparams['no_lm']:
        print('Evaluate without LM.')
        hparams['test_search'] = hparams['valid_search']
        hparams["output_wer_folder"] = os.path.join(hparams["output_wer_folder"], 'no_lm')

    # Initialize H-Mamba ASR trainer
    asr_brain = HMambaASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = train_dataloader_opts.get("collate_fn", None)
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = valid_dataloader_opts.get("collate_fn", None)
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}
        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    # Training
    if not hparams['skip_train']:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
    
    # Close HMamba logger and generate summary report
    if hasattr(asr_brain, 'hmamba_logger') and asr_brain.hmamba_logger is not None:
        asr_brain.hmamba_logger.close()