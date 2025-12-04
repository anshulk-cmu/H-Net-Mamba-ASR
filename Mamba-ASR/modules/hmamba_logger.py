"""
H-Mamba ASR Research Logger

Comprehensive logging for Dynamic Chunking ASR experiments including:
- GPU/VRAM monitoring
- Real-Time Factor (RTF) computation
- Compression statistics tracking
- Training dynamics (loss, gradients, learning rate)
- WER/CER tracking
- Per-epoch and per-batch statistics
- CSV export for analysis
- TensorBoard integration (optional)

Usage:
    from hmamba_logger import HMambaLogger
    
    logger = HMambaLogger(
        log_dir="results/S2S/hmamba_S_S2S/7778/logs",
        experiment_name="hmamba_N2_split6",
        use_tensorboard=True,
    )
    
    # In training loop:
    logger.log_batch(batch_idx, metrics_dict)
    logger.log_epoch(epoch, train_stats, valid_stats)

Author: Anshul Kumar 2024
"""

import os
import sys
import csv
import json
import time
import torch
import psutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import threading

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# =============================================================================
# Data Classes for Structured Logging
# =============================================================================

@dataclass
class GPUMetrics:
    """GPU utilization and memory metrics."""
    gpu_id: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    gpu_utilization: float
    temperature: float
    power_draw: float
    timestamp: str


@dataclass
class BatchMetrics:
    """Per-batch training metrics."""
    epoch: int
    batch_idx: int
    global_step: int
    
    # Losses
    total_loss: float
    ctc_loss: float
    seq_loss: float
    dc_loss: float
    
    # DC statistics
    compression_ratio: float
    num_chunks: int
    avg_chunk_size: float
    boundary_prob_mean: float
    target_N: float
    
    # Timing
    batch_time_ms: float
    data_load_time_ms: float
    forward_time_ms: float
    backward_time_ms: float
    
    # GPU
    vram_used_mb: float
    vram_allocated_mb: float
    
    # Gradient stats
    grad_norm: float
    
    timestamp: str


@dataclass
class EpochMetrics:
    """Per-epoch aggregated metrics."""
    epoch: int
    
    # Training
    train_loss: float
    train_ctc_loss: float
    train_seq_loss: float
    train_dc_loss: float
    
    # Validation
    valid_loss: float
    valid_acc: float
    valid_wer: Optional[float]
    valid_cer: Optional[float]
    
    # DC statistics
    avg_compression_ratio: float
    target_compression_ratio: float
    compression_ratio_std: float
    avg_num_chunks: float
    avg_boundary_prob: float
    
    # Learning rate
    learning_rate: float
    
    # Timing
    epoch_time_sec: float
    avg_batch_time_ms: float
    total_audio_sec: float
    rtf: float
    
    # GPU
    peak_vram_mb: float
    avg_vram_mb: float
    
    # Gradient stats
    avg_grad_norm: float
    max_grad_norm: float
    
    timestamp: str


@dataclass 
class InferenceMetrics:
    """Inference/evaluation metrics."""
    dataset_name: str
    num_samples: int
    wer: float
    cer: float
    total_inference_time_sec: float
    total_audio_duration_sec: float
    rtf: float
    avg_latency_ms: float
    avg_compression_ratio: float
    peak_vram_mb: float
    timestamp: str


# =============================================================================
# GPU Monitoring Utilities
# =============================================================================

class GPUMonitor:
    """Monitor GPU metrics using pynvml or torch."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.nvml_available = False
        
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.nvml_available = True
            self.pynvml = pynvml
        except (ImportError, Exception):
            pass
    
    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        timestamp = datetime.now().isoformat()
        
        if self.nvml_available:
            return self._get_nvml_metrics(timestamp)
        else:
            return self._get_torch_metrics(timestamp)
    
    def _get_nvml_metrics(self, timestamp: str) -> GPUMetrics:
        """Get detailed metrics via NVML."""
        pynvml = self.pynvml
        handle = self.nvml_handle
        
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = 0.0
        
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except:
            power = 0.0
        
        return GPUMetrics(
            gpu_id=self.device_id,
            name=name,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_total_mb=memory.total / 1024 / 1024,
            memory_percent=memory.used / memory.total * 100,
            gpu_utilization=utilization.gpu,
            temperature=temperature,
            power_draw=power,
            timestamp=timestamp,
        )
    
    def _get_torch_metrics(self, timestamp: str) -> GPUMetrics:
        """Get basic metrics via PyTorch (fallback)."""
        if not torch.cuda.is_available():
            return GPUMetrics(
                gpu_id=self.device_id, name="N/A", memory_used_mb=0,
                memory_total_mb=0, memory_percent=0, gpu_utilization=0,
                temperature=0, power_draw=0, timestamp=timestamp,
            )
        
        memory_allocated = torch.cuda.memory_allocated(self.device_id) / 1024 / 1024
        memory_reserved = torch.cuda.memory_reserved(self.device_id) / 1024 / 1024
        memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024 / 1024
        
        return GPUMetrics(
            gpu_id=self.device_id,
            name=torch.cuda.get_device_name(self.device_id),
            memory_used_mb=memory_reserved,
            memory_total_mb=memory_total,
            memory_percent=memory_reserved / memory_total * 100,
            gpu_utilization=0,
            temperature=0,
            power_draw=0,
            timestamp=timestamp,
        )
    
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device_id) / 1024 / 1024
        return 0.0
    
    def reset_peak_memory(self):
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)


# =============================================================================
# Timer Utility
# =============================================================================

class Timer:
    """High-precision timer for profiling."""
    
    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.start_time = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
    
    def elapsed(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_ms


# =============================================================================
# Main Logger Class
# =============================================================================

class HMambaLogger:
    """
    Comprehensive logger for H-Mamba ASR research experiments.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        log_every_n_batches: int = 10,
        gpu_device_id: int = 0,
        sample_rate: int = 16000,
        verbose: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.log_every_n_batches = log_every_n_batches
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_monitor = GPUMonitor(gpu_device_id)
        
        self.global_step = 0
        self.current_epoch = 0
        self.epoch_start_time = None
        self.batch_metrics_buffer: List[BatchMetrics] = []
        self.epoch_metrics_history: List[EpochMetrics] = []
        
        self._reset_epoch_accumulators()
        self._setup_file_logger()
        
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        self.batch_csv_path = self.log_dir / "batch_metrics.csv"
        self.epoch_csv_path = self.log_dir / "epoch_metrics.csv"
        self.inference_csv_path = self.log_dir / "inference_metrics.csv"
        
        self._init_csv_files()
        
        self.logger.info(f"HMambaLogger initialized: {self.log_dir}")
        self.logger.info(f"TensorBoard: {'enabled' if self.use_tensorboard else 'disabled'}")
    
    def _setup_file_logger(self):
        """Setup Python file logger."""
        self.logger = logging.getLogger(f"HMamba.{self.experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.verbose else logging.WARNING)
        ch.setFormatter(logging.Formatter('%(message)s'))
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        if not self.batch_csv_path.exists():
            with open(self.batch_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'batch_idx', 'global_step',
                    'total_loss', 'ctc_loss', 'seq_loss', 'dc_loss',
                    'compression_ratio', 'num_chunks', 'avg_chunk_size', 
                    'boundary_prob_mean', 'target_N',
                    'batch_time_ms', 'forward_time_ms', 'backward_time_ms',
                    'vram_used_mb', 'vram_allocated_mb', 'grad_norm',
                    'timestamp'
                ])
        
        if not self.epoch_csv_path.exists():
            with open(self.epoch_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_ctc_loss', 'train_seq_loss', 'train_dc_loss',
                    'valid_loss', 'valid_acc', 'valid_wer', 'valid_cer',
                    'avg_compression_ratio', 'target_compression_ratio', 'compression_ratio_std',
                    'avg_num_chunks', 'avg_boundary_prob',
                    'learning_rate', 'epoch_time_sec', 'avg_batch_time_ms',
                    'total_audio_sec', 'rtf', 'peak_vram_mb', 'avg_vram_mb',
                    'avg_grad_norm', 'max_grad_norm', 'timestamp'
                ])
    
    def _reset_epoch_accumulators(self):
        """Reset accumulators for new epoch."""
        self.acc = {
            'total_loss': [],
            'ctc_loss': [],
            'seq_loss': [],
            'dc_loss': [],
            'compression_ratio': [],
            'num_chunks': [],
            'boundary_prob': [],
            'batch_time': [],
            'vram': [],
            'grad_norm': [],
            'audio_duration': [],
            'target_N': [],
        }
    
    # =========================================================================
    # Batch-Level Logging
    # =========================================================================
    
    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        losses: Dict[str, float],
        dc_stats: Dict[str, Any],
        timing: Dict[str, float],
        grad_norm: float,
        audio_duration_sec: float,
        target_N: float,
    ):
        """Log metrics for a single training batch."""
        self.global_step += 1
        self.current_epoch = epoch
        
        gpu_metrics = self.gpu_monitor.get_metrics()
        
        metrics = BatchMetrics(
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=self.global_step,
            total_loss=losses.get('total', 0.0),
            ctc_loss=losses.get('ctc', 0.0),
            seq_loss=losses.get('seq', 0.0),
            dc_loss=losses.get('dc', 0.0),
            compression_ratio=dc_stats.get('compression_ratio', 0.0),
            num_chunks=dc_stats.get('num_chunks', 0),
            avg_chunk_size=dc_stats.get('avg_chunk_size', 0.0),
            boundary_prob_mean=dc_stats.get('boundary_prob_mean', 0.0),
            target_N=target_N,
            batch_time_ms=timing.get('batch', 0.0),
            data_load_time_ms=timing.get('data_load', 0.0),
            forward_time_ms=timing.get('forward', 0.0),
            backward_time_ms=timing.get('backward', 0.0),
            vram_used_mb=gpu_metrics.memory_used_mb,
            vram_allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
            grad_norm=grad_norm,
            timestamp=datetime.now().isoformat(),
        )
        
        # Accumulate
        self.acc['total_loss'].append(metrics.total_loss)
        self.acc['ctc_loss'].append(metrics.ctc_loss)
        self.acc['seq_loss'].append(metrics.seq_loss)
        self.acc['dc_loss'].append(metrics.dc_loss)
        self.acc['compression_ratio'].append(metrics.compression_ratio)
        self.acc['num_chunks'].append(metrics.num_chunks)
        self.acc['boundary_prob'].append(metrics.boundary_prob_mean)
        self.acc['batch_time'].append(metrics.batch_time_ms)
        self.acc['vram'].append(metrics.vram_used_mb)
        self.acc['grad_norm'].append(metrics.grad_norm)
        self.acc['audio_duration'].append(audio_duration_sec)
        self.acc['target_N'].append(target_N)
        
        if batch_idx % self.log_every_n_batches == 0:
            self._write_batch_csv(metrics)
            
            if self.tb_writer:
                self._write_batch_tensorboard(metrics)
            
            if self.verbose:
                self.logger.info(
                    f"E{epoch} B{batch_idx:04d} | "
                    f"loss={metrics.total_loss:.3f} "
                    f"(ctc={metrics.ctc_loss:.3f} seq={metrics.seq_loss:.3f} dc={metrics.dc_loss:.4f}) | "
                    f"comp={metrics.compression_ratio:.3f} (target={1/target_N:.3f}) | "
                    f"VRAM={metrics.vram_used_mb:.0f}MB | "
                    f"t={metrics.batch_time_ms:.0f}ms"
                )
    
    def _write_batch_csv(self, metrics: BatchMetrics):
        """Append batch metrics to CSV."""
        with open(self.batch_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.epoch, metrics.batch_idx, metrics.global_step,
                f"{metrics.total_loss:.6f}", f"{metrics.ctc_loss:.6f}", 
                f"{metrics.seq_loss:.6f}", f"{metrics.dc_loss:.6f}",
                f"{metrics.compression_ratio:.6f}", metrics.num_chunks, 
                f"{metrics.avg_chunk_size:.2f}", f"{metrics.boundary_prob_mean:.6f}",
                f"{metrics.target_N:.2f}",
                f"{metrics.batch_time_ms:.2f}", f"{metrics.forward_time_ms:.2f}",
                f"{metrics.backward_time_ms:.2f}",
                f"{metrics.vram_used_mb:.2f}", f"{metrics.vram_allocated_mb:.2f}",
                f"{metrics.grad_norm:.6f}",
                metrics.timestamp
            ])
    
    def _write_batch_tensorboard(self, metrics: BatchMetrics):
        """Write batch metrics to TensorBoard."""
        step = metrics.global_step
        
        self.tb_writer.add_scalar('batch/loss_total', metrics.total_loss, step)
        self.tb_writer.add_scalar('batch/loss_ctc', metrics.ctc_loss, step)
        self.tb_writer.add_scalar('batch/loss_seq', metrics.seq_loss, step)
        self.tb_writer.add_scalar('batch/loss_dc', metrics.dc_loss, step)
        
        self.tb_writer.add_scalar('batch/compression_ratio', metrics.compression_ratio, step)
        self.tb_writer.add_scalar('batch/num_chunks', metrics.num_chunks, step)
        self.tb_writer.add_scalar('batch/boundary_prob_mean', metrics.boundary_prob_mean, step)
        self.tb_writer.add_scalar('batch/target_N', metrics.target_N, step)
        
        self.tb_writer.add_scalar('batch/time_ms', metrics.batch_time_ms, step)
        self.tb_writer.add_scalar('batch/vram_mb', metrics.vram_used_mb, step)
        self.tb_writer.add_scalar('batch/grad_norm', metrics.grad_norm, step)
    
    # =========================================================================
    # Epoch-Level Logging
    # =========================================================================
    
    def start_epoch(self, epoch: int):
        """Mark start of epoch for timing."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self._reset_epoch_accumulators()
        self.gpu_monitor.reset_peak_memory()
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Starting Epoch {epoch}")
        self.logger.info(f"{'=' * 60}")
    
    def log_epoch(
        self,
        epoch: int,
        valid_loss: float,
        valid_acc: float,
        valid_wer: Optional[float],
        valid_cer: Optional[float],
        learning_rate: float,
    ):
        """Log aggregated metrics at end of epoch."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        
        import statistics
        
        def safe_mean(lst):
            return statistics.mean(lst) if lst else 0.0
        
        def safe_stdev(lst):
            return statistics.stdev(lst) if len(lst) > 1 else 0.0
        
        total_audio_sec = sum(self.acc['audio_duration'])
        avg_batch_time_ms = safe_mean(self.acc['batch_time'])
        
        rtf = epoch_time / total_audio_sec if total_audio_sec > 0 else 0.0
        
        target_N_mean = safe_mean(self.acc['target_N']) if self.acc['target_N'] else 2.0
        
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=safe_mean(self.acc['total_loss']),
            train_ctc_loss=safe_mean(self.acc['ctc_loss']),
            train_seq_loss=safe_mean(self.acc['seq_loss']),
            train_dc_loss=safe_mean(self.acc['dc_loss']),
            valid_loss=valid_loss,
            valid_acc=valid_acc,
            valid_wer=valid_wer,
            valid_cer=valid_cer,
            avg_compression_ratio=safe_mean(self.acc['compression_ratio']),
            target_compression_ratio=1.0 / target_N_mean,
            compression_ratio_std=safe_stdev(self.acc['compression_ratio']),
            avg_num_chunks=safe_mean(self.acc['num_chunks']),
            avg_boundary_prob=safe_mean(self.acc['boundary_prob']),
            learning_rate=learning_rate,
            epoch_time_sec=epoch_time,
            avg_batch_time_ms=avg_batch_time_ms,
            total_audio_sec=total_audio_sec,
            rtf=rtf,
            peak_vram_mb=self.gpu_monitor.get_peak_memory_mb(),
            avg_vram_mb=safe_mean(self.acc['vram']),
            avg_grad_norm=safe_mean(self.acc['grad_norm']),
            max_grad_norm=max(self.acc['grad_norm']) if self.acc['grad_norm'] else 0.0,
            timestamp=datetime.now().isoformat(),
        )
        
        self.epoch_metrics_history.append(metrics)
        
        self._write_epoch_csv(metrics)
        
        if self.tb_writer:
            self._write_epoch_tensorboard(metrics)
        
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Epoch {epoch} Summary")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Training:")
        self.logger.info(f"  Loss: {metrics.train_loss:.4f} (CTC={metrics.train_ctc_loss:.4f}, Seq={metrics.train_seq_loss:.4f}, DC={metrics.train_dc_loss:.4f})")
        self.logger.info(f"Validation:")
        self.logger.info(f"  Loss: {metrics.valid_loss:.4f}, ACC: {metrics.valid_acc:.4f}")
        if valid_wer is not None:
            self.logger.info(f"  WER: {metrics.valid_wer:.2f}%")
        self.logger.info(f"Dynamic Chunking:")
        self.logger.info(f"  Compression: {metrics.avg_compression_ratio:.3f} ± {metrics.compression_ratio_std:.3f} (target: {metrics.target_compression_ratio:.3f})")
        self.logger.info(f"  Avg chunks: {metrics.avg_num_chunks:.1f}, Boundary prob: {metrics.avg_boundary_prob:.3f}")
        self.logger.info(f"Performance:")
        if metrics.rtf > 0:
            self.logger.info(f"  RTF: {metrics.rtf:.4f} ({1/metrics.rtf:.1f}x realtime)")
        else:
            self.logger.info(f"  RTF: N/A")
        self.logger.info(f"  Epoch time: {metrics.epoch_time_sec:.1f}s, Avg batch: {metrics.avg_batch_time_ms:.1f}ms")
        self.logger.info(f"  Audio processed: {metrics.total_audio_sec/3600:.2f} hours")
        self.logger.info(f"GPU:")
        self.logger.info(f"  Peak VRAM: {metrics.peak_vram_mb:.0f}MB, Avg VRAM: {metrics.avg_vram_mb:.0f}MB")
        self.logger.info(f"Optimization:")
        self.logger.info(f"  LR: {metrics.learning_rate:.6f}, Grad norm: {metrics.avg_grad_norm:.4f} (max: {metrics.max_grad_norm:.4f})")
        self.logger.info(f"{'=' * 60}\n")
    
    def _write_epoch_csv(self, metrics: EpochMetrics):
        """Append epoch metrics to CSV."""
        with open(self.epoch_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.epoch,
                f"{metrics.train_loss:.6f}", f"{metrics.train_ctc_loss:.6f}",
                f"{metrics.train_seq_loss:.6f}", f"{metrics.train_dc_loss:.6f}",
                f"{metrics.valid_loss:.6f}", f"{metrics.valid_acc:.6f}",
                f"{metrics.valid_wer:.4f}" if metrics.valid_wer else "",
                f"{metrics.valid_cer:.4f}" if metrics.valid_cer else "",
                f"{metrics.avg_compression_ratio:.6f}", f"{metrics.target_compression_ratio:.6f}",
                f"{metrics.compression_ratio_std:.6f}",
                f"{metrics.avg_num_chunks:.2f}", f"{metrics.avg_boundary_prob:.6f}",
                f"{metrics.learning_rate:.8f}",
                f"{metrics.epoch_time_sec:.2f}", f"{metrics.avg_batch_time_ms:.2f}",
                f"{metrics.total_audio_sec:.2f}", f"{metrics.rtf:.6f}",
                f"{metrics.peak_vram_mb:.2f}", f"{metrics.avg_vram_mb:.2f}",
                f"{metrics.avg_grad_norm:.6f}", f"{metrics.max_grad_norm:.6f}",
                metrics.timestamp
            ])
    
    def _write_epoch_tensorboard(self, metrics: EpochMetrics):
        """Write epoch metrics to TensorBoard."""
        epoch = metrics.epoch
        
        self.tb_writer.add_scalar('epoch/train_loss', metrics.train_loss, epoch)
        self.tb_writer.add_scalar('epoch/train_ctc_loss', metrics.train_ctc_loss, epoch)
        self.tb_writer.add_scalar('epoch/train_seq_loss', metrics.train_seq_loss, epoch)
        self.tb_writer.add_scalar('epoch/train_dc_loss', metrics.train_dc_loss, epoch)
        
        self.tb_writer.add_scalar('epoch/valid_loss', metrics.valid_loss, epoch)
        self.tb_writer.add_scalar('epoch/valid_acc', metrics.valid_acc, epoch)
        if metrics.valid_wer is not None:
            self.tb_writer.add_scalar('epoch/valid_wer', metrics.valid_wer, epoch)
        
        self.tb_writer.add_scalar('epoch/compression_ratio', metrics.avg_compression_ratio, epoch)
        self.tb_writer.add_scalar('epoch/compression_ratio_std', metrics.compression_ratio_std, epoch)
        self.tb_writer.add_scalar('epoch/target_compression', metrics.target_compression_ratio, epoch)
        
        self.tb_writer.add_scalar('epoch/rtf', metrics.rtf, epoch)
        self.tb_writer.add_scalar('epoch/epoch_time_sec', metrics.epoch_time_sec, epoch)
        
        self.tb_writer.add_scalar('epoch/peak_vram_mb', metrics.peak_vram_mb, epoch)
        
        self.tb_writer.add_scalar('epoch/learning_rate', metrics.learning_rate, epoch)
        self.tb_writer.add_scalar('epoch/grad_norm', metrics.avg_grad_norm, epoch)
    
    # =========================================================================
    # Inference Logging
    # =========================================================================
    
    def log_inference(
        self,
        dataset_name: str,
        num_samples: int,
        wer: float,
        cer: float,
        total_inference_time: float,
        total_audio_duration: float,
        avg_compression_ratio: float,
    ):
        """Log inference/evaluation results."""
        rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0
        avg_latency = (total_inference_time / num_samples * 1000) if num_samples > 0 else 0.0
        
        metrics = InferenceMetrics(
            dataset_name=dataset_name,
            num_samples=num_samples,
            wer=wer,
            cer=cer,
            total_inference_time_sec=total_inference_time,
            total_audio_duration_sec=total_audio_duration,
            rtf=rtf,
            avg_latency_ms=avg_latency,
            avg_compression_ratio=avg_compression_ratio,
            peak_vram_mb=self.gpu_monitor.get_peak_memory_mb(),
            timestamp=datetime.now().isoformat(),
        )
        
        with open(self.inference_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.path.getsize(self.inference_csv_path) == 0:
                writer.writerow(list(asdict(metrics).keys()))
            writer.writerow(list(asdict(metrics).values()))
        
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Inference Results: {dataset_name}")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Samples: {num_samples}")
        self.logger.info(f"WER: {wer:.2f}%, CER: {cer:.2f}%")
        if rtf > 0:
            self.logger.info(f"RTF: {rtf:.4f} ({1/rtf:.1f}x realtime)")
        else:
            self.logger.info(f"RTF: N/A")
        self.logger.info(f"Avg latency: {avg_latency:.1f}ms per sample")
        self.logger.info(f"Compression ratio: {avg_compression_ratio:.3f}")
        self.logger.info(f"Peak VRAM: {metrics.peak_vram_mb:.0f}MB")
        self.logger.info(f"{'=' * 60}\n")
        
        return metrics
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def save_experiment_config(self, config: Dict[str, Any]):
        """Save experiment configuration to JSON."""
        config_path = self.log_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        self.logger.info(f"Config saved to {config_path}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report of the experiment."""
        if not self.epoch_metrics_history:
            return "No epochs completed yet."
        
        best_wer_epoch = None
        for e in self.epoch_metrics_history:
            if e.valid_wer is not None:
                if best_wer_epoch is None or e.valid_wer < best_wer_epoch.valid_wer:
                    best_wer_epoch = e
        
        last_epoch = self.epoch_metrics_history[-1]
        
        total_time_hrs = sum(e.epoch_time_sec for e in self.epoch_metrics_history) / 3600
        total_audio_hrs = sum(e.total_audio_sec for e in self.epoch_metrics_history) / 3600
        
        report = f"""
{'=' * 70}
H-MAMBA ASR EXPERIMENT SUMMARY
Experiment: {self.experiment_name}
Generated: {datetime.now().isoformat()}
{'=' * 70}

TRAINING OVERVIEW
-----------------
Total epochs: {len(self.epoch_metrics_history)}
Total training time: {total_time_hrs:.2f} hours
Total audio processed: {total_audio_hrs:.2f} hours

"""
        if best_wer_epoch:
            report += f"""BEST RESULTS (Epoch {best_wer_epoch.epoch})
------------------------------------
WER: {best_wer_epoch.valid_wer:.2f}%
Validation ACC: {best_wer_epoch.valid_acc:.4f}
Compression ratio: {best_wer_epoch.avg_compression_ratio:.3f}

"""
        
        report += f"""FINAL EPOCH ({last_epoch.epoch})
--------------------------------
Train loss: {last_epoch.train_loss:.4f}
Valid loss: {last_epoch.valid_loss:.4f}
"""
        if last_epoch.valid_wer:
            report += f"WER: {last_epoch.valid_wer:.2f}%\n"
        
        report += f"""
DYNAMIC CHUNKING ANALYSIS
-------------------------
Final compression ratio: {last_epoch.avg_compression_ratio:.3f} (target: {last_epoch.target_compression_ratio:.3f})
Compression stability (std): {last_epoch.compression_ratio_std:.4f}
Average boundary probability: {last_epoch.avg_boundary_prob:.3f}

PERFORMANCE METRICS
-------------------
Average RTF: {sum(e.rtf for e in self.epoch_metrics_history)/len(self.epoch_metrics_history):.4f}
Peak VRAM: {max(e.peak_vram_mb for e in self.epoch_metrics_history):.0f}MB
Average VRAM: {sum(e.avg_vram_mb for e in self.epoch_metrics_history)/len(self.epoch_metrics_history):.0f}MB

{'=' * 70}
"""
        
        report_path = self.log_dir / "experiment_summary.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def close(self):
        """Close logger and cleanup."""
        if self.tb_writer:
            self.tb_writer.close()
        
        report = self.generate_summary_report()
        self.logger.info(report)
        self.logger.info("Logger closed.")


# =============================================================================
# Integration Helper for train_s2s_hmamba.py
# =============================================================================

class TrainingContext:
    """Context manager for timing training phases."""
    
    def __init__(self, logger: HMambaLogger):
        self.logger = logger
        self.timers = {}
    
    def time_phase(self, name: str) -> Timer:
        """Create a timer for a training phase."""
        timer = Timer(cuda_sync=True)
        self.timers[name] = timer
        return timer
    
    def get_timing_dict(self) -> Dict[str, float]:
        """Get timing results as dict."""
        return {name: timer.elapsed() for name, timer in self.timers.items()}


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing HMambaLogger...")
    
    logger = HMambaLogger(
        log_dir="/tmp/hmamba_test_logs",
        experiment_name="test_run",
        use_tensorboard=TENSORBOARD_AVAILABLE,
        log_every_n_batches=5,
    )
    
    logger.start_epoch(1)
    
    for batch_idx in range(20):
        logger.log_batch(
            epoch=1,
            batch_idx=batch_idx,
            losses={'total': 100 - batch_idx, 'ctc': 50 - batch_idx/2, 'seq': 50 - batch_idx/2, 'dc': 1.0},
            dc_stats={'compression_ratio': 0.1 + batch_idx * 0.02, 'num_chunks': 10, 'avg_chunk_size': 5.0, 'boundary_prob_mean': 0.4},
            timing={'batch': 100, 'forward': 60, 'backward': 40, 'data_load': 10},
            grad_norm=1.0,
            audio_duration_sec=16.0,
            target_N=2.0,
        )
    
    logger.log_epoch(
        epoch=1,
        valid_loss=50.0,
        valid_acc=0.75,
        valid_wer=15.5,
        valid_cer=5.2,
        learning_rate=0.001,
    )
    
    report = logger.generate_summary_report()
    print(report)
    
    logger.close()
    print("\n✅ Logger test passed!")