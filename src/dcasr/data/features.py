"""Acoustic frontend: 80-d log-Mel @ 100 Hz, global CMVN, SpecAugment [14].

Locks the tensor contract every downstream module consumes:
    waveform [B, N] @ 16 kHz  ->  features [B, T, 80],  T = 1 + (N - 400) // 160.
STFT uses center=False (no edge padding), so every frame covers real samples
only — features of a zero-padded batch are bit-identical to per-utterance
features on all valid frames (masked by the returned lengths). CMVN stats
accumulate in float64 (train-960 is ~3e8 frames; fp32 sums drift).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

SAMPLE_RATE = 16000
N_MELS = 80
WIN_LENGTH = 400          # 25 ms
HOP_LENGTH = 160          # 10 ms -> 100 Hz
LOG_FLOOR = 1e-10


class LogMelFrontend(nn.Module):
    """waveform [B, N] or [N] -> (feats [B, T, 80], feat_lengths [B])."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, n_mels: int = N_MELS,
                 win_length: int = WIN_LENGTH, hop_length: int = HOP_LENGTH):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=win_length, win_length=win_length,
            hop_length=hop_length, f_min=0.0, f_max=sample_rate / 2.0,
            n_mels=n_mels, power=2.0, center=False)
        logger.debug("LogMelFrontend(sr=%d, n_mels=%d, win=%d, hop=%d)",
                     sample_rate, n_mels, win_length, hop_length)

    def frame_count(self, num_samples: torch.Tensor) -> torch.Tensor:
        return ((num_samples - self.win_length) // self.hop_length + 1).clamp_min(0)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor | None = None):
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        if wave.dim() != 2:
            raise ValueError(f"expected waveform [B, N] or [N], got {tuple(wave.shape)}")
        if wave.shape[-1] < self.win_length:
            raise ValueError(f"waveform ({wave.shape[-1]} samples) shorter than one "
                             f"window ({self.win_length})")
        wave = wave.float()                                       # soundfile yields fp64
        if lengths is None:
            lengths = torch.full((wave.shape[0],), wave.shape[-1],
                                 dtype=torch.long, device=wave.device)
        lengths = lengths.to(device=wave.device, dtype=torch.long)
        if wave.shape[0] == 0:                                    # cuFFT rejects B=0
            T = (wave.shape[-1] - self.win_length) // self.hop_length + 1
            return wave.new_zeros(0, T, self.n_mels), lengths
        mel = self.melspec(wave)                                  # [B, 80, T]
        feats = torch.log(mel.clamp_min(LOG_FLOOR)).transpose(1, 2).contiguous()
        return feats, self.frame_count(lengths)


class GlobalCMVN(nn.Module):
    """Apply frozen global mean/variance normalisation: (x - mean) / std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("istd", 1.0 / std.float())

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return (feats - self.mean) * self.istd

    @classmethod
    def load(cls, path: str | Path) -> "GlobalCMVN":
        stats = torch.load(path, map_location="cpu", weights_only=True)
        logger.debug("GlobalCMVN loaded from %s (count=%d)", path, stats["count"])
        return cls(stats["mean"], stats["std"])


class CMVNAccumulator:
    """Streaming fp64 mean/var stats over valid (unpadded) frames of train-960."""

    def __init__(self, n_mels: int = N_MELS):
        self.sum = torch.zeros(n_mels, dtype=torch.float64, device="cpu")
        self.sumsq = torch.zeros(n_mels, dtype=torch.float64, device="cpu")
        self.count = 0

    def update(self, feats: torch.Tensor, lengths: torch.Tensor | None = None) -> None:
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if lengths is not None:
            valid = torch.arange(feats.shape[1], device=feats.device) < lengths[:, None]
            feats = feats[valid]                                  # [K, n_mels]
        else:
            feats = feats.reshape(-1, feats.shape[-1])
        x = feats.double()
        self.sum += x.sum(dim=0).cpu()
        self.sumsq += x.pow(2).sum(dim=0).cpu()
        self.count += x.shape[0]

    def finalize(self, var_floor: float = 1e-8) -> dict:
        if self.count == 0:
            raise RuntimeError("no frames accumulated")
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.pow(2)).clamp_min(var_floor)
        return {"mean": mean.float(), "std": var.sqrt().float(), "count": self.count}

    def save(self, path: str | Path) -> dict:
        stats = self.finalize()
        torch.save(stats, path)
        logger.info("CMVN stats saved to %s (count=%d)", path, stats["count"])
        return stats


class SpecAugment(nn.Module):
    """Frequency + time masking (SpecAugment LD policy defaults). Train-mode only;
    masks fill with 0.0 (= the global mean after CMVN) and never start inside padding.
    """

    def __init__(self, freq_masks: int = 2, freq_width: int = 27,
                 time_masks: int = 2, time_width: int = 100):
        super().__init__()
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks, self.time_width = time_masks, time_width
        logger.debug("SpecAugment(F=%dx%d, T=%dx%d)",
                     freq_masks, freq_width, time_masks, time_width)

    @staticmethod
    def _mask(size: int, widths: torch.Tensor, max_start: torch.Tensor):
        starts = (torch.rand_like(widths.float()) * (max_start + 1).float()).long()
        pos = torch.arange(size, device=widths.device)
        hit = (pos >= starts[..., None]) & (pos < (starts + widths)[..., None])
        return hit.any(dim=1)                                     # [B, size]

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None):
        if not self.training:
            return feats
        B, T, F = feats.shape
        dev = feats.device
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=dev)
        if self.freq_masks > 0:
            w = torch.randint(0, self.freq_width + 1, (B, self.freq_masks), device=dev)
            fmask = self._mask(F, w, (F - w).clamp_min(0))
            feats = feats.masked_fill(fmask[:, None, :], 0.0)
        if self.time_masks > 0:
            w = torch.randint(0, self.time_width + 1, (B, self.time_masks), device=dev)
            w = torch.minimum(w, lengths[:, None])
            tmask = self._mask(T, w, (lengths[:, None] - w).clamp_min(0))
            feats = feats.masked_fill(tmask[:, :, None], 0.0)
        return feats
